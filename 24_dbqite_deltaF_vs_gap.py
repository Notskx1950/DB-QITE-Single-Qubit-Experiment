from typing import Optional, List, Literal
import numpy as np
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *


# =========================
# Your theory functions
# =========================
from dbqite_theory_sim import (
    simulate_dbqite_single_qubit,
    make_H,
    ground_state,
    hamiltonian_axis,
    bloch_vector,
)


# =========================
# Parameters (Qualibrate)
# =========================
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q4"]

    # Figure-1 specific
    F0_target: float = 0.10
    S_GAP: float = 0.05

    DBQITE_METHOD: Literal["GC", "HOPF"] = "GC"

    # Two "problem Hamiltonians" for the figure
    DELTA_A: float = 1.0
    OMEGA_A: float = 0.7
    DELTA_B: float = 0.4
    OMEGA_B: float = 0.2

    num_averages: int = 9000  # shots per case

    # Reset behavior
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    thermal_reset_time_ns: int = 460000  # set >= ~5*T1 if you know T1
    depletion_time_ns: int = 4000  # resonator depletion wait

    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 200


node = QualibrationNode(name="DBQITE_Fig1_DeltaF_vs_Gap", parameters=Parameters())


# =========================
# Initialize QuAM + QOP
# =========================
u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# =========================
# Helper math
# =========================
psi_zero = np.array([1.0, 0.0], dtype=complex)


def Ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def spectral_gap(delta, omega):
    return 2.0 * np.sqrt(delta**2 + omega**2)


def theta_ground(delta, omega):
    # Ground state direction for H = (delta/2) Z + (omega/2) X
    return wrap_to_pi(np.arctan2(omega, delta) + np.pi)


def theta_prep_for_target_F0(theta_g, F0):
    # For Ry(theta)|0>, fidelity to Ry(theta_g)|0> is cos^2((theta-theta_g)/2)
    dtheta = 2.0 * np.arccos(np.sqrt(F0))
    return wrap_to_pi(theta_g + dtheta)


def theta_from_state_on_ry_meridian(psi):
    a = psi[0]
    ph = np.angle(a) if np.abs(a) > 1e-12 else 0.0
    psi2 = psi * np.exp(-1j * ph)
    a = np.real_if_close(psi2[0])
    b = np.real_if_close(psi2[1])
    return wrap_to_pi(2.0 * np.arctan2(np.real(b), np.real(a)))


def compute_theta_step_from_sim(delta, omega, s, theta_prep):
    psi0 = Ry(theta_prep) @ psi_zero
    Es, Vs, Fs, states = simulate_dbqite_single_qubit(
        delta, omega, [s], psi0, method=node.parameters.DBQITE_METHOD, return_states=True
    )
    th0 = theta_from_state_on_ry_meridian(states[0])
    th1 = theta_from_state_on_ry_meridian(states[1])
    return wrap_to_pi(th1 - th0)


def correct_p1_spam(p1_raw, p1_when_0, p1_when_1):
    """
    Correct P1 measurement for SPAM errors.
    """
    denom = p1_when_1 - p1_when_0
    if abs(denom) < 0.01:
        print("WARNING: Readout contrast too low!")
        return p1_raw
    p1_corrected = (p1_raw - p1_when_0) / denom
    return np.clip(p1_corrected, 0, 1)


# =========================
# Run experiment for one qubit
# =========================
def run_for_one_qubit(qubit):
    """
    Run Figure 1 experiment using QuAM macros (same style as working debug code).
    """
    qmm = machine.connect()

    results = []

    for label, (delta, omega) in [
        ("A", (node.parameters.DELTA_A, node.parameters.OMEGA_A)),
        ("B", (node.parameters.DELTA_B, node.parameters.OMEGA_B)),
    ]:
        Hn = 0.5 * np.sqrt(delta**2 + omega**2)
        gap = spectral_gap(delta, omega)
        s = gap / (12.0 * (Hn**3))

        # Compute angles
        theta_g = theta_ground(delta, omega)
        theta_prep = theta_prep_for_target_F0(theta_g, node.parameters.F0_target)
        theta_step = compute_theta_step_from_sim(delta, omega, s, theta_prep)

        # Convert to y180 amplitude scaling
        a_prep = float(theta_prep / np.pi)
        a_overlap = float((-theta_g) / np.pi)
        a_step = float(theta_step / np.pi)

        print(f"\n[{qubit.name}] Case {label}:")
        print(f"  delta={delta}, omega={omega}")
        print(f"  gap={gap:.4f}, s={s:.6f}")
        print(f"  theta_g={np.degrees(theta_g):.2f}°, theta_prep={np.degrees(theta_prep):.2f}°")
        print(f"  theta_step={np.degrees(theta_step):.2f}°")
        print(f"  a_prep={a_prep:.4f}, a_overlap={a_overlap:.4f}, a_step={a_step:.4f}")

        n_avg = node.parameters.num_averages
        reset_type = node.parameters.reset_type_thermal_or_active

        # Build QUA program using QuAM macros (like debug code)
        with program() as prog:
            I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
            state = declare(int)
            st = declare_stream()

            # Also measure SPAM calibration
            st_spam = declare_stream()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ---- SPAM calibration: measure |0⟩ ----
                if reset_type == "active":
                    active_reset(qubit, "readout")
                else:
                    qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
                reset_frame(qubit.xy.name)
                qubit.align()
                readout_state(qubit, state)
                save(state, st_spam)

                # ---- SPAM calibration: measure |1⟩ via x180 ----
                if reset_type == "active":
                    active_reset(qubit, "readout")
                else:
                    qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
                reset_frame(qubit.xy.name)
                qubit.align()
                qubit.xy.play("x180")
                qubit.align()
                readout_state(qubit, state)
                save(state, st_spam)

                # ---- Case 0: prep -> overlap -> measure ----
                if reset_type == "active":
                    active_reset(qubit, "readout")
                else:
                    qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
                reset_frame(qubit.xy.name)
                qubit.align()

                # Prep: Ry(theta_prep)|0⟩
                qubit.xy.play("y180", amplitude_scale=a_prep)
                qubit.align()

                # Overlap rotation: map |g⟩ -> |0⟩
                qubit.xy.play("y180", amplitude_scale=a_overlap)
                qubit.align()

                readout_state(qubit, state)
                save(state, st)

                # ---- Case 1: prep -> step -> overlap -> measure ----
                if reset_type == "active":
                    active_reset(qubit, "readout")
                else:
                    qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
                reset_frame(qubit.xy.name)
                qubit.align()

                # Prep: Ry(theta_prep)|0⟩
                qubit.xy.play("y180", amplitude_scale=a_prep)
                qubit.align()

                # DB-QITE step: Ry(theta_step)
                qubit.xy.play("y180", amplitude_scale=a_step)
                qubit.align()

                # Overlap rotation: map |g⟩ -> |0⟩
                qubit.xy.play("y180", amplitude_scale=a_overlap)
                qubit.align()

                readout_state(qubit, state)
                save(state, st)

            with stream_processing():
                n_st.save("n")
                st_spam.buffer(2).average().save("p1_spam")  # [P1|0⟩, P1|1⟩]
                st.buffer(2).average().save("p1_cases")  # [P1_case0, P1_case1]

        # Execute
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(prog)
            results_fetcher = fetching_tool(job, ["n"], mode="live")
            while results_fetcher.is_processing():
                n_done = results_fetcher.fetch_all()[0]
                progress_counter(n_done, n_avg, start_time=results_fetcher.start_time)

            job.result_handles.wait_for_all_values()
            p1_spam = job.result_handles.get("p1_spam").fetch_all()
            p1_cases = job.result_handles.get("p1_cases").fetch_all()

        p1_when_0, p1_when_1 = float(p1_spam[0]), float(p1_spam[1])
        p1_0_raw, p1_1_raw = float(p1_cases[0]), float(p1_cases[1])

        contrast = p1_when_1 - p1_when_0
        print(f"  SPAM: P1|0⟩={p1_when_0:.4f}, P1|1⟩={p1_when_1:.4f}, contrast={contrast:.4f}")

        # Apply SPAM correction
        p1_0_corr = correct_p1_spam(p1_0_raw, p1_when_0, p1_when_1)
        p1_1_corr = correct_p1_spam(p1_1_raw, p1_when_0, p1_when_1)

        # Compute fidelities
        # F = 1 - P1 (probability of measuring |0⟩ after overlap rotation)
        F0_raw = 1.0 - p1_0_raw
        F1_raw = 1.0 - p1_1_raw
        dF_raw = F1_raw - F0_raw

        F0_corr = 1.0 - p1_0_corr
        F1_corr = 1.0 - p1_1_corr
        dF_corr = F1_corr - F0_corr

        print(f"  Raw:  F0={F0_raw:.4f}, F1={F1_raw:.4f}, ΔF={dF_raw:.4f}")
        print(f"  Corr: F0={F0_corr:.4f}, F1={F1_corr:.4f}, ΔF={dF_corr:.4f}")

        results.append({
            "label": label,
            "delta": delta,
            "omega": omega,
            "gap": gap,
            "s": s,
            "theta_g": theta_g,
            "theta_prep": theta_prep,
            "theta_step": theta_step,
            "a_prep": a_prep,
            "a_overlap": a_overlap,
            "a_step": a_step,
            "p1_when_0": p1_when_0,
            "p1_when_1": p1_when_1,
            "contrast": contrast,
            "p1_0_raw": p1_0_raw,
            "p1_1_raw": p1_1_raw,
            "p1_0_corr": p1_0_corr,
            "p1_1_corr": p1_1_corr,
            "F0_raw": F0_raw,
            "F1_raw": F1_raw,
            "dF_raw": dF_raw,
            "F0_corr": F0_corr,
            "F1_corr": F1_corr,
            "dF_corr": dF_corr,
        })

    return results


# =========================
# Main execution
# =========================
all_results = {}
for q in qubits[:1]:
    all_results[q.name] = run_for_one_qubit(q)

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for qname, res_list in all_results.items():
    print(f"\n{qname}:")
    print(f"  {'Case':<6} {'Gap':<10} {'F0_corr':<10} {'F1_corr':<10} {'ΔF_corr':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in res_list:
        print(f"  {r['label']:<6} {r['gap']:<10.4f} {r['F0_corr']:<10.4f} {r['F1_corr']:<10.4f} {r['dF_corr']:<10.4f}")

# Plot (Figure 1 style)
for qname, res_list in all_results.items():
    gaps = np.array([r["gap"] for r in res_list])
    dFs_raw = np.array([r["dF_raw"] for r in res_list])
    dFs_corr = np.array([r["dF_corr"] for r in res_list])

    plt.figure(figsize=(8, 5))
    plt.plot(gaps, dFs_raw, "o--", label="Raw", alpha=0.7)
    plt.plot(gaps, dFs_corr, "s-", label="SPAM corrected")
    plt.xlabel("Spectral gap")
    plt.ylabel(r"$\Delta F = F_1 - F_0$")
    plt.title(f"DB-QITE fidelity gain vs gap (hardware) [{qname}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

node.results = {"fig1_results": all_results}
node.save()