from typing import Optional, List, Literal
import numpy as np
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
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

    DBQITE_METHOD: Literal["GC", "HOPF"] = "GC"

    # Hamiltonian parameters
    DELTA: float = 1.0
    OMEGA: float = 0.7

    # Step size for single DB-QITE step
    S_GAP: float = 0.02

    # Initial state scan range (theta for Ry(theta)|0>)
    THETA_MIN: float = 0.0
    THETA_MAX: float = np.pi
    N_THETA_POINTS: int = 15

    # shots per point
    num_averages: int = 3000

    # Reset behavior
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    thermal_reset_time_ns: int = 20000
    depletion_time_ns: int = 4000

    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 200


node = QualibrationNode(name="DBQITE_FidelityGain_vs_InitialFidelity", parameters=Parameters())

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


def theta_from_state_on_ry_meridian(psi):
    a = psi[0]
    ph = np.angle(a) if np.abs(a) > 1e-12 else 0.0
    psi2 = psi * np.exp(-1j * ph)
    a = np.real_if_close(psi2[0])
    b = np.real_if_close(psi2[1])
    return wrap_to_pi(2.0 * np.arctan2(np.real(b), np.real(a)))


def theta_ground_state(delta, omega):
    """
    For H = delta*Z + omega*X (or similar), the ground state direction.
    Returns theta such that Ry(theta)|0> ~ |gs>.
    """
    H = make_H(delta, omega)
    _, psi_gs = ground_state(H)
    return theta_from_state_on_ry_meridian(psi_gs)


def compute_theta_after_one_step(delta, omega, s, theta_init, method="GC"):
    """
    Simulate one DB-QITE step and return the resulting state's theta.
    """
    psi0 = Ry(theta_init) @ psi_zero

    Es, Vs, Fs, states = simulate_dbqite_single_qubit(
        delta, omega, [s], psi0, method=method, return_states=True
    )

    # states[0] = initial, states[1] = after one step
    theta_after = theta_from_state_on_ry_meridian(states[1])
    return theta_after, Fs[0], Fs[1]


def theta_overlap_with_gs(delta, omega):
    """
    Return theta_gs such that Ry(-theta_gs) rotates ground state onto |0>.
    Measuring P(|0>) after this rotation gives fidelity with ground state.
    """
    theta_gs = theta_ground_state(delta, omega)
    return -theta_gs


# =========================
# Run single fidelity measurement (debug style)
# =========================
def run_fidelity_before_after_measurement(qubit, a_prep, a_step, a_overlap, n_avg, reset_type):
    """
    Measures fidelity before (F0) and after (F1) one DB-QITE step using QuAM macros.
    
    Returns P1 for two cases:
      case0: prep -> overlap -> measure  (gives F0 = 1 - P1)
      case1: prep -> step -> overlap -> measure  (gives F1 = 1 - P1)
    """
    qmm = machine.connect()

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # -------- case 0: fidelity before step (F0) --------
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()

            # Prepare initial state: Ry(theta_init)|0⟩
            qubit.xy.play("y180", amplitude_scale=a_prep)
            qubit.align()

            # Rotate to overlap measurement basis: Ry(-theta_gs)
            qubit.xy.play("y180", amplitude_scale=a_overlap)
            qubit.align()

            # Measure
            readout_state(qubit, state)
            save(state, st)

            # -------- case 1: fidelity after one step (F1) --------
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()

            # Prepare initial state: Ry(theta_init)|0⟩
            qubit.xy.play("y180", amplitude_scale=a_prep)
            qubit.align()

            # Apply one DB-QITE step: Ry(delta_theta)
            qubit.xy.play("y180", amplitude_scale=a_step)
            qubit.align()

            # Rotate to overlap measurement basis: Ry(-theta_gs)
            qubit.xy.play("y180", amplitude_scale=a_overlap)
            qubit.align()

            # Measure
            readout_state(qubit, state)
            save(state, st)

        with stream_processing():
            n_st.save("n")
            st.buffer(2).average().save("p1_cases")  # [P1_before, P1_after]

    # Execute
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog)
        results_fetcher = fetching_tool(job, ["n"], mode="live")
        while results_fetcher.is_processing():
            n_done = results_fetcher.fetch_all()[0]
            progress_counter(n_done, n_avg, start_time=results_fetcher.start_time)

        job.result_handles.wait_for_all_values()
        p1_cases = job.result_handles.get("p1_cases").fetch_all()

    p1_before, p1_after = float(p1_cases[0]), float(p1_cases[1])

    return p1_before, p1_after


# =========================
# Run experiment
# =========================
def run_fidelity_gain_vs_initial(qubit):
    delta = node.parameters.DELTA
    omega = node.parameters.OMEGA
    s_gap = node.parameters.S_GAP
    method = node.parameters.DBQITE_METHOD
    n_avg = node.parameters.num_averages
    reset_type = node.parameters.reset_type_thermal_or_active

    # Overlap rotation angle (to measure fidelity with GS)
    a_overlap = float(theta_overlap_with_gs(delta, omega) / np.pi)

    # Theta values to scan
    theta_values = np.linspace(
        node.parameters.THETA_MIN,
        node.parameters.THETA_MAX,
        node.parameters.N_THETA_POINTS,
    )

    print(f"\n{'='*60}")
    print(f"DB-QITE Fidelity Gain vs Initial Fidelity Experiment")
    print(f"{'='*60}")
    print(f"Qubit: {qubit.name}")
    print(f"Hamiltonian: delta={delta}, omega={omega}")
    print(f"Step size s = {s_gap}")
    print(f"a_overlap = {a_overlap:.4f}")
    print(f"Scanning {len(theta_values)} initial angles from {node.parameters.THETA_MIN:.3f} to {node.parameters.THETA_MAX:.3f}")
    print(f"Reset type: {reset_type}")
    print(f"{'='*60}\n")

    # Theory predictions
    F0_theory = np.zeros_like(theta_values)
    F1_theory = np.zeros_like(theta_values)
    theta_after_theory = np.zeros_like(theta_values)

    for i, th in enumerate(theta_values):
        theta_after, f0, f1 = compute_theta_after_one_step(delta, omega, s_gap, th, method=method)
        F0_theory[i] = f0
        F1_theory[i] = f1
        theta_after_theory[i] = theta_after

    deltaF_theory = F1_theory - F0_theory

    print("Theory predictions:")
    for i, th in enumerate(theta_values[:5]):
        print(f"  theta={th:.3f}: F0={F0_theory[i]:.4f}, F1={F1_theory[i]:.4f}, ΔF={deltaF_theory[i]:.4f}")
    if len(theta_values) > 5:
        print(f"  ... ({len(theta_values) - 5} more points)")

    # Hardware measurements
    F0_hw = np.zeros_like(theta_values)
    F1_hw = np.zeros_like(theta_values)

    print("\n--- Hardware measurements ---")
    for i, th in enumerate(theta_values):
        # Initial state prep angle
        a_prep = float(th / np.pi)

        # Compute the incremental step rotation from theory
        theta_after = theta_after_theory[i]
        delta_theta = wrap_to_pi(theta_after - th)
        a_step = float(delta_theta / np.pi)

        print(f"\nPoint {i+1}/{len(theta_values)}: theta={th:.3f} rad = {np.degrees(th):.1f}°")
        print(f"  a_prep={a_prep:.4f}, a_step={a_step:.4f}")

        p1_before, p1_after = run_fidelity_before_after_measurement(
            qubit=qubit,
            a_prep=a_prep,
            a_step=a_step,
            a_overlap=a_overlap,
            n_avg=n_avg,
            reset_type=reset_type,
        )

        # Fidelity = P(|0>) in overlap basis = 1 - P1
        F0_hw[i] = 1.0 - p1_before
        F1_hw[i] = 1.0 - p1_after

        print(
            f"  P1_before={p1_before:.4f}, P1_after={p1_after:.4f}"
        )
        print(
            f"  F0_hw={F0_hw[i]:.4f} (theory={F0_theory[i]:.4f}), "
            f"F1_hw={F1_hw[i]:.4f} (theory={F1_theory[i]:.4f})"
        )

    deltaF_hw = F1_hw - F0_hw

    return {
        "theta_values": theta_values,
        "F0_hw": F0_hw,
        "F1_hw": F1_hw,
        "deltaF_hw": deltaF_hw,
        "F0_theory": F0_theory,
        "F1_theory": F1_theory,
        "deltaF_theory": deltaF_theory,
        "s_gap": s_gap,
    }


# =========================
# Main execution
# =========================
all_results = {}
for q in qubits[:1]:
    all_results[q.name] = run_fidelity_gain_vs_initial(q)

# Plot results
for qname, r in all_results.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Hardware results
    ax1 = axes[0]
    ax1.plot(r["F0_hw"], r["deltaF_hw"], marker="o", label="Hardware")
    ax1.set_xlabel(r"Initial fidelity $F_0$")
    ax1.set_ylabel(r"$\Delta F = F_1 - F_0$")
    ax1.set_title(f"DB-QITE fidelity gain vs initial fidelity (hardware) [{qname}]")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Theory comparison
    ax2 = axes[1]
    ax2.plot(r["F0_theory"], r["deltaF_theory"], marker="o", label="Theory")
    ax2.set_xlabel(r"Initial fidelity $F_0$")
    ax2.set_ylabel(r"$\Delta F = F_1 - F_0$")
    ax2.set_title("DB-QITE fidelity gain vs initial fidelity (theory)")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"dbqite_fidelity_gain_{qname}.png", dpi=150)
    plt.show()

    # Combined plot (hardware vs theory)
    plt.figure(figsize=(8, 6))
    plt.plot(r["F0_hw"], r["deltaF_hw"], "o-", color="blue", label="Hardware")
    plt.plot(r["F0_theory"], r["deltaF_theory"], "s--", color="red", alpha=0.7, label="Theory")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel(r"Initial fidelity $F_0$")
    plt.ylabel(r"$\Delta F = F_1 - F_0$")
    plt.title(f"DB-QITE fidelity gain vs initial fidelity [{qname}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dbqite_fidelity_gain_combined_{qname}.png", dpi=150)
    plt.show()

    # Additional plot: F0 and F1 vs theta
    plt.figure(figsize=(8, 6))
    plt.plot(r["theta_values"], r["F0_hw"], "o-", color="blue", label="$F_0$ (hardware)")
    plt.plot(r["theta_values"], r["F1_hw"], "s-", color="green", label="$F_1$ (hardware)")
    plt.plot(r["theta_values"], r["F0_theory"], "--", color="blue", alpha=0.5, label="$F_0$ (theory)")
    plt.plot(r["theta_values"], r["F1_theory"], "--", color="green", alpha=0.5, label="$F_1$ (theory)")
    plt.xlabel(r"Initial angle $\theta$")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity before/after DB-QITE step vs initial angle [{qname}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dbqite_F0_F1_vs_theta_{qname}.png", dpi=150)
    plt.show()

    # Summary table
    print(f"\n{'='*80}")
    print(f"Summary for {qname}")
    print(f"{'='*80}")
    print(f"{'theta':<10} {'F0_hw':<10} {'F0_th':<10} {'F1_hw':<10} {'F1_th':<10} {'ΔF_hw':<10} {'ΔF_th':<10}")
    print("-" * 80)
    for i, th in enumerate(r["theta_values"]):
        print(
            f"{th:<10.4f} {r['F0_hw'][i]:<10.4f} {r['F0_theory'][i]:<10.4f} "
            f"{r['F1_hw'][i]:<10.4f} {r['F1_theory'][i]:<10.4f} "
            f"{r['deltaF_hw'][i]:<10.4f} {r['deltaF_theory'][i]:<10.4f}"
        )
    print(f"{'='*80}\n")

node.results = {"fidelity_gain_vs_initial": all_results}
node.save()