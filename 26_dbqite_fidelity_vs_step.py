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

    # Step size (fixed for convergence)
    S_CONV: float = 0.02

    # Number of DB-QITE steps
    N_STEPS: int = 5

    # Target initial fidelity for warm start (will find theta giving F0 <= this)
    TARGET_F0_WARM: float = 0.6

    # shots per step
    num_averages: int = 9000

    # Reset behavior
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    thermal_reset_time_ns: int = 200000
    depletion_time_ns: int = 4000

    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 200


node = QualibrationNode(name="DBQITE_Fidelity_vs_Step", parameters=Parameters())

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


def find_warm_start_theta(delta, omega, target_F0_max=0.6):
    """
    Find theta for Ry(theta)|0> that gives F0 <= target_F0_max (closest to it).
    """
    H = make_H(delta, omega)
    _, psi_gs = ground_state(H)

    thetas = np.linspace(0, np.pi, 2001)
    best = (-1, None)
    for th in thetas:
        psi = Ry(th) @ psi_zero
        F0 = np.abs(np.vdot(psi_gs, psi)) ** 2
        if F0 > best[0] and F0 <= target_F0_max:
            best = (F0, th)

    print(f"Warm start: best F0 = {best[0]:.4f}, theta = {best[1]:.4f}")
    return best[1], best[0]


def compute_dbqite_theta_sequence(delta, omega, s, theta_init, n_steps, method="GC"):
    """
    Simulate n_steps of DB-QITE and return the sequence of rotation angles
    (cumulative from |0>) for each step.
    """
    psi0 = Ry(theta_init) @ psi_zero
    s_list = [s] * n_steps

    Es, Vs, Fs, states = simulate_dbqite_single_qubit(
        delta, omega, s_list, psi0, method=method, return_states=True
    )

    # Extract theta for each state
    thetas = [theta_from_state_on_ry_meridian(st) for st in states]
    return thetas, Fs, Es, Vs


def theta_overlap_with_gs(delta, omega):
    """
    Return theta_gs such that Ry(-theta_gs) rotates ground state onto |0>.
    Measuring P(|0>) after this rotation gives fidelity with ground state.
    """
    theta_gs = theta_ground_state(delta, omega)
    return -theta_gs  # rotate GS -> |0>, so measure overlap


# =========================
# Run single fidelity measurement (debug style)
# =========================
def run_single_fidelity_measurement(qubit, a_total, a_overlap, n_avg, reset_type):
    """
    Measures fidelity for a single step count using QuAM macros.
    a_total is the cumulative rotation from |0> to reach the state after k steps.
    a_overlap is the rotation to measure overlap with ground state.
    
    Returns P(|1>) which gives F = 1 - P(|1>) = P(|0>) in overlap basis.
    """
    qmm = machine.connect()

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # Reset
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()

            # Prepare state after k DB-QITE steps: Ry(theta_total)|0⟩
            qubit.xy.play("y180", amplitude_scale=a_total)
            qubit.align()

            # Rotate to overlap measurement basis: Ry(-theta_gs)
            qubit.xy.play("y180", amplitude_scale=a_overlap)
            qubit.align()

            # Measure
            readout_state(qubit, state)
            save(state, st)

        with stream_processing():
            n_st.save("n")
            st.average().save("p1")

    # Execute
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog)
        results_fetcher = fetching_tool(job, ["n"], mode="live")
        while results_fetcher.is_processing():
            n_done = results_fetcher.fetch_all()[0]
            progress_counter(n_done, n_avg, start_time=results_fetcher.start_time)

        job.result_handles.wait_for_all_values()
        p1 = job.result_handles.get("p1").fetch_all()

    return float(p1)


# =========================
# Run experiment
# =========================
def run_fidelity_vs_step(qubit):
    delta = node.parameters.DELTA
    omega = node.parameters.OMEGA
    s_conv = node.parameters.S_CONV
    n_steps = node.parameters.N_STEPS
    method = node.parameters.DBQITE_METHOD
    n_avg = node.parameters.num_averages
    reset_type = node.parameters.reset_type_thermal_or_active

    # Overlap rotation angle
    a_overlap = float(theta_overlap_with_gs(delta, omega) / np.pi)

    print(f"\n{'='*60}")
    print(f"DB-QITE Fidelity vs Step Experiment")
    print(f"{'='*60}")
    print(f"Qubit: {qubit.name}")
    print(f"Hamiltonian: delta={delta}, omega={omega}")
    print(f"Step size s = {s_conv}")
    print(f"Number of steps = {n_steps}")
    print(f"a_overlap = {a_overlap:.4f}")
    print(f"Reset type: {reset_type}")
    print(f"{'='*60}\n")

    # Cold start: theta_init = 0 (|0>)
    theta_cold = 0.0
    thetas_cold, Fs_cold_theory, Es_cold, Vs_cold = compute_dbqite_theta_sequence(
        delta, omega, s_conv, theta_cold, n_steps, method=method
    )

    # Warm start: find theta giving F0 ~ target
    theta_warm, F0_warm = find_warm_start_theta(
        delta, omega, target_F0_max=node.parameters.TARGET_F0_WARM
    )
    thetas_warm, Fs_warm_theory, Es_warm, Vs_warm = compute_dbqite_theta_sequence(
        delta, omega, s_conv, theta_warm, n_steps, method=method
    )

    print(f"Cold start theta_init = {theta_cold:.4f} rad = {np.degrees(theta_cold):.2f}°")
    print(f"Warm start theta_init = {theta_warm:.4f} rad = {np.degrees(theta_warm):.2f}° (F0={F0_warm:.4f})")
    print(f"\nCold start thetas (theory): {[f'{t:.4f}' for t in thetas_cold[:6]]}")
    print(f"Warm start thetas (theory): {[f'{t:.4f}' for t in thetas_warm[:6]]}")
    print(f"Theory fidelities cold: {[f'{f:.4f}' for f in Fs_cold_theory[:6]]}")
    print(f"Theory fidelities warm: {[f'{f:.4f}' for f in Fs_warm_theory[:6]]}")

    # Measure fidelity for cold start
    Fs_cold_hw = np.zeros(n_steps + 1)
    Fs_cold_hw_err = np.zeros(n_steps + 1)
    print("\n--- Cold start measurements ---")
    for k in range(n_steps + 1):
        a_total = float(thetas_cold[k] / np.pi)
        print(f"\nStep {k}: theta={thetas_cold[k]:.4f} rad, a_total={a_total:.4f}")

        p1 = run_single_fidelity_measurement(
            qubit=qubit,
            a_total=a_total,
            a_overlap=a_overlap,
            n_avg=n_avg,
            reset_type=reset_type,
        )
        p1_c = np.clip(p1, 1e-12, 1 - 1e-12)
        sigma_F = np.sqrt(p1_c * (1 - p1_c) / n_avg)

        Fs_cold_hw[k] = 1.0 - p1  # F = P(|0>) in overlap basis
        Fs_cold_hw_err[k] = sigma_F
        print(f"  P1={p1:.4f}, F_hw={Fs_cold_hw[k]:.4f}, F_theory={Fs_cold_theory[k]:.4f}")

    # Measure fidelity for warm start
    Fs_warm_hw = np.zeros(n_steps + 1)
    Fs_warm_hw_err = np.zeros(n_steps + 1)
    print("\n--- Warm start measurements ---")
    for k in range(n_steps + 1):
        a_total = float(thetas_warm[k] / np.pi)
        print(f"\nStep {k}: theta={thetas_warm[k]:.4f} rad, a_total={a_total:.4f}")

        p1 = run_single_fidelity_measurement(
            qubit=qubit,
            a_total=a_total,
            a_overlap=a_overlap,
            n_avg=n_avg,
            reset_type=reset_type,
        )
        p1_c = np.clip(p1, 1e-12, 1 - 1e-12)
        sigma_F = np.sqrt(p1_c * (1 - p1_c) / n_avg)

        Fs_warm_hw[k] = 1.0 - p1
        Fs_warm_hw_err[k] = sigma_F
        print(f"  P1={p1:.4f}, F_hw={Fs_warm_hw[k]:.4f}, F_theory={Fs_warm_theory[k]:.4f}")

    return {
        "ks": np.arange(n_steps + 1),
        "Fs_cold_hw": Fs_cold_hw,
        "Fs_warm_hw": Fs_warm_hw,
        "Fs_cold_theory": np.array(Fs_cold_theory),
        "Fs_warm_theory": np.array(Fs_warm_theory),
        "thetas_cold": thetas_cold,
        "thetas_warm": thetas_warm,
        "theta_warm_init": theta_warm,
        "F0_warm": F0_warm,
        "Fs_cold_hw_err": Fs_cold_hw_err,
        "Fs_warm_hw_err": Fs_warm_hw_err
    }


# =========================
# Main execution
# =========================
all_results = {}
for q in qubits[:1]:
    all_results[q.name] = run_fidelity_vs_step(q)

# Plot results
for qname, r in all_results.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Hardware results
    ax1 = axes[0]
    ax1.plot(r["ks"], r["Fs_cold_hw"], marker="o", label="Cold start (hardware)")
    ax1.plot(r["ks"], r["Fs_warm_hw"], marker="x", linestyle="--", label="Warm start (hardware)")
    ax1.set_xlabel("DB-QITE step $k$")
    ax1.set_ylabel("Ground-state fidelity $F_k$")
    ax1.set_title(f"DB-QITE fidelity vs step (hardware) [{qname}]")
    ax1.legend()
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)

    # Right: Theory comparison
    ax2 = axes[1]
    ax2.plot(r["ks"], r["Fs_cold_theory"], marker="o", label="Cold start (theory)")
    ax2.plot(r["ks"], r["Fs_warm_theory"], marker="x", linestyle="--", label="Warm start (theory)")
    ax2.set_xlabel("DB-QITE step $k$")
    ax2.set_ylabel("Ground-state fidelity $F_k$")
    ax2.set_title("DB-QITE fidelity vs step (theory)")
    ax2.legend()
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"dbqite_fidelity_vs_step_{qname}.png", dpi=150)
    plt.show()

    # Combined plot (hardware vs theory)
    plt.figure(figsize=(10, 6))

    # Hardware with error bars
    plt.errorbar(
        r["ks"], r["Fs_cold_hw"], yerr=r["Fs_cold_hw_err"],
        fmt="o-", color="blue", capsize=4, label="Cold (hardware)"
    )
    plt.plot(
        r["ks"], r["Fs_cold_theory"],
        "s--", color="blue", alpha=0.5, label="Cold (theory)"
    )

    plt.errorbar(
        r["ks"], r["Fs_warm_hw"], yerr=r["Fs_warm_hw_err"],
        fmt="o-", color="red", capsize=4, label="Warm (hardware)"
    )
    plt.plot(
        r["ks"], r["Fs_warm_theory"],
        "s--", color="red", alpha=0.5, label="Warm (theory)"
    )

    plt.xlabel("DB-QITE step $k$")
    plt.ylabel("Ground-state fidelity $F_k$")
    plt.title(f"DB-QITE fidelity vs step [{qname}]")
    plt.legend()
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dbqite_fidelity_combined_{qname}.png", dpi=150)
    plt.show()


    # Summary table
    print(f"\n{'='*70}")
    print(f"Summary for {qname}")
    print(f"{'='*70}")
    print(f"{'Step':<6} {'F_cold_hw':<12} {'F_cold_th':<12} {'F_warm_hw':<12} {'F_warm_th':<12}")
    print("-" * 70)
    for k in r["ks"]:
        print(
            f"{k:<6} {r['Fs_cold_hw'][k]:<12.4f} {r['Fs_cold_theory'][k]:<12.4f} "
            f"{r['Fs_warm_hw'][k]:<12.4f} {r['Fs_warm_theory'][k]:<12.4f}"
        )
    print(f"{'='*70}\n")

node.results = {"fidelity_vs_step": all_results}
node.save()