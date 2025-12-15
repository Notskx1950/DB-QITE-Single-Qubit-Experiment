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


node = QualibrationNode(name="DBQITE_Energy_vs_Step", parameters=Parameters())

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


def theta_hamiltonian_axis(delta, omega):
    """
    For H in XZ plane, energy measurement can be done by rotating the Hamiltonian axis onto Z.
    theta_h = atan2(omega, delta).
    Pre-rotate by Ry(-theta_h), then Z readout gives <sigma_h>.
    """
    return wrap_to_pi(np.arctan2(omega, delta))


def get_hamiltonian_norm(delta, omega):
    """
    Get the energy scale Hnorm such that eigenvalues are ±Hnorm.
    For H = delta*Z + omega*X, eigenvalues are ±sqrt(delta^2 + omega^2).
    """
    H = make_H(delta, omega)
    evals = np.linalg.eigvalsh(H)
    return (evals.max() - evals.min()) / 2.0


def get_ground_energy(delta, omega):
    """Get the ground state energy."""
    H = make_H(delta, omega)
    evals = np.linalg.eigvalsh(H)
    return evals.min()


# =========================
# Run single energy measurement (debug style)
# =========================
def run_single_energy_measurement(qubit, a_state, a_energy, n_avg, reset_type):
    """
    Measures energy expectation value for a given state using QuAM macros.
    
    Protocol:
    1. Prepare state: Ry(theta_k)|0>
    2. Rotate to energy basis: Ry(-theta_h)
    3. Measure Z -> gives <sigma_h>
    4. Energy = Hnorm * <sigma_h>
    
    Returns P(|1>) which gives <Z> = 1 - 2*P1.
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

            # Prepare state after k DB-QITE steps: Ry(theta_k)|0⟩
            qubit.xy.play("y180", amplitude_scale=a_state)
            qubit.align()

            # Rotate to energy measurement basis: Ry(-theta_h)
            qubit.xy.play("y180", amplitude_scale=a_energy)
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
def run_energy_vs_step(qubit):
    delta = node.parameters.DELTA
    omega = node.parameters.OMEGA
    s_conv = node.parameters.S_CONV
    n_steps = node.parameters.N_STEPS
    method = node.parameters.DBQITE_METHOD
    n_avg = node.parameters.num_averages
    reset_type = node.parameters.reset_type_thermal_or_active

    # Energy measurement rotation angle
    theta_h = theta_hamiltonian_axis(delta, omega)
    a_energy = float((-theta_h) / np.pi)  # Ry(-theta_h) rotates H axis onto Z

    # Energy scale and ground energy
    Hnorm = get_hamiltonian_norm(delta, omega)
    Egs = get_ground_energy(delta, omega)

    print(f"\n{'='*60}")
    print(f"DB-QITE Energy vs Step Experiment")
    print(f"{'='*60}")
    print(f"Qubit: {qubit.name}")
    print(f"Hamiltonian: delta={delta}, omega={omega}")
    print(f"Energy scale Hnorm = {Hnorm:.4f}")
    print(f"Ground state energy Egs = {Egs:.4f}")
    print(f"theta_h = {theta_h:.4f} rad = {np.degrees(theta_h):.2f}°")
    print(f"a_energy = {a_energy:.4f}")
    print(f"Step size s = {s_conv}")
    print(f"Number of steps = {n_steps}")
    print(f"Reset type: {reset_type}")
    print(f"{'='*60}\n")

    # Cold start: theta_init = 0 (|0>)
    theta_cold = 0.0
    thetas_cold, Fs_cold_theory, Es_cold_theory, Vs_cold_theory = compute_dbqite_theta_sequence(
        delta, omega, s_conv, theta_cold, n_steps, method=method
    )

    # Warm start: find theta giving F0 ~ target
    theta_warm, F0_warm = find_warm_start_theta(
        delta, omega, target_F0_max=node.parameters.TARGET_F0_WARM
    )
    thetas_warm, Fs_warm_theory, Es_warm_theory, Vs_warm_theory = compute_dbqite_theta_sequence(
        delta, omega, s_conv, theta_warm, n_steps, method=method
    )

    print(f"Cold start theta_init = {theta_cold:.4f} rad = {np.degrees(theta_cold):.2f}°")
    print(f"Warm start theta_init = {theta_warm:.4f} rad = {np.degrees(theta_warm):.2f}° (F0={F0_warm:.4f})")
    print(f"\nCold start thetas (theory): {[f'{t:.4f}' for t in thetas_cold[:6]]}")
    print(f"Warm start thetas (theory): {[f'{t:.4f}' for t in thetas_warm[:6]]}")
    print(f"Theory energies cold: {[f'{e:.4f}' for e in Es_cold_theory[:6]]}")
    print(f"Theory energies warm: {[f'{e:.4f}' for e in Es_warm_theory[:6]]}")

    # Measure energies on hardware cold start
    Es_cold_hw = np.zeros(n_steps + 1)
    Es_cold_hw_err = np.zeros(n_steps + 1)
    print("\n--- Cold start energy measurements ---")
    for k in range(n_steps + 1):
        a_state = float(thetas_cold[k] / np.pi)

        p1 = run_single_energy_measurement(
            qubit=qubit,
            a_state=a_state,
            a_energy=a_energy,
            n_avg=n_avg,
            reset_type=reset_type,
        )

        z_expect = 1.0 - 2.0 * p1
        E = Hnorm * z_expect

        # shot-noise (±1σ)
        p1_c = np.clip(p1, 1e-12, 1 - 1e-12)
        sigma_p1 = np.sqrt(p1_c * (1 - p1_c) / n_avg)
        sigma_E = 2.0 * Hnorm * sigma_p1

        Es_cold_hw[k] = E
        Es_cold_hw_err[k] = sigma_E

    # Measure energies on hardware warm start
    Es_warm_hw = np.zeros(n_steps + 1)
    Es_warm_hw_err = np.zeros(n_steps + 1)
    print("\n--- Warm start energy measurements ---")
    for k in range(n_steps + 1):
        a_state = float(thetas_warm[k] / np.pi)

        p1 = run_single_energy_measurement(
            qubit=qubit,
            a_state=a_state,
            a_energy=a_energy,
            n_avg=n_avg,
            reset_type=reset_type,
        )

        z_expect = 1.0 - 2.0 * p1
        E = Hnorm * z_expect

        p1_c = np.clip(p1, 1e-12, 1 - 1e-12)
        sigma_p1 = np.sqrt(p1_c * (1 - p1_c) / n_avg)
        sigma_E = 2.0 * Hnorm * sigma_p1

        Es_warm_hw[k] = E
        Es_warm_hw_err[k] = sigma_E


    return {
        "ks": np.arange(n_steps + 1),
        "Es_cold_hw": Es_cold_hw,
        "Es_warm_hw": Es_warm_hw,
        "Es_cold_theory": np.array(Es_cold_theory),
        "Es_warm_theory": np.array(Es_warm_theory),
        "Vs_cold_theory": np.array(Vs_cold_theory),
        "Vs_warm_theory": np.array(Vs_warm_theory),
        "thetas_cold": thetas_cold,
        "thetas_warm": thetas_warm,
        "theta_warm_init": theta_warm,
        "F0_warm": F0_warm,
        "Egs": Egs,
        "Hnorm": Hnorm,
        "Es_cold_hw_err": Es_cold_hw_err,
        "Es_warm_hw_err": Es_warm_hw_err,
    }


# =========================
# Main execution
# =========================
all_results = {}
for q in qubits[:1]:
    all_results[q.name] = run_energy_vs_step(q)

# Plot results
for qname, r in all_results.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Hardware results
    ax1 = axes[0]
    ax1.axhline(r["Egs"], color="gray", linestyle=":", label="Ground energy")
    ax1.plot(r["ks"], r["Es_cold_hw"], marker="o", label="Cold start (hardware)")
    ax1.plot(r["ks"], r["Es_warm_hw"], marker="x", linestyle="--", label="Warm start (hardware)")
    ax1.set_xlabel("DB-QITE step $k$")
    ax1.set_ylabel("Energy $E_k$")
    ax1.set_title(f"DB-QITE energy vs step (hardware) [{qname}]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Theory comparison
    ax2 = axes[1]
    ax2.axhline(r["Egs"], color="gray", linestyle=":", label="Ground energy")
    ax2.plot(r["ks"], r["Es_cold_theory"], marker="o", label="Cold start (theory)")
    ax2.plot(r["ks"], r["Es_warm_theory"], marker="x", linestyle="--", label="Warm start (theory)")
    ax2.set_xlabel("DB-QITE step $k$")
    ax2.set_ylabel("Energy $E_k$")
    ax2.set_title("DB-QITE energy vs step (theory)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"dbqite_energy_vs_step_{qname}.png", dpi=150)
    plt.show()

    # Combined plot (hardware vs theory)
    plt.figure(figsize=(10, 6))
    plt.axhline(r["Egs"], color="gray", linestyle=":", linewidth=2, label="Ground energy")

    plt.errorbar(
        r["ks"], r["Es_cold_hw"], yerr=r["Es_cold_hw_err"],
        fmt="o-", color="blue", capsize=4, label="Cold (hardware)"
    )
    plt.plot(
        r["ks"], r["Es_cold_theory"],
        "s--", color="blue", alpha=0.5, label="Cold (theory)"
    )

    plt.errorbar(
        r["ks"], r["Es_warm_hw"], yerr=r["Es_warm_hw_err"],
        fmt="o-", color="red", capsize=4, label="Warm (hardware)"
    )
    plt.plot(
        r["ks"], r["Es_warm_theory"],
        "s--", color="red", alpha=0.5, label="Warm (theory)"
    )

    plt.xlabel("DB-QITE step $k$")
    plt.ylabel("Energy $E_k$")
    plt.title(f"DB-QITE energy vs step [{qname}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dbqite_energy_combined_{qname}.png", dpi=150)
    plt.show()


    # Summary table
    print(f"\n{'='*70}")
    print(f"Summary for {qname}")
    print(f"{'='*70}")
    print(f"Ground state energy: {r['Egs']:.4f}")
    print(f"{'Step':<6} {'E_cold_hw':<12} {'E_cold_th':<12} {'E_warm_hw':<12} {'E_warm_th':<12}")
    print("-" * 70)
    for k in r["ks"]:
        print(
            f"{k:<6} {r['Es_cold_hw'][k]:<12.4f} {r['Es_cold_theory'][k]:<12.4f} "
            f"{r['Es_warm_hw'][k]:<12.4f} {r['Es_warm_theory'][k]:<12.4f}"
        )
    print(f"{'='*70}\n")

node.results = {"energy_vs_step": all_results}
node.save()