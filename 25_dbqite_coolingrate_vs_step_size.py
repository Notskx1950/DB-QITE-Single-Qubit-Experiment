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

    # Step-size scan range
    S_MIN: float = 0.1
    S_MAX: float = 0.5
    N_S_POINTS: int = 5

    # Initial state psi0 = Ry(THETA_INIT)|0>
    THETA_INIT: float = np.pi / 4.0

    # shots
    num_averages: int = 9000

    # Reset behavior
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    thermal_reset_time_ns: int = 200000
    depletion_time_ns: int = 4000

    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 200


node = QualibrationNode(name="DBQITE_CoolingRate_vs_StepSize", parameters=Parameters())

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


def get_hamiltonian_norm(delta, omega):
    """
    Get the energy scale Hnorm such that eigenvalues are ±Hnorm.
    """
    H = make_H(delta, omega)
    evals = np.linalg.eigvalsh(H)
    return (evals.max() - evals.min()) / 2.0


def theta_hamiltonian_axis(delta, omega):
    """
    For H in XZ plane, energy measurement can be done by rotating the Hamiltonian axis onto Z.
    theta_h = atan2(omega, delta).
    Pre-rotate by Ry(-theta_h), then Z readout gives <sigma_h>.
    """
    return wrap_to_pi(np.arctan2(omega, delta))


def compute_theta_step_from_sim(delta, omega, s, theta_prep, method="GC"):
    """
    Compute the incremental rotation angle for one DB-QITE step.
    """
    psi0 = Ry(theta_prep) @ psi_zero
    Es, Vs, Fs, states = simulate_dbqite_single_qubit(
        delta, omega, [s], psi0, method=method, return_states=True
    )
    th0 = theta_from_state_on_ry_meridian(states[0])
    th1 = theta_from_state_on_ry_meridian(states[1])
    return wrap_to_pi(th1 - th0)


def compute_theory_cooling_rate(delta, omega, s_values, theta_init, method="GC"):
    """
    Compute theory predictions for E0, E1, V0, and cooling rate for each step size.
    """
    n_points = len(s_values)
    E0_theory = np.zeros(n_points)
    E1_theory = np.zeros(n_points)
    V0_theory = np.zeros(n_points)

    psi0 = Ry(theta_init) @ psi_zero
    H = make_H(delta, omega)

    for i, s in enumerate(s_values):
        Es, Vs, Fs, states = simulate_dbqite_single_qubit(
            delta, omega, [s], psi0, method=method, return_states=True
        )
        E0_theory[i] = Es[0]
        E1_theory[i] = Es[1]
        V0_theory[i] = Vs[0]

    cooling_rate_theory = (E1_theory - E0_theory) / s_values
    minus_2V0_theory = -2.0 * V0_theory

    return {
        "E0": E0_theory,
        "E1": E1_theory,
        "V0": V0_theory,
        "cooling_rate": cooling_rate_theory,
        "minus_2V0": minus_2V0_theory,
    }


# =========================
# Run experiment for one step size
# =========================
def run_one_step_size(qubit, a_prep, a_step, a_energy, n_avg, reset_type):
    """
    Measures energy before (E0) and after (E1) one DB-QITE step.
    Uses QuAM macros (same style as working debug code).
    
    Returns P1 for two cases:
      case0: prep -> energy_basis -> measure  (gives E0)
      case1: prep -> step -> energy_basis -> measure  (gives E1)
    """
    qmm = machine.connect()

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # -------- case 0: energy before step (E0) --------
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()

            # Prepare initial state: Ry(theta_prep)|0⟩
            qubit.xy.play("y180", amplitude_scale=a_prep)
            qubit.align()

            # Rotate to energy measurement basis: Ry(-theta_h)
            qubit.xy.play("y180", amplitude_scale=a_energy)
            qubit.align()

            readout_state(qubit, state)
            save(state, st)

            # -------- case 1: energy after one step (E1) --------
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()

            # Prepare initial state: Ry(theta_prep)|0⟩
            qubit.xy.play("y180", amplitude_scale=a_prep)
            qubit.align()

            # Apply one DB-QITE step: Ry(theta_step)
            qubit.xy.play("y180", amplitude_scale=a_step)
            qubit.align()

            # Rotate to energy measurement basis: Ry(-theta_h)
            qubit.xy.play("y180", amplitude_scale=a_energy)
            qubit.align()

            readout_state(qubit, state)
            save(state, st)

        with stream_processing():
            n_st.save("n")
            st.buffer(2).average().save("p1_cases")  # [P1_E0, P1_E1]

    # Execute
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog)
        results_fetcher = fetching_tool(job, ["n"], mode="live")
        while results_fetcher.is_processing():
            n_done = results_fetcher.fetch_all()[0]
            progress_counter(n_done, n_avg, start_time=results_fetcher.start_time)

        job.result_handles.wait_for_all_values()
        p1_cases = job.result_handles.get("p1_cases").fetch_all()

    p1_E0, p1_E1 = float(p1_cases[0]), float(p1_cases[1])

    return p1_E0, p1_E1


# =========================
# Run experiment
# =========================
def run_cooling_rate_vs_step_size(qubit):
    delta = node.parameters.DELTA
    omega = node.parameters.OMEGA
    method = node.parameters.DBQITE_METHOD
    theta_init = float(node.parameters.THETA_INIT)
    n_avg = node.parameters.num_averages
    reset_type = node.parameters.reset_type_thermal_or_active

    # Step size scan
    s_values = np.linspace(
        node.parameters.S_MIN, node.parameters.S_MAX, node.parameters.N_S_POINTS
    )

    # Hamiltonian parameters
    Hnorm = get_hamiltonian_norm(delta, omega)
    theta_h = theta_hamiltonian_axis(delta, omega)
    a_energy = float((-theta_h) / np.pi)  # Ry(-theta_h) rotates H axis onto Z
    a_prep = float(theta_init / np.pi)

    print(f"\n{'='*60}")
    print(f"DB-QITE Cooling Rate vs Step Size Experiment")
    print(f"{'='*60}")
    print(f"Qubit: {qubit.name}")
    print(f"Hamiltonian: delta={delta}, omega={omega}")
    print(f"Hnorm = {Hnorm:.4f}")
    print(f"theta_h = {theta_h:.4f} rad = {np.degrees(theta_h):.2f}°")
    print(f"Initial state theta = {theta_init:.4f} rad = {np.degrees(theta_init):.2f}°")
    print(f"a_prep = {a_prep:.4f}, a_energy = {a_energy:.4f}")
    print(f"Step sizes: {s_values}")
    print(f"Reset type: {reset_type}")
    print(f"{'='*60}\n")

    # Theory predictions
    theory = compute_theory_cooling_rate(delta, omega, s_values, theta_init, method=method)

    print("Theory predictions:")
    for i, s in enumerate(s_values):
        print(
            f"  s={s:.4f}: E0={theory['E0'][i]:.4f}, E1={theory['E1'][i]:.4f}, "
            f"V0={theory['V0'][i]:.4f}, rate={theory['cooling_rate'][i]:.4f}"
        )

    # Hardware measurements
    E0_hw = np.zeros_like(s_values, dtype=float)
    E1_hw = np.zeros_like(s_values, dtype=float)
    V0_hw = np.zeros_like(s_values, dtype=float)
    cooling_rate_err = np.zeros_like(s_values, dtype=float)
    minus2V0_err = np.zeros_like(s_values, dtype=float)

    print("\n--- Hardware measurements ---")
    for i, s in enumerate(s_values):
        # Compute step rotation from theory
        theta_step = compute_theta_step_from_sim(delta, omega, float(s), theta_init, method=method)
        a_step = float(theta_step / np.pi)

        print(f"\nStep size s={s:.4f}:")
        print(f"  theta_step = {theta_step:.4f} rad = {np.degrees(theta_step):.2f}°")
        print(f"  a_step = {a_step:.4f}")

        # Run measurement
        p1_E0, p1_E1 = run_one_step_size(
            qubit=qubit,
            a_prep=a_prep,
            a_step=a_step,
            a_energy=a_energy,
            n_avg=n_avg,
            reset_type=reset_type,
        )

        # Convert P1 to <Z> in energy basis, then to energy
        z_before = 1.0 - 2.0 * p1_E0
        z_after = 1.0 - 2.0 * p1_E1

        E0_hw[i] = Hnorm * z_before
        E1_hw[i] = Hnorm * z_after

        # Variance: for 1-qubit H = Hnorm * sigma_h, <H^2> = Hnorm^2
        # So Var(H) = <H^2> - <H>^2 = Hnorm^2 - E0^2
        V0_hw[i] = (Hnorm ** 2) - (E0_hw[i] ** 2)

        N = node.parameters.num_averages  # e.g. 6000
        p1_0 = p1_E0
        p1_1 = p1_E1
        gap = np.sqrt(delta**2 + omega**2)     # Δgap = sqrt(Δ^2+Ω^2)

        sig_p0 = np.sqrt(p1_0*(1-p1_0)/N)
        sig_p1 = np.sqrt(p1_1*(1-p1_1)/N)
        sig_Z0 = 2*sig_p0
        sig_Z1 = 2*sig_p1
        sig_E0 = 0.5*gap*sig_Z0
        sig_E1 = 0.5*gap*sig_Z1

        cooling_rate_err[i] = np.sqrt(sig_E0**2 + sig_E1**2) / s
        minus2V0_err[i]     = abs(4*E0_hw[i]) * sig_E0

        print(
            f"  P1_E0={p1_E0:.4f}, P1_E1={p1_E1:.4f}"
        )
        print(
            f"  E0_hw={E0_hw[i]:.4f} (th={theory['E0'][i]:.4f}), "
            f"E1_hw={E1_hw[i]:.4f} (th={theory['E1'][i]:.4f}), "
            f"V0_hw={V0_hw[i]:.4f} (th={theory['V0'][i]:.4f})"
        )


    # Compute cooling rates
    cooling_rate_hw = (E1_hw - E0_hw) / s_values
    minus_2V0_hw = -2.0 * V0_hw

    return {
        "s_values": s_values,
        "E0_hw": E0_hw,
        "E1_hw": E1_hw,
        "V0_hw": V0_hw,
        "cooling_rate_hw": cooling_rate_hw,
        "minus_2V0_hw": minus_2V0_hw,
        "E0_theory": theory["E0"],
        "E1_theory": theory["E1"],
        "V0_theory": theory["V0"],
        "cooling_rate_theory": theory["cooling_rate"],
        "minus_2V0_theory": theory["minus_2V0"],
        "Hnorm": Hnorm,
        "theta_h": theta_h,
        "theta_init": theta_init,
        "cooling_rate_err": cooling_rate_err,
        "minus_2V0_err": minus2V0_err,
    }


# =========================
# Main execution
# =========================
all_results = {}
for q in qubits[:1]:
    all_results[q.name] = run_cooling_rate_vs_step_size(q)

# Plot results
for qname, r in all_results.items():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Hardware results
    ax1 = axes[0]
    ax1.plot(
        r["s_values"],
        r["cooling_rate_hw"],
        "o-",
        color="blue",
        markersize=8,
        linewidth=2,
        label=r"$(E_1 - E_0)/s$ (hardware)",
    )
    ax1.plot(
        r["s_values"],
        r["minus_2V0_hw"],
        "x--",
        color="red",
        markersize=8,
        linewidth=2,
        label=r"$-2V_0$ (hardware)",
    )
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Step size $s$")
    ax1.set_ylabel("Cooling rate")
    ax1.set_title(f"DB-QITE cooling rate vs step size (hardware) [{qname}]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Theory results
    ax2 = axes[1]
    ax2.plot(
        r["s_values"],
        r["cooling_rate_theory"],
        "o-",
        color="blue",
        markersize=8,
        linewidth=2,
        label=r"$(E_1 - E_0)/s$ (theory)",
    )
    ax2.plot(
        r["s_values"],
        r["minus_2V0_theory"],
        "x--",
        color="red",
        markersize=8,
        linewidth=2,
        label=r"$-2V_0$ (theory)",
    )
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Step size $s$")
    ax2.set_ylabel("Cooling rate")
    ax2.set_title("DB-QITE cooling rate vs step size (theory)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"dbqite_cooling_rate_{qname}.png", dpi=150)
    plt.show()

    # Combined plot (hardware vs theory)
    plt.figure(figsize=(10, 6))

    s = np.array(r["s_values"], dtype=float)

    # --- hardware cooling rate with error bars ---
    plt.errorbar(
        s,
        r["cooling_rate_hw"],
        yerr=r["cooling_rate_err"],   # <-- add this
        fmt="o-",
        color="blue",
        markersize=10,
        linewidth=2,
        capsize=4,
        label=r"$(E_1 - E_0)/s$ (hardware)",
    )

    # theory cooling rate (no error bars)
    plt.plot(
        s,
        r["cooling_rate_theory"],
        "s--",
        color="blue",
        markersize=8,
        linewidth=2,
        alpha=0.5,
        label=r"$(E_1 - E_0)/s$ (theory)",
    )

    # --- hardware -2V0 with error bars ---
    plt.errorbar(
        s,
        r["minus_2V0_hw"],
        yerr=r["minus_2V0_err"],      # <-- add this
        fmt="o-",
        color="red",
        markersize=10,
        linewidth=2,
        capsize=4,
        label=r"$-2V_0$ (hardware)",
    )

    # theory -2V0 (no error bars)
    plt.plot(
        s,
        r["minus_2V0_theory"],
        "s--",
        color="red",
        markersize=8,
        linewidth=2,
        alpha=0.5,
        label=r"$-2V_0$ (theory)",
    )

    plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("Step size $s$")
    plt.ylabel("Cooling rate")
    plt.title(f"DB-QITE cooling rate vs step size [{qname}]")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dbqite_cooling_rate_combined_{qname}.png", dpi=150)
    plt.show()


    # Additional plot: E0 and E1 vs step size
    plt.figure(figsize=(10, 6))
    plt.plot(
        r["s_values"],
        r["E0_hw"],
        "o-",
        color="blue",
        markersize=8,
        linewidth=2,
        label=r"$E_0$ (hardware)",
    )
    plt.plot(
        r["s_values"],
        r["E0_theory"],
        "s--",
        color="blue",
        markersize=6,
        linewidth=2,
        alpha=0.5,
        label=r"$E_0$ (theory)",
    )
    plt.plot(
        r["s_values"],
        r["E1_hw"],
        "o-",
        color="green",
        markersize=8,
        linewidth=2,
        label=r"$E_1$ (hardware)",
    )
    plt.plot(
        r["s_values"],
        r["E1_theory"],
        "s--",
        color="green",
        markersize=6,
        linewidth=2,
        alpha=0.5,
        label=r"$E_1$ (theory)",
    )
    plt.xlabel("Step size $s$")
    plt.ylabel("Energy")
    plt.title(f"DB-QITE energy before/after step vs step size [{qname}]")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dbqite_energy_vs_stepsize_{qname}.png", dpi=150)
    plt.show()

    # Summary table
    print(f"\n{'='*80}")
    print(f"Summary for {qname}")
    print(f"{'='*80}")
    print(
        f"{'s':<10} {'E0_hw':<10} {'E0_th':<10} {'E1_hw':<10} {'E1_th':<10} "
        f"{'rate_hw':<12} {'rate_th':<12} {'-2V0_hw':<12} {'-2V0_th':<12}"
    )
    print("-" * 80)
    for i, s in enumerate(r["s_values"]):
        print(
            f"{s:<10.4f} {r['E0_hw'][i]:<10.4f} {r['E0_theory'][i]:<10.4f} "
            f"{r['E1_hw'][i]:<10.4f} {r['E1_theory'][i]:<10.4f} "
            f"{r['cooling_rate_hw'][i]:<12.4f} {r['cooling_rate_theory'][i]:<12.4f} "
            f"{r['minus_2V0_hw'][i]:<12.4f} {r['minus_2V0_theory'][i]:<12.4f}"
        )
    print(f"{'='*80}\n")

node.results = {"cooling_rate_vs_step_size": all_results}
node.save()