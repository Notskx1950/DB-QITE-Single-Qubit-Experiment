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

    # SPAM correction
    apply_spam_correction: bool = True

    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 200


node = QualibrationNode(name="DBQITE_CoolingRate_Debug", parameters=Parameters())

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
    H = make_H(delta, omega)
    evals = np.linalg.eigvalsh(H)
    return (evals.max() - evals.min()) / 2.0


def theta_hamiltonian_axis(delta, omega):
    return wrap_to_pi(np.arctan2(omega, delta))


def compute_theta_step_from_sim(delta, omega, s, theta_prep, method="GC"):
    psi0 = Ry(theta_prep) @ psi_zero
    Es, Vs, Fs, states = simulate_dbqite_single_qubit(
        delta, omega, [s], psi0, method=method, return_states=True
    )
    th0 = theta_from_state_on_ry_meridian(states[0])
    th1 = theta_from_state_on_ry_meridian(states[1])
    return wrap_to_pi(th1 - th0)


def compute_theory_cooling_rate(delta, omega, s_values, theta_init, method="GC"):
    n_points = len(s_values)
    E0_theory = np.zeros(n_points)
    E1_theory = np.zeros(n_points)
    V0_theory = np.zeros(n_points)

    psi0 = Ry(theta_init) @ psi_zero

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


def correct_p1_spam(p1_raw, p1_when_0, p1_when_1):
    """
    Correct P1 measurement for SPAM errors.
    p1_raw = p1_true * p1_when_1 + (1 - p1_true) * p1_when_0
    Solving: p1_true = (p1_raw - p1_when_0) / (p1_when_1 - p1_when_0)
    """
    denom = p1_when_1 - p1_when_0
    if abs(denom) < 0.01:
        print("WARNING: Readout contrast too low!")
        return p1_raw
    p1_corrected = (p1_raw - p1_when_0) / denom
    return np.clip(p1_corrected, 0, 1)


def run_xy_gate_sanity(qubit, n_avg=5000, reset_type="active"):
    """
    Sanity check for x90/x180 using Z readout:
      0) Id
      1) x180
      2) x90
      3) x90 + x90

    Returns raw and SPAM-corrected <Z> estimates.
    """
    qmm = machine.connect()

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        st = declare_stream()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # ---- Case 0: Id ----
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()
            readout_state(qubit, state)
            save(state, st)

            # ---- Case 1: x180 ----
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()
            qubit.xy.play("x180")
            qubit.align()
            readout_state(qubit, state)
            save(state, st)

            # ---- Case 2: x90 ----
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)

            reset_frame(qubit.xy.name)
            qubit.align()

            qubit.xy.play("y180")      # <-- use native y180 from config
            qubit.align()

            readout_state(qubit, state)
            save(state, st)

            # ---- Case 3: x90 + x90 ----
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()
            qubit.xy.play("y90")
            qubit.align()
            readout_state(qubit, state)
            save(state, st)
            # ---- Case 4: x90 + (-x90) ----
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()
            qubit.xy.frame_rotation(-np.pi/2)
            qubit.align()
            qubit.xy.play("x90")
            qubit.align()
            qubit.xy.frame_rotation(+np.pi/2)
            qubit.align()
            readout_state(qubit, state)
            save(state, st)

            # ---- Case 5: (-x90) + (-x90) ----
            if reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)
            reset_frame(qubit.xy.name)
            qubit.align()
            qubit.xy.frame_rotation(np.pi)
            qubit.align()
            qubit.xy.play("x90")
            qubit.align()
            qubit.xy.frame_rotation(-np.pi)
            qubit.align()
            qubit.xy.frame_rotation(np.pi)
            qubit.align()
            qubit.xy.play("x90")
            qubit.align()
            qubit.xy.frame_rotation(-np.pi)
            qubit.align()
            readout_state(qubit, state)
            save(state, st)

        with stream_processing():
            st.buffer(6).average().save("p1_cases")

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog)
        job.result_handles.wait_for_all_values()
        p1_cases = job.result_handles.get("p1_cases").fetch_all()

    p1_id, p1_x180, p1_x90, p1_x90x2, p1_x90_xm90, p1_m90_m90 = map(float, p1_cases)


    # Convert to <Z> without SPAM correction
    Z_raw = [
            1 - 2*p1_id,
            1 - 2*p1_x180,
            1 - 2*p1_x90,
            1 - 2*p1_x90x2,
            1 - 2*p1_x90_xm90,
            1 - 2*p1_m90_m90,
        ]

    print("\nXY gate sanity check (raw):")
    print(f"  Id:        P1={p1_id:.4f}     <Z>={Z_raw[0]:+.4f}   (expect ~ +1)")
    print(f"  x180:      P1={p1_x180:.4f}   <Z>={Z_raw[1]:+.4f}   (expect ~ -1)")
    print(f"  x90:       P1={p1_x90:.4f}    <Z>={Z_raw[2]:+.4f}   (expect ~  0)")
    print(f"  x90+x90:   P1={p1_x90x2:.4f}  <Z>={Z_raw[3]:+.4f}   (expect ~ -1)")
    print(f"  x90-x90:   P1={p1_x90_xm90:.4f}  <Z>={Z_raw[4]:+.4f}   (expect ~ +1)")
    print(f"  -x90-x90:  P1={p1_m90_m90:.4f}   <Z>={Z_raw[5]:+.4f}   (expect ~ -1)")

    return {
        "p1": [p1_id, p1_x180, p1_x90, p1_x90x2, p1_x90_xm90, p1_m90_m90],
        "Z_raw": Z_raw
    }

# =========================
# Main execution
# =========================
all_results = {}
for q in qubits[:1]:
    all_results[q.name] = run_xy_gate_sanity(q)

node.results = {"cooling_rate_debug": all_results}
node.save()