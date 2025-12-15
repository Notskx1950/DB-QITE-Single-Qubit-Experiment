"""
DB-QITE Comprehensive Debug using readout_state macro.
This version uses the same patterns as the working Z-gate calibration.
"""
from typing import Optional, List, Literal
import numpy as np
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *

# =========================
# Theory functions
# =========================
from dbqite_theory_sim import (
    simulate_dbqite_single_qubit,
    make_H,
    ground_state,
    hamiltonian_axis,
)

# =========================
# Parameters
# =========================
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q4"]
    
    DBQITE_METHOD: Literal["GC", "HOPF"] = "GC"
    
    # Hamiltonian parameters
    DELTA: float = 1.0
    OMEGA: float = 0.7
    
    # Initial state angle
    THETA_INIT: float = np.pi / 4.0
    
    # Energy rotation scan
    A_ENERGY_SCAN_POINTS: int = 21
    A_ENERGY_SCAN_RANGE: float = 0.5  # ±50% around nominal
    
    # Step pulse scan
    A_STEP_SCAN_POINTS: int = 11
    A_STEP_MAX: float = 0.3
    
    num_averages: int = 5000
    reset_type: Literal["thermal", "active"] = "active"
    thermal_reset_time_ns: int = 200000
    
    simulate: bool = False
    timeout: int = 100

node = QualibrationNode(name="DBQITE_Debug_v2", parameters=Parameters())

# =========================
# Initialize
# =========================
u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()

if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

num_qubits = len(qubits)

# =========================
# Helper math
# =========================
psi_zero = np.array([1.0, 0.0], dtype=complex)

def Ry(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def wrap_to_pi(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def theta_from_state_on_ry_meridian(psi):
    a = psi[0]
    ph = np.angle(a) if np.abs(a) > 1e-12 else 0.0
    psi2 = psi * np.exp(-1j * ph)
    a, b = np.real_if_close(psi2[0]), np.real_if_close(psi2[1])
    return wrap_to_pi(2.0 * np.arctan2(np.real(b), np.real(a)))

def get_hamiltonian_norm(delta, omega):
    H = make_H(delta, omega)
    evals = np.linalg.eigvalsh(H)
    return (evals[1] - evals[0]) / 2.0

def theta_hamiltonian_axis(delta, omega):
    return wrap_to_pi(np.arctan2(omega, delta))

def correct_p1_spam(p1_raw, p1_when_0, p1_when_1):
    denom = p1_when_1 - p1_when_0
    if abs(denom) < 0.01:
        return p1_raw
    return np.clip((p1_raw - p1_when_0) / denom, 0, 1)

# =========================
# Run debug
# =========================


def run_debug(qubit):
    n_avg = node.parameters.num_averages
    reset_type = node.parameters.reset_type
    
    delta = node.parameters.DELTA
    omega = node.parameters.OMEGA
    theta_init = float(node.parameters.THETA_INIT)
    
    # Theory calculations
    H = make_H(delta, omega)
    evals, evecs = np.linalg.eigh(H)
    Hnorm = (evals[1] - evals[0]) / 2.0
    theta_h = theta_hamiltonian_axis(delta, omega)
    
    a_prep = float(theta_init / np.pi)
    a_energy_nominal = float(-theta_h / np.pi)
    
    # Expected values
    psi_prep = Ry(theta_init) @ psi_zero
    Z_expected = np.abs(psi_prep[0])**2 - np.abs(psi_prep[1])**2
    X_expected = 2 * np.real(psi_prep[0] * np.conj(psi_prep[1]))
    P1_expected = np.abs(psi_prep[1])**2
    E_expected = 0.5 * (delta * Z_expected + omega * X_expected)
    
    print("\n" + "=" * 70)
    print("DB-QITE DEBUG v2 (using readout_state macro)")
    print("=" * 70)
    print(f"\nHamiltonian: H = 0.5*({delta}*Z + {omega}*X)")
    print(f"Eigenvalues: {evals}")
    print(f"Hnorm = {Hnorm:.6f}")
    print(f"theta_h = {theta_h:.4f} rad = {np.degrees(theta_h):.2f} deg")
    print(f"theta_init = {theta_init:.4f} rad = {np.degrees(theta_init):.2f} deg")
    print(f"a_prep = {a_prep:.4f}, a_energy_nominal = {a_energy_nominal:.4f}")
    print(f"\nExpected: P1={P1_expected:.4f}, ⟨Z⟩={Z_expected:.4f}, ⟨X⟩={X_expected:.4f}, E={E_expected:.4f}")
    
    results = {}
    qmm = machine.connect()
    
    def do_reset_and_frame():
        """Reset qubit and reset XY frame (good hygiene; prevents phase accumulation)."""
        if reset_type == "active":
            active_reset(qubit, "readout")
        else:
            qubit.wait(node.parameters.thermal_reset_time_ns * u.ns)

        reset_frame(qubit.xy.name)   # per-shot / per-point
        qubit.align()


    def ry_from_a(a, scale_tmp=None):
        """
        Ry(theta=a*pi) using y90/-y90 scaling.
        RY_SIGN lets you flip the convention (hardware vs theory).
        RY_SMALL_ANGLE_GAIN corrects under-rotation.
        """
        RY_SMALL_ANGLE_GAIN = 1.0   # start with this; later fit it from step_calib
        RY_SIGN = -1  
        # constant case
        if isinstance(a, (float, int, np.floating)):
            a = float(a) * RY_SIGN
            s = RY_SMALL_ANGLE_GAIN * (2.0 * abs(a))  # base scale is 2|a|
            if a >= 0:
                qubit.xy.play("y90", amplitude_scale=s)
            else:
                qubit.xy.play("-y90", amplitude_scale=s)
            qubit.align()
            return

        # QUA var case
        if scale_tmp is None:
            raise ValueError("need scale_tmp for QUA fixed a")

        with if_(a >= 0):
            # scale_tmp = gain * 2a
            assign(scale_tmp, a + a)                 # 2a
            assign(scale_tmp, scale_tmp * RY_SMALL_ANGLE_GAIN)
            qubit.xy.play("y90", amplitude_scale=scale_tmp)
        with else_():
            # scale_tmp = gain * 2|a| = gain * (-2a)
            assign(scale_tmp, 0 - a)
            assign(scale_tmp, scale_tmp + scale_tmp)
            assign(scale_tmp, scale_tmp * RY_SMALL_ANGLE_GAIN)
            qubit.xy.play("-y90", amplitude_scale=scale_tmp)
        qubit.align()


    def meas_X_prerot_candidateB():
        """
        X measurement pre-rotation (Candidate B from your sanity test):
        Apply -Y90 then measure Z => returns <X>.
        """
        qubit.xy.play("-y90")
        qubit.align()

    # =========================================
    # TEST 1: READOUT CALIBRATION
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 1: READOUT CALIBRATION")
    print("-" * 70)
    
    with program() as prog_calib:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        state_st = declare_stream()
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # |0>
            do_reset_and_frame()
            readout_state(qubit, state)
            save(state, state_st)

            # |1> via x180
            do_reset_and_frame()
            qubit.xy.play("x180")
            qubit.align()
            readout_state(qubit, state)
            save(state, state_st)

            # |1> via y180 (native)
            do_reset_and_frame()
            qubit.xy.play("y180")
            qubit.align()
            readout_state(qubit, state)
            save(state, state_st)
        
        with stream_processing():
            n_st.save("n")
            state_st.buffer(3).average().save("p1_calib")
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog_calib)
        job.result_handles.wait_for_all_values()
        p1_calib = job.result_handles.get("p1_calib").fetch_all()
    
    P1_when_0, P1_when_1_x, P1_when_1_y = p1_calib[0], p1_calib[1], p1_calib[2]
    contrast = P1_when_1_x - P1_when_0
    
    print(f"P(1|0⟩) = {P1_when_0:.4f}, P(1|1⟩_x180) = {P1_when_1_x:.4f}, P(1|1⟩_y180) = {P1_when_1_y:.4f}")
    print(f"Contrast = {contrast:.4f}, Avg fidelity = {(1-P1_when_0+P1_when_1_x)/2:.4f}")
    
    results["readout"] = {"P1_when_0": P1_when_0, "P1_when_1": P1_when_1_x, "contrast": contrast}
    
    # =========================================
    # TEST 2: PREP VERIFICATION (TOMOGRAPHY)
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 2: PREP VERIFICATION (TOMOGRAPHY)")
    print("-" * 70)
    
    with program() as prog_tomo:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        state_st = declare_stream()
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            # Measure Z: prep -> Z
            do_reset_and_frame()
            ry_from_a(a_prep)                # a_prep is Python float
            readout_state(qubit, state)
            save(state, state_st)

            # Measure X: prep -> (-y90) -> Z
            do_reset_and_frame()
            ry_from_a(a_prep)
            meas_X_prerot_candidateB()       # native -y90
            readout_state(qubit, state)
            save(state, state_st)
        
        with stream_processing():
            n_st.save("n")
            state_st.buffer(2).average().save("p1_tomo")
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog_tomo)
        job.result_handles.wait_for_all_values()
        p1_tomo = job.result_handles.get("p1_tomo").fetch_all()
    
    P1_z_raw, P1_x_raw = p1_tomo[0], p1_tomo[1]
    P1_z_corr = correct_p1_spam(P1_z_raw, P1_when_0, P1_when_1_x)
    P1_x_corr = correct_p1_spam(P1_x_raw, P1_when_0, P1_when_1_x)
    
    Z_corr = 1 - 2*P1_z_corr
    X_corr = 1 - 2*P1_x_corr
    E_tomo = 0.5 * (delta * Z_corr + omega * X_corr)
    
    print(f"Raw: P1_z={P1_z_raw:.4f}, P1_x={P1_x_raw:.4f}")
    print(f"Corrected: P1_z={P1_z_corr:.4f}, P1_x={P1_x_corr:.4f}")
    print(f"⟨Z⟩={Z_corr:.4f} (exp: {Z_expected:.4f}), ⟨X⟩={X_corr:.4f} (exp: {X_expected:.4f})")
    print(f"E_tomo={E_tomo:.4f} (exp: {E_expected:.4f}), error={abs(E_tomo-E_expected):.4f}")
    
    results["tomo"] = {"Z": Z_corr, "X": X_corr, "E": E_tomo, "E_expected": E_expected}

    # =========================================
    # TEST 3: ENERGY ROTATION CALIBRATION
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 3: ENERGY ROTATION CALIBRATION")
    print("-" * 70)
    
    a_energy_range = node.parameters.A_ENERGY_SCAN_RANGE
    a_energy_list = np.linspace(
        a_energy_nominal * (1 - a_energy_range),
        a_energy_nominal * (1 + a_energy_range),
        node.parameters.A_ENERGY_SCAN_POINTS
    )
    n_energy_pts = len(a_energy_list)
    
    with program() as prog_energy:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        state_st = declare_stream()
        idx = declare(int)
        a_energy = declare(fixed)
        a_energy_arr = declare(fixed, value=list(a_energy_list))
        
        scale_tmp = declare(fixed)  # <-- temp for ry_from_a with QUA vars

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(idx, 0, idx < n_energy_pts, idx + 1):
                assign(a_energy, a_energy_arr[idx])

                do_reset_and_frame()
                ry_from_a(a_prep)                    # constant
                ry_from_a(a_energy, scale_tmp)       # QUA var, can be negative
                readout_state(qubit, state)
                save(state, state_st)

        
        with stream_processing():
            n_st.save("n")
            state_st.buffer(n_energy_pts).average().save("p1_energy")
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog_energy)
        job.result_handles.wait_for_all_values()
        p1_energy = job.result_handles.get("p1_energy").fetch_all()
    
    p1_energy_corr = np.array([correct_p1_spam(p, P1_when_0, P1_when_1_x) for p in p1_energy])
    z_energy = 1 - 2*p1_energy_corr
    E_energy = Hnorm * z_energy
    
    idx_best = np.argmin(np.abs(E_energy - E_expected))
    a_energy_calibrated = a_energy_list[idx_best]
    cal_factor = a_energy_calibrated / a_energy_nominal if a_energy_nominal != 0 else 1
    
    print(f"Scanning a_energy: {a_energy_list[0]:.4f} to {a_energy_list[-1]:.4f}")
    print(f"Nominal a_energy = {a_energy_nominal:.4f}")
    print(f"Best a_energy = {a_energy_calibrated:.4f} (gives E={E_energy[idx_best]:.4f})")
    print(f"Calibration factor = {cal_factor:.4f}")
    
    results["energy_scan"] = {
        "a_energy_list": a_energy_list, "E_energy": E_energy,
        "a_energy_calibrated": a_energy_calibrated, "cal_factor": cal_factor
    }
    
    # =========================================
    # TEST 4: STEP PULSE CALIBRATION
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 4: STEP PULSE CALIBRATION")
    print("-" * 70)
    
    a_step_list = np.linspace(0, node.parameters.A_STEP_MAX, node.parameters.A_STEP_SCAN_POINTS)
    n_step_pts = len(a_step_list)
    
    with program() as prog_step:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        state_z_st = declare_stream()
        state_x_st = declare_stream()
        idx = declare(int)
        a_step = declare(fixed)
        a_step_arr = declare(fixed, value=list(a_step_list))
        
        scale_tmp = declare(fixed)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(idx, 0, idx < n_step_pts, idx + 1):
                assign(a_step, a_step_arr[idx])

                # Measure Z after step
                do_reset_and_frame()
                ry_from_a(a_step, scale_tmp)
                readout_state(qubit, state)
                save(state, state_z_st)

                # Measure X after step
                do_reset_and_frame()
                ry_from_a(a_step, scale_tmp)
                meas_X_prerot_candidateB()
                readout_state(qubit, state)
                save(state, state_x_st)

        
        with stream_processing():
            n_st.save("n")
            state_z_st.buffer(n_step_pts).average().save("p1_z")
            state_x_st.buffer(n_step_pts).average().save("p1_x")
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog_step)
        job.result_handles.wait_for_all_values()
        p1_z = job.result_handles.get("p1_z").fetch_all()
        p1_x = job.result_handles.get("p1_x").fetch_all()
    
    p1_z_corr = np.array([correct_p1_spam(p, P1_when_0, P1_when_1_x) for p in p1_z])
    p1_x_corr = np.array([correct_p1_spam(p, P1_when_0, P1_when_1_x) for p in p1_x])
    
    Z_step = 1 - 2*p1_z_corr
    X_step = 1 - 2*p1_x_corr
    theta_actual = -np.arctan2(X_step, Z_step)
    theta_commanded = a_step_list * np.pi
    
    print(f"{'a_step':<10} {'θ_cmd':<10} {'⟨Z⟩':<10} {'⟨X⟩':<10} {'θ_actual':<10} {'ratio':<10}")
    for i in range(n_step_pts):
        ratio = theta_actual[i]/theta_commanded[i] if theta_commanded[i] != 0 else 0
        print(f"{a_step_list[i]:<10.4f} {theta_commanded[i]:<10.4f} {Z_step[i]:<10.4f} {X_step[i]:<10.4f} {theta_actual[i]:<10.4f} {ratio:<10.4f}")
    
    # Fit
    mask = a_step_list > 0.02
    if np.sum(mask) > 2:
        slope, intercept = np.polyfit(theta_commanded[mask], theta_actual[mask], 1)
        print(f"\nFit: θ_actual = {slope:.4f} * θ_commanded + {intercept:.4f}")
    
    results["step_calib"] = {"a_step_list": a_step_list, "Z": Z_step, "X": X_step, "theta_actual": theta_actual}
    
    # =========================================
    # TEST 5: E0 STABILITY (ALL IN ONE PROGRAM)
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 5: E0 STABILITY TEST")
    print("-" * 70)
    
    a_step_test = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    n_test_pts = len(a_step_test)
    
    with program() as prog_stability:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        state_e0_st = declare_stream()
        state_e1_st = declare_stream()
        idx = declare(int)
        a_step = declare(fixed)
        a_step_arr = declare(fixed, value=list(a_step_test))
        
        scale_tmp = declare(fixed)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(idx, 0, idx < n_test_pts, idx + 1):
                assign(a_step, a_step_arr[idx])

                # E0: prep only
                do_reset_and_frame()
                ry_from_a(a_prep)
                readout_state(qubit, state)
                save(state, state_e0_st)

                # E1: prep + step
                do_reset_and_frame()
                ry_from_a(a_prep)
                ry_from_a(a_step, scale_tmp)
                readout_state(qubit, state)
                save(state, state_e1_st)
        
        with stream_processing():
            n_st.save("n")
            state_e0_st.buffer(n_test_pts).average().save("p1_e0")
            state_e1_st.buffer(n_test_pts).average().save("p1_e1")
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(prog_stability)
        job.result_handles.wait_for_all_values()
        p1_e0 = job.result_handles.get("p1_e0").fetch_all()
        p1_e1 = job.result_handles.get("p1_e1").fetch_all()
    
    p1_e0_corr = np.array([correct_p1_spam(p, P1_when_0, P1_when_1_x) for p in p1_e0])
    p1_e1_corr = np.array([correct_p1_spam(p, P1_when_0, P1_when_1_x) for p in p1_e1])
    
    Z_e0 = 1 - 2*p1_e0_corr
    Z_e1 = 1 - 2*p1_e1_corr
    
    print(f"{'a_step':<10} {'P1_E0':<10} {'P1_E1':<10} {'⟨Z⟩_E0':<10} {'⟨Z⟩_E1':<10} {'Δ⟨Z⟩':<10}")
    for i in range(n_test_pts):
        print(f"{a_step_test[i]:<10.4f} {p1_e0_corr[i]:<10.4f} {p1_e1_corr[i]:<10.4f} {Z_e0[i]:<10.4f} {Z_e1[i]:<10.4f} {Z_e1[i]-Z_e0[i]:<10.4f}")
    
    print(f"\nE0 std across s: {np.std(Z_e0):.4f}")
    if np.std(Z_e0) < 0.05:
        print("✓ E0 is stable")
    else:
        print("⚠️ E0 varies with s")
    
    results["stability"] = {"a_step_test": a_step_test, "Z_e0": Z_e0, "Z_e1": Z_e1}
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n1. READOUT: {'✓ Good' if contrast > 0.7 else '⚠️ Low'} (contrast={contrast:.4f})")
    
    tomo_err = abs(E_tomo - E_expected)
    print(f"2. PREP: {'✓ Good' if tomo_err < 0.05 else '⚠️ Mismatch'} (E_error={tomo_err:.4f})")
    
    print(f"3. ENERGY ROTATION: {'✓ OK' if abs(cal_factor-1) < 0.15 else '⚠️ Needs cal'} (factor={cal_factor:.4f})")
    
    e0_std = np.std(Z_e0)
    print(f"4. E0 STABILITY: {'✓ Stable' if e0_std < 0.05 else '⚠️ Varies'} (std={e0_std:.4f})")
    
    # =========================================
    # PLOTS
    # =========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy rotation scan
    ax1 = axes[0, 0]
    ax1.plot(a_energy_list, E_energy, 'o-')
    ax1.axhline(E_expected, color='r', linestyle='--', label=f'E_expected={E_expected:.3f}')
    ax1.axvline(a_energy_nominal, color='g', linestyle=':', label=f'nominal={a_energy_nominal:.3f}')
    ax1.axvline(a_energy_calibrated, color='b', linestyle=':', label=f'calibrated={a_energy_calibrated:.3f}')
    ax1.set_xlabel('a_energy')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Rotation Calibration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Step calibration
    ax2 = axes[0, 1]
    ax2.plot(theta_commanded, theta_actual, 'o-', label='Measured')
    ax2.plot([0, max(theta_commanded)], [0, max(theta_commanded)], 'r--', label='Ideal')
    ax2.set_xlabel('θ commanded (rad)')
    ax2.set_ylabel('θ actual (rad)')
    ax2.set_title('Step Pulse Calibration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bloch coords
    ax3 = axes[1, 0]
    ax3.plot(a_step_list, Z_step, 'o-', label='⟨Z⟩')
    ax3.plot(a_step_list, X_step, 's-', label='⟨X⟩')
    ax3.axhline(Z_expected, color='b', linestyle='--', alpha=0.5)
    ax3.set_xlabel('a_step')
    ax3.set_ylabel('Expectation')
    ax3.set_title('Step: Bloch Coordinates')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # E0 stability
    ax4 = axes[1, 1]
    ax4.plot(a_step_test, Z_e0, 'o-', label='⟨Z⟩ E0')
    ax4.plot(a_step_test, Z_e1, 's-', label='⟨Z⟩ E1')
    ax4.axhline(Z_expected, color='r', linestyle='--', label=f'⟨Z⟩ expected={Z_expected:.3f}')
    ax4.set_xlabel('a_step')
    ax4.set_ylabel('⟨Z⟩')
    ax4.set_title('E0/E1 Stability')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


# =========================
# Main
# =========================
all_results = {}
for qubit in qubits[:1]:
    all_results[qubit.name] = run_debug(qubit)

node.results = {"debug_v2": all_results}
node.machine = machine
node.save()