import numpy as np
from numpy.linalg import norm, eigh

# ============================================================
# Basic single-qubit objects
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)
Z = np.array([[1,  0],
              [0, -1]], dtype=complex)


# ------------------------------------------------------------
# Hamiltonian and ground state
# ------------------------------------------------------------

def make_H(delta, omega_x):
    """
    Single-qubit Hamiltonian:
        H = (delta/2) * Z + (omega_x/2) * X.
    """
    return 0.5 * (delta * Z + omega_x * X)


def ground_state(H):
    """
    Return (E0, psi0), where E0 is the ground-state energy (float)
    and psi0 is the normalized ground-state eigenvector (shape (2,)).
    """
    vals, vecs = eigh(H)
    idx = np.argmin(vals.real)
    psi0 = vecs[:, idx]
    psi0 /= norm(psi0)
    return vals[idx].real, psi0


def hamiltonian_axis(delta, omega_x):
    """
    Return (gap, n_hat) where
        gap  = sqrt(delta^2 + omega_x^2),
        n_hat = (omega_x, 0, delta) / gap
    is the Bloch-axis of the Hamiltonian.
    """
    gap = np.sqrt(delta**2 + omega_x**2)
    if gap == 0.0:
        return 0.0, np.array([0.0, 0.0, 1.0])
    n = np.array([omega_x, 0.0, delta])
    n /= gap
    return gap, n


# ------------------------------------------------------------
# Basic expectations
# ------------------------------------------------------------

def expectation(psi, O):
    """⟨psi| O |psi⟩ for normalized 2-component state psi."""
    return np.vdot(psi, O @ psi)


def bloch_vector(psi):
    """Return Bloch vector (⟨X⟩,⟨Y⟩,⟨Z⟩) for normalized psi."""
    return np.array([
        np.real(expectation(psi, X)),
        np.real(expectation(psi, Y)),
        np.real(expectation(psi, Z)),
    ])


# ============================================================
# Exponentials and projectors
# ============================================================

def projector_from_state(psi):
    """Rank-1 projector |psi⟩⟨psi| as a 2x2 matrix."""
    return np.outer(psi, np.conjugate(psi))


def exp_iH(alpha, H):
    """
    U = exp(i * alpha * H) for 2x2 Hermitian H,
    via spectral decomposition.
    """
    vals, vecs = eigh(H)
    phases = np.exp(1j * alpha * vals)
    return vecs @ np.diag(phases) @ vecs.conj().T


def exp_i_projector(alpha, psi):
    """
    U = exp(i * alpha * |psi⟩⟨psi|) using P^2 = P:
        exp(i alpha P) = I + (e^{i alpha} - 1) P.
    psi is assumed normalized.
    """
    P = projector_from_state(psi)
    return I2 + (np.exp(1j * alpha) - 1.0) * P


# ============================================================
# DB-QITE single step (GC and HOPF)
# ============================================================

def dbqite_step_gc(psi_k, H, s_k):
    """
    Single DB-QITE step using the Group Commutator (GC) approximation:
        |ω_{k+1}⟩ = e^{i√s_k H} e^{i√s_k |ω_k⟩⟨ω_k|}
                    e^{-i√s_k H} |ω_k⟩.
    """
    psi_k = psi_k / norm(psi_k)
    root_s = np.sqrt(s_k)

    U_H = exp_iH(root_s, H)
    U_H_dag = U_H.conj().T
    U_P = exp_i_projector(root_s, psi_k)

    psi_next = U_H @ U_P @ U_H_dag @ psi_k
    psi_next /= norm(psi_next)
    return psi_next


def dbqite_step_hopf(psi_k, H, s_k):
    """
    Single DB-QITE step using the higher-order product formula (HOPF)
    approximation (matching the QITE docstring):

        U_{k+1} = e^{i φ√s_k H} e^{i φ√s_k ω_k} e^{-i√s_k H}
                  e^{-i(1+φ)√s_k ω_k} e^{i(1-φ)√s_k H} U_k

    so on the state level:

        |ω_{k+1}⟩ = e^{i φ√s_k H} e^{i φ√s_k ω_k} e^{-i√s_k H}
                    e^{-i(1+φ)√s_k ω_k} e^{i(1-φ)√s_k H} |ω_k⟩,
    where ω_k = |ω_k⟩⟨ω_k|.
    """
    psi_k = psi_k / norm(psi_k)
    root_s = np.sqrt(s_k)
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # golden-ratio-related constant

    # Define the needed unitaries
    U_H_phi = exp_iH(phi * root_s, H)
    U_H_minus1 = exp_iH(-root_s, H)
    U_H_1_minus_phi = exp_iH((1.0 - phi) * root_s, H)

    U_P_phi = exp_i_projector(phi * root_s, psi_k)
    U_P_minus_1_plus_phi = exp_i_projector(-(1.0 + phi) * root_s, psi_k)

    psi_next = U_H_phi @ U_P_phi @ U_H_minus1 @ U_P_minus_1_plus_phi @ U_H_1_minus_phi @ psi_k
    psi_next /= norm(psi_next)
    return psi_next


# ============================================================
# Observables per step
# ============================================================

def energy_and_variance(psi, H):
    """
    Return (E, V) for state psi and Hamiltonian H:
        E = ⟨H⟩,  V = ⟨H^2⟩ - ⟨H⟩^2.
    """
    psi = psi / norm(psi)
    Hpsi = H @ psi
    E = np.vdot(psi, Hpsi).real
    H2psi = H @ Hpsi
    E2 = np.vdot(psi, H2psi).real
    V = E2 - E**2
    return E, V


def fidelity(psi, psi_ref):
    """Fidelity F = |⟨psi_ref | psi⟩|^2."""
    psi = psi / norm(psi)
    psi_ref = psi_ref / norm(psi_ref)
    return np.abs(np.vdot(psi_ref, psi))**2


# ============================================================
# Full DB-QITE simulation (for given s-list)
# ============================================================

def simulate_dbqite_single_qubit(delta,
                                 omega_x,
                                 s_list,
                                 psi0,
                                 method="GC",
                                 return_states=False):
    """
    Theoretical DB-QITE simulation for a single qubit:

      H = (delta/2) Z + (omega_x/2) X
      |ω_0⟩ = psi0
      |ω_{k+1}⟩ = DB-QITE-step(|ω_k⟩, H, s_k)

    Parameters
    ----------
    delta, omega_x : float
        Hamiltonian parameters (Z and X coefficients).
    s_list : sequence of float
        List of step parameters s_k, one per DB-QITE iteration.
        Length of s_list determines the number of steps (K).
    psi0 : np.ndarray, shape (2,)
        Initial normalized state vector (cold or warm start).
    method : {"GC", "HOPF"}
        DB-QITE approximation to use:
          "GC"   : group-commutator scheme (Eq. (12)).
          "HOPF" : higher-order product formula (matching QITE code).
    return_states : bool
        If True, also return the list of |ω_k⟩ states.

    Returns
    -------
    Es, Vs, Fs : np.ndarray
        Arrays of length K+1 with energy E_k, variance V_k,
        and ground-state fidelity F_k for k = 0..K.
    states (optional) : list[np.ndarray]
        List of normalized state vectors |ω_k⟩.
    """
    H = make_H(delta, omega_x)
    E_gs, psi_gs = ground_state(H)

    psi = psi0 / norm(psi0)
    K = len(s_list)

    Es = np.zeros(K + 1, dtype=float)
    Vs = np.zeros(K + 1, dtype=float)
    Fs = np.zeros(K + 1, dtype=float)
    states = [psi.copy()]

    # k = 0
    E0, V0 = energy_and_variance(psi, H)
    F0 = fidelity(psi, psi_gs)
    Es[0], Vs[0], Fs[0] = E0, V0, F0

    # Choose step function
    if method == "GC":
        step_func = dbqite_step_gc
    elif method == "HOPF":
        step_func = dbqite_step_hopf
    else:
        raise ValueError(f"Unknown DB-QITE method '{method}'. Use 'GC' or 'HOPF'.")

    # k = 0..K-1
    for k, s_k in enumerate(s_list):
        psi = step_func(psi, H, s_k)
        states.append(psi.copy())

        E_k, V_k = energy_and_variance(psi, H)
        F_k = fidelity(psi, psi_gs)

        Es[k + 1] = E_k
        Vs[k + 1] = V_k
        Fs[k + 1] = F_k

    if return_states:
        return Es, Vs, Fs, states
    return Es, Vs, Fs
