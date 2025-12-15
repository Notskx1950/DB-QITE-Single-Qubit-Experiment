# DB-QITE Single-Qubit Experiment Suite

This repository contains Python scripts for running Deterministic Bayesian Quantum Imaginary Time Evolution (DB-QITE) experiments on a single qubit using Quantum Machines' OPX hardware and the QuAM framework. To sucessfully run the python file, the code relies on the qolab-start library.

## Overview

DB-QITE is an algorithm that cools a quantum state toward the ground state of a Hamiltonian through iterative unitary operations. These scripts implement hardware experiments to validate DB-QITE theory predictions.

**Hamiltonian:** $\hat H = \frac{\Delta}{2} Z + \frac{\Omega}{2} X$

## File Descriptions

### Debug & Calibration Scripts

| File | Purpose |
|------|---------|
| `30_X_Y_Z_debug.py` | XY gate sanity check - verifies x90, x180, y90, y180 gates work correctly |
| `29_elementary_debug.py` | Comprehensive 5-test debug suite: readout calibration, prep verification, energy rotation calibration, step pulse calibration, and E0 stability |

### Main Experiment Scripts

| File | Purpose |
|------|---------|
| `24_dbqite_deltaF_vs_gap.py` | Figure 1 reproduction - compares fidelity gain `ΔF` vs spectral gap for two different Hamiltonians (A and B) |
| `25_dbqite_coolingrate_vs_step_size` | Measures cooling rate `(E₁-E₀)/s` vs step size `s` and compares to `-2V₀` (theory prediction) |
| `26_dbqite_fidelity_vs_step.py` | Tracks ground-state fidelity `Fₖ` over multiple DB-QITE steps for cold and warm starts |
| `27_dbqite_energy_vs_step.py` | Tracks energy `Eₖ` over multiple DB-QITE steps for cold and warm starts |
| `28_dbqite_fidelity_gain_vs_f_i.py` | Measures fidelity gain `ΔF = F₁ - F₀` as a function of initial fidelity `F₀` |

### Theory Simulation

| File | Purpose |
|------|---------|
| `dbqite_theory_sim.py` | Pure Python theory simulation - implements GC (Group Commutator) and HOPF (Higher-Order Product Formula) DB-QITE methods |
|`theory_demonstration.ipynb`| Jupyter notebook that runs the theory simulations and reproduces the paper-style figures (step-size scan, gap dependence, initial-fidelity dependence, cold vs warm start comparisons), with plotting + parameter sweeps for analysis |

---

## Versioned Parameters

Each script uses Qualibrate's `NodeParameters` class for configuration. Below are the key parameters organized by category.

### Hamiltonian Parameters

| Parameter | Description | Default | Used In |
|-----------|-------------|---------|---------|
| `DELTA` | Z coefficient (Δ) | 1.0 | All experiments |
| `OMEGA` | X coefficient (Ω) | 0.7 | All experiments |
| `DELTA_A`, `OMEGA_A` | Hamiltonian A for Fig1 | 1.0, 0.7 | `24_dbqite_deltaF_vs_gap.py` |
| `DELTA_B`, `OMEGA_B` | Hamiltonian B for Fig1 | 0.4, 0.2 | `24_dbqite_deltaF_vs_gap.py` |

### DB-QITE Method Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `DBQITE_METHOD` | Approximation scheme | `"GC"` | `"GC"`, `"HOPF"` |
| `S_CONV` | Fixed step size for convergence experiments | 0.02 | Float |
| `S_MIN`, `S_MAX` | Step size scan range | 0.1, 0.5 | Float |
| `N_S_POINTS` | Number of step sizes to scan | 5 | Int |
| `S_GAP` | Step size for single-step experiments | 0.02 | Float |

### Initial State Parameters

| Parameter | Description | Default | Used In |
|-----------|-------------|---------|---------|
| `THETA_INIT` | Initial state angle (rad) for `Ry(θ)|0⟩` | π/4 | Cooling rate, debug |
| `TARGET_F0_WARM` | Target fidelity for warm start | 0.6 | Fidelity/Energy vs step |
| `F0_target` | Target initial fidelity | 0.10 | Fig1 experiment |
| `THETA_MIN`, `THETA_MAX` | Angle scan range | 0, π | Fidelity gain vs initial |
| `N_THETA_POINTS` | Number of angles to scan | 15 | Fidelity gain vs initial |

### Convergence Parameters

| Parameter | Description | Default | Used In |
|-----------|-------------|---------|---------|
| `N_STEPS` | Number of DB-QITE steps | 5 | Fidelity/Energy vs step |

### Measurement Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `num_averages` | Shots per measurement point | 3000-9000 | Higher = better statistics |
| `reset_type_thermal_or_active` | Reset method | `"thermal"` | `"thermal"` or `"active"` |
| `thermal_reset_time_ns` | Thermal reset wait time | 100000-460000 | Should be ≥5×T₁ |
| `depletion_time_ns` | Resonator depletion wait | 4000 | After active reset |

### Calibration Scan Parameters (Debug Script)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `A_ENERGY_SCAN_POINTS` | Energy rotation scan points | 21 |
| `A_ENERGY_SCAN_RANGE` | Energy rotation scan range (±%) | 0.5 |
| `A_STEP_SCAN_POINTS` | Step pulse calibration points | 11 |
| `A_STEP_MAX` | Maximum step amplitude | 0.3 |

### Simulation/Debug Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `simulate` | Run in simulation mode | `False` |
| `simulation_duration_ns` | Simulation duration | 4000 |
| `timeout` | QM session timeout (seconds) | 100-200 |

---

## Key Concepts

### Measurement Protocol

All experiments use the same measurement approach:

1. **State Preparation:** `Ry(θ)|0⟩` using `y180` pulse with amplitude scaling
2. **Energy Measurement:** Rotate to Hamiltonian eigenbasis with `Ry(-θ_h)` where `θ_h = atan2(Ω, Δ)`
3. **Fidelity Measurement:** Rotate to ground state basis with `Ry(-θ_gs)`
4. **Z Readout:** Use `readout_state` macro for state discrimination

### Amplitude Scaling Convention

All rotations use `y180` with amplitude scaling:
- `Ry(θ)` → `qubit.xy.play("y180", amplitude_scale=θ/π)`
- Example: `Ry(π/4)` → `amplitude_scale=0.25`

### Code Style (Debug Style)

All corrected scripts use QuAM macros consistently:

```python
# Reset and prepare
if reset_type == "active":
    active_reset(qubit, "readout")
else:
    qubit.wait(thermal_reset_time_ns * u.ns)
reset_frame(qubit.xy.name)
qubit.align()

# Apply rotation
qubit.xy.play("y180", amplitude_scale=a_prep)
qubit.align()

# Measure
readout_state(qubit, state)
```
---

## Experiment Workflow

### Recommended Order

1. **Run sanity check** (`30_X_Y_Z_debug.py`)
   - Verify: Id→⟨Z⟩≈+1, x180→⟨Z⟩≈-1, x90→⟨Z⟩≈0
   - If x180 gives ⟨Z⟩≈-0.7 instead of -1, recalibrate pulse amplitude

2. **Run comprehensive debug** (`29_elementary_debug.py`)
   - Check readout contrast (should be >0.7)
   - Verify prep state tomography matches theory
   - Calibrate energy rotation if needed
   - Check E0 stability across step sizes

3. **Run main experiments**
   - Cooling rate vs step size
   - Fidelity/Energy vs step number
   - Fidelity gain vs initial fidelity
   - Figure 1 (ΔF vs spectral gap)

---

## Theory Reference

The DB-QITE step using Group Commutator (GC) approximation:

$$|\psi_{k+1}\rangle = e^{i\sqrt{s} H} e^{i\sqrt{s} |\psi_k\rangle\langle\psi_k|} e^{-i\sqrt{s} H} |\psi_k\rangle$$

Key observables:
- **Energy:** $E = \langle H \rangle = \frac{\Delta}{2}\langle Z \rangle + \frac{\Omega}{2}\langle X \rangle$
- **Variance:** $V = \langle H^2 \rangle - \langle H \rangle^2$
- **Fidelity:** $F = |\langle \text{gs} | \psi \rangle|^2$
- **Cooling rate:** $\frac{dE}{ds} \approx \frac{E_1 - E_0}{s} \to -2V_0$ as $s \to 0$
- **Spectral gap:** $\Delta_{\text{gap}} = 2\sqrt{\Delta^2 + \Omega^2}$

---

## Output

Each script produces:
- Console output with measurement results and theory comparisons
- Summary tables comparing hardware vs theory
- PNG plots saved to current directory
- Results saved via `node.save()` for Qualibrate integration

---

## Dependencies

- `qualibrate` - Calibration framework
- `quam_libs` - QuAM components and macros
- `qualang_tools` - QUA utilities
- `qm-qua` - Quantum Machines QUA language
- `numpy`, `matplotlib` - Standard scientific Python

---

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| x180 gives ⟨Z⟩≈-0.7 | Pulse amplitude miscalibrated | Re-run power_rabi calibration |
| E0 varies with step size | Reset not working properly | Increase thermal reset time or use active reset |
| Theory/hardware mismatch | Sign convention mismatch | Check `RY_SIGN` in debug script, flip if needed |
| Low readout contrast | IQ blob overlap | Re-run readout optimization |
| SPAM correction fails | Contrast < 0.01 | Check discriminator threshold and angle |

---

## Quick Start

```python
# 1. Modify parameters in script
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q4"]  # Your qubit
    DELTA: float = 1.0
    OMEGA: float = 0.7
    num_averages: int = 5000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    thermal_reset_time_ns: int = 200000

# 2. Run script
python dbqite_sanity_check.py

# 3. Check output
# - Console prints measurement results
# - Plots appear/save automatically
# - Results saved to Qualibrate node
```