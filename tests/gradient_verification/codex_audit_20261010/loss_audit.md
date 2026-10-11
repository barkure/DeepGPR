# Repository-only losses and signal-processing gradient audit

Counts: {'PASS': 114, 'FAIL': 0, 'INCONCLUSIVE': 13}. Forward reference checks: 17.

## Scope and provenance

Only the current `src/DeepGPR` package and `examples` notebook **code cells** were inspected. Only CPU synthetic random/Ricker/time-shifted/amplitude-scaled/1e-8-amplitude inputs were executed. Ricker signals came from the actual repository helper. No external experiments, MSSIM packages, desktop data, original `t06_losses.py`, or FDTD solvers were imported or executed. The old external MSSIM failure is outside this repository-only audit.

Actual notebook objectives: 2D uses `torch.nn.L1Loss() * 1e7`, plus anisotropic `TVRegularization(weight_ep=20, weight_sigma=1e-10)`; 3D uses `torch.nn.MSELoss() * 1e4`, plus `TVRegularization(weight_ep=20, weight_sigma=1e-4)`. Envelope/FIR losses below are explicitly labelled audit compositions of actual public signal helpers with official Torch MSE, not invented repository loss classes.

MSSIM/MS-SSIM, OT/Wasserstein/Sinkhorn, cross-correlation misfit and SIREN: **N/A**, absent from the inspected implementation and official example code.

## Method

Every smooth case runs full float64 `torch.autograd.gradcheck` (eps=1e-6, atol=1e-7, rtol=2e-5, fast_mode=False), followed by 3 seeded independent central-FD directions over 17 steps from 2e-2 to 1e-7. Each direction must agree at two adjacent steps within relative 1e-6 (or absolute 1e-9 for a stationary derivative). Numerical differences use forward evaluations only. Vector transform tests check the full output Jacobian and use an independent random projection for directional FD. All steps, AD/FD values, one-sided derivatives, Taylor remainders, errors and gradcheck failure matrices are saved in JSON.

Inputs are expressed as `x=a*u` with fixed `a=max(abs(x))` (or 1 at zero) and the objective is divided by a fixed amplitude scale. This is an exact linear change of coordinates, including its chain rule, and prevents low-amplitude cases from trivially passing an absolute tolerance. No gradients or finite differences are normalized after evaluation to force agreement. FIR frequency/tap-count are fixed nondifferentiable configuration parameters; differentiation is with respect to waveform data only.

## Material finding and minimal fix: TV output precision

Before the fix, `src/DeepGPR/inversion/regularization.py:140` initialized `loss` without the input dtype. Under the normal float32 default, a float64 model produced a float32 TV scalar; all three variants failed both strict float64 gradcheck and independent directional FD. The float32 accumulator rounded forward values while autograd propagated an unrounded derivative. The full before-fix results and source hashes are retained in `before_tv_fix` in JSON, rather than suppressed.

The only production change is `loss = reference.new_zeros(())`, preserving the model dtype/device. All three default-dtype float64 TV variants now pass both checks. The separate default-dtype=float64-context checks are also retained. Across 12 float32 shape/method fixtures, the scalar value and both input-gradient bit patterns are identical before/after: **True**. The existing signal/regularization unittest file, including a dtype/gradcheck regression, runs 12 tests and passes; output is saved in `logs/loss_regression.log`.

## Non-smooth inputs

Exact L1 zero residuals, envelope p=1 at zero analytic amplitude, anisotropic TV at zero differences, unsmoothed isotropic TV at constant models, and legacy TV with partial-zero differences are **INCONCLUSIVE**. A central FD zero at a kink does not establish a unique derivative. Smoothed isotropic TV at constant models and envelope p=2 at zero are smooth and checked normally. Envelope p=1.5 has a first derivative at zero but slower finite-difference convergence, and its measured result is reported.

Eight checks of isotropic/legacy TV with 1e-8-amplitude model values and fixed epsilon=1e-8 are additionally **INCONCLUSIVE** for finite-precision resolution. Their large constant floor relative to the perturbation-induced change causes subtractive cancellation, particularly for the 1e-3-weighted conductivity term. All original mismatching Jacobians and FD curves are retained. Their tolerances were not loosened, and these cases are not counted as passing.

## Results

| Check | Case | Status | Gradcheck | Independent FD |
|---|---|---|---|---|
| L2 (official torch MSELoss) | random | PASS | True | True |
| L1 (official torch L1Loss) | random | PASS | True | True |
| 2D actual data objective L1 x 1e7 | random | PASS | True | True |
| 3D actual data objective MSE x 1e4 | random | PASS | True | True |
| Envelope p=1 followed by MSE | random | PASS | True | True |
| Envelope p=1.5 followed by MSE | random | PASS | True | True |
| Envelope p=2 followed by MSE | random | PASS | True | True |
| FIR cutoff=0.12 followed by MSE | random | PASS | True | True |
| FIR cutoff=0.14 followed by MSE | random | PASS | True | True |
| L2 (official torch MSELoss) | ricker | PASS | True | True |
| L1 (official torch L1Loss) | ricker | PASS | True | True |
| 2D actual data objective L1 x 1e7 | ricker | PASS | True | True |
| 3D actual data objective MSE x 1e4 | ricker | PASS | True | True |
| Envelope p=1 followed by MSE | ricker | PASS | True | True |
| Envelope p=1.5 followed by MSE | ricker | PASS | True | True |
| Envelope p=2 followed by MSE | ricker | PASS | True | True |
| FIR cutoff=0.12 followed by MSE | ricker | PASS | True | True |
| FIR cutoff=0.14 followed by MSE | ricker | PASS | True | True |
| L2 (official torch MSELoss) | shifted | PASS | True | True |
| L1 (official torch L1Loss) | shifted | PASS | True | True |
| 2D actual data objective L1 x 1e7 | shifted | PASS | True | True |
| 3D actual data objective MSE x 1e4 | shifted | PASS | True | True |
| Envelope p=1 followed by MSE | shifted | PASS | True | True |
| Envelope p=1.5 followed by MSE | shifted | PASS | True | True |
| Envelope p=2 followed by MSE | shifted | PASS | True | True |
| FIR cutoff=0.12 followed by MSE | shifted | PASS | True | True |
| FIR cutoff=0.14 followed by MSE | shifted | PASS | True | True |
| L2 (official torch MSELoss) | scaled | PASS | True | True |
| L1 (official torch L1Loss) | scaled | PASS | True | True |
| 2D actual data objective L1 x 1e7 | scaled | PASS | True | True |
| 3D actual data objective MSE x 1e4 | scaled | PASS | True | True |
| Envelope p=1 followed by MSE | scaled | PASS | True | True |
| Envelope p=1.5 followed by MSE | scaled | PASS | True | True |
| Envelope p=2 followed by MSE | scaled | PASS | True | True |
| FIR cutoff=0.12 followed by MSE | scaled | PASS | True | True |
| FIR cutoff=0.14 followed by MSE | scaled | PASS | True | True |
| L2 (official torch MSELoss) | low_amplitude | PASS | True | True |
| L1 (official torch L1Loss) | low_amplitude | PASS | True | True |
| 2D actual data objective L1 x 1e7 | low_amplitude | PASS | True | True |
| 3D actual data objective MSE x 1e4 | low_amplitude | PASS | True | True |
| Envelope p=1 followed by MSE | low_amplitude | PASS | True | True |
| Envelope p=1.5 followed by MSE | low_amplitude | PASS | True | True |
| Envelope p=2 followed by MSE | low_amplitude | PASS | True | True |
| FIR cutoff=0.12 followed by MSE | low_amplitude | PASS | True | True |
| FIR cutoff=0.14 followed by MSE | low_amplitude | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.0, remove_dc=False, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.5, remove_dc=False, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=2.0, remove_dc=False, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.0, remove_dc=True, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.5, remove_dc=True, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=2.0, remove_dc=True, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.0, remove_dc=False, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.5, remove_dc=False, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=2.0, remove_dc=False, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.0, remove_dc=True, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=1.5, remove_dc=True, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=9, p=2.0, remove_dc=True, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.0, remove_dc=False, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.5, remove_dc=False, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=2.0, remove_dc=False, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.0, remove_dc=True, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.5, remove_dc=True, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=2.0, remove_dc=True, legacy=False | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.0, remove_dc=False, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.5, remove_dc=False, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=2.0, remove_dc=False, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.0, remove_dc=True, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=1.5, remove_dc=True, legacy=True | PASS | True | True |
| Hilbert direct Jacobian | nt=10, p=2.0, remove_dc=True, legacy=True | PASS | True | True |
| FIR direct Jacobian | shape=(33,), cutoff=0.12, legacy=False | PASS | True | True |
| FIR direct Jacobian | shape=(33,), cutoff=0.12, legacy=True | PASS | True | True |
| FIR direct Jacobian | shape=(33,), cutoff=0.14, legacy=False | PASS | True | True |
| FIR direct Jacobian | shape=(33,), cutoff=0.14, legacy=True | PASS | True | True |
| FIR direct Jacobian | shape=(2, 33, 3), cutoff=0.12, legacy=False | PASS | True | True |
| FIR direct Jacobian | shape=(2, 33, 3), cutoff=0.12, legacy=True | PASS | True | True |
| FIR direct Jacobian | shape=(2, 33, 3), cutoff=0.14, legacy=False | PASS | True | True |
| FIR direct Jacobian | shape=(2, 33, 3), cutoff=0.14, legacy=True | PASS | True | True |
| TV default-dtype contract | anisotropic | PASS | True | True |
| TV default-dtype contract | isotropic | PASS | True | True |
| TV default-dtype contract | legacy_isotropic | PASS | True | True |
| TV float64 context | anisotropic, (4, 5), random | PASS | True | True |
| TV float64 context | isotropic, (4, 5), random | PASS | True | True |
| TV float64 context | legacy_isotropic, (4, 5), random | PASS | True | True |
| TV float64 context | anisotropic, (4, 5), scaled | PASS | True | True |
| TV float64 context | isotropic, (4, 5), scaled | PASS | True | True |
| TV float64 context | legacy_isotropic, (4, 5), scaled | PASS | True | True |
| TV float64 context | anisotropic, (4, 5), low_amplitude | PASS | True | True |
| TV float64 context | isotropic, (4, 5), low_amplitude | INCONCLUSIVE | False | False |
| TV float64 context | legacy_isotropic, (4, 5), low_amplitude | INCONCLUSIVE | False | True |
| TV float64 context | anisotropic, (4, 5, 1), random | PASS | True | True |
| TV float64 context | isotropic, (4, 5, 1), random | PASS | True | True |
| TV float64 context | legacy_isotropic, (4, 5, 1), random | PASS | True | True |
| TV float64 context | anisotropic, (4, 5, 1), scaled | PASS | True | True |
| TV float64 context | isotropic, (4, 5, 1), scaled | PASS | True | True |
| TV float64 context | legacy_isotropic, (4, 5, 1), scaled | PASS | True | True |
| TV float64 context | anisotropic, (4, 5, 1), low_amplitude | PASS | True | True |
| TV float64 context | isotropic, (4, 5, 1), low_amplitude | INCONCLUSIVE | False | False |
| TV float64 context | legacy_isotropic, (4, 5, 1), low_amplitude | INCONCLUSIVE | False | True |
| TV float64 context | anisotropic, (3, 4, 3), random | PASS | True | True |
| TV float64 context | isotropic, (3, 4, 3), random | PASS | True | True |
| TV float64 context | legacy_isotropic, (3, 4, 3), random | PASS | True | True |
| TV float64 context | anisotropic, (3, 4, 3), scaled | PASS | True | True |
| TV float64 context | isotropic, (3, 4, 3), scaled | PASS | True | True |
| TV float64 context | legacy_isotropic, (3, 4, 3), scaled | PASS | True | True |
| TV float64 context | anisotropic, (3, 4, 3), low_amplitude | PASS | True | True |
| TV float64 context | isotropic, (3, 4, 3), low_amplitude | INCONCLUSIVE | False | False |
| TV float64 context | legacy_isotropic, (3, 4, 3), low_amplitude | INCONCLUSIVE | False | True |
| TV float64 context | anisotropic, (2, 2, 4, 5), random | PASS | True | True |
| TV float64 context | isotropic, (2, 2, 4, 5), random | PASS | True | True |
| TV float64 context | legacy_isotropic, (2, 2, 4, 5), random | PASS | True | True |
| TV float64 context | anisotropic, (2, 2, 4, 5), scaled | PASS | True | True |
| TV float64 context | isotropic, (2, 2, 4, 5), scaled | PASS | True | True |
| TV float64 context | legacy_isotropic, (2, 2, 4, 5), scaled | PASS | True | True |
| TV float64 context | anisotropic, (2, 2, 4, 5), low_amplitude | PASS | True | True |
| TV float64 context | isotropic, (2, 2, 4, 5), low_amplitude | INCONCLUSIVE | False | False |
| TV float64 context | legacy_isotropic, (2, 2, 4, 5), low_amplitude | INCONCLUSIVE | False | True |
| 2D notebook joint objective |  | PASS | True | True |
| 3D notebook joint objective |  | PASS | True | True |
| L1 zero residual |  | INCONCLUSIVE | False | False |
| Envelope at zero | p=1.0 | INCONCLUSIVE | True | True |
| Envelope at zero | p=1.5 | PASS | True | True |
| Envelope squared at zero |  | PASS | True | True |
| TV at constant model | anisotropic | INCONCLUSIVE | True | True |
| TV at constant model | isotropic | PASS | True | True |
| TV at constant model | legacy_isotropic | PASS | True | True |
| TV isotropic epsilon=0 at constant |  | INCONCLUSIVE | False | False |
| TV legacy partial zero differences |  | INCONCLUSIVE | True | True |

## Reproduction and environment

```sh
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python tests/gradient_verification/codex_audit_20261010/loss_verification.py
```

Python: `/home/ll/anaconda3/envs/deepgpr/bin/python` (3.10.16); PyTorch: `1.11.0+cu115`; NumPy: `1.26.4`. OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES empty, torch threads=1, CPU float64. Complete platform information, current Git commit, source SHA256, import path and timestamps are in `data/loss_results.json`.

Before-fix run output: `logs/loss_verification_before.log`; after-fix run output: `logs/loss_verification.log`. For the after-fix audit, the recorded command adds `--after-tv-fix`, preserving the previous results. Regression command: `OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python -m unittest discover -s tests -p test_signal_and_regularization.py -v`. No sandbox escalation was needed. Original gradient-verification files were preserved; the TV accumulator fix and its existing-unit-test regression are the only associated source edits. This audit proves local waveform/regularizer derivatives at tested inputs, not global inversion convergence or FDTD adjoint correctness.
