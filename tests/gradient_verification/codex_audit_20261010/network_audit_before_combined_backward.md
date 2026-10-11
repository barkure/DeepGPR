# Synthetic grid-FWI and SIREN-fixture chain verification

This audit uses the actual public CPU `DeepGPR.compute()` and the official examples' L1×1e7+TV and MSE×1e4+TV objectives on a 20×20 physical grid, 250 time samples, dx=0.02 m, dt=3e-11 s. All observations and models are synthetic. No external experiment/data modules or original gradient-verification scripts were imported. No production network was added.

`SingleSirenFixture` and `DualSirenFixture` are small, self-contained **test fixtures**, not existing repository product networks. Repository SIREN, MSSIM, OT and cross-correlation misfits remain **N/A**. Both output physical mappings are explicit: eps=4+0.25*tanh(out0), sigma=.01+.002*sigmoid(out1). Single has one sin branch (omega=3); Dual has low/high branches (omega=3/9) and a learnable alpha=0.4, fused as low+alpha*high. Parameters are float64.

## Precision and independent FD

Native FDTD supports float32. Fixture float64 fields are cast with a differentiable `.float()` before compute; receiver amplitudes are accumulated in float64 losses. A full solver float64 gradcheck would be invalid and is not claimed. AD comes from the public solver backward. Numerical derivatives come exclusively from fresh `save_wavefield_history=False` forward calls at theta±h*p for 18 steps 0.3…3e-7. Every FD rebuilds CPML through the public API. No boundary mask or frozen-CPML substitute is used.

A direction passes only when two neighboring FD steps both agree with AD within relative 0.002, their FD values are stable within 0.005, and their signal exceeds 10× the measured uncertainty. Twelve actual forward probes perturb each active physical material cell by +/- one representable native float32 ULP; their maximum objective change is a conservative empirical resolution bound. This measures response to quantization-sized inputs, not run-to-run stochastic noise. Deterministic repeated forward values are recorded separately. TV uses a float64 arithmetic bound. FAIL requires a stable FD pair with a resolved signal (10× its resolution bound), and a bias larger than relative 0.005 plus 3× uncertainty. A coarse biased pair is rejected when a finer resolved step approaches AD within 0.5%, or its <1% noise bound covers the discrepancy; that counterevidence gives INCONCLUSIVE, never PASS. This prevents a coarse finite-step abs kink or a spurious plateau from being diagnosed as an adjoint defect. Unresolved/unstable checks are INCONCLUSIVE. The earlier relative-only status and every mismatch are retained. All FD curves and Taylor remainders are retained. The data term, TV term and total are checked separately so a large TV term cannot hide a propagation derivative mismatch. Directions are epsilon-only, sigma-only, joint; fixture low/high/alpha/all groups. None/NaN/Inf and zero parameter-gradient tensors are recorded explicitly.

Activation checkpoint on/off uses identical fixtures, weights, observations and directions for joint inversion. Reentrant checkpoint uses coordinate inputs requiring grad under Torch 1.11. Forward values and gradient SHA256 are compared directly; this checks activation recomputation without claiming the external experiments' checkpoint implementation was audited.

The actual float64 fixture module + bounded material mapping also receives a full parameter-Jacobian `torch.autograd.gradcheck` at four spatial points, via functional parameter substitution (no duplicated forward formula). That check excludes FDTD and is reported separately from native chain FD.

## Counts and checkpoint parity

Case count: 36. Term-direction counts: {'PASS': 139, 'FAIL': 64, 'INCONCLUSIVE': 73}. Checkpoint parity: [{'name': 'SingleSiren', 'pml': 0, 'objective': 'L1x1e7', 'forward_identical': True, 'gradient_identical': True}, {'name': 'DualSiren', 'pml': 0, 'objective': 'L1x1e7', 'forward_identical': True, 'gradient_identical': True}, {'name': 'SingleSiren', 'pml': 0, 'objective': 'MSEx1e4', 'forward_identical': True, 'gradient_identical': True}, {'name': 'DualSiren', 'pml': 0, 'objective': 'MSEx1e4', 'forward_identical': True, 'gradient_identical': True}, {'name': 'SingleSiren', 'pml': 6, 'objective': 'L1x1e7', 'forward_identical': True, 'gradient_identical': True}, {'name': 'DualSiren', 'pml': 6, 'objective': 'L1x1e7', 'forward_identical': True, 'gradient_identical': True}, {'name': 'SingleSiren', 'pml': 6, 'objective': 'MSEx1e4', 'forward_identical': True, 'gradient_identical': True}, {'name': 'DualSiren', 'pml': 6, 'objective': 'MSEx1e4', 'forward_identical': True, 'gradient_identical': True}]

Fixture mapping gradchecks: [{'name': 'SingleSiren', 'fixture_only': True, 'dtype': 'float64', 'FDTD_executed': False, 'eps': 1e-06, 'atol': 1e-07, 'rtol': 2e-05, 'parameter_names': ['low.first.weight', 'low.first.bias', 'low.final.weight', 'low.final.bias'], 'coordinate_indices': [0, 45, 199, 399], 'ok': True, 'error': ''}, {'name': 'DualSiren', 'fixture_only': True, 'dtype': 'float64', 'FDTD_executed': False, 'eps': 1e-06, 'atol': 1e-07, 'rtol': 2e-05, 'parameter_names': ['alpha', 'low.first.weight', 'low.first.bias', 'low.final.weight', 'low.final.bias', 'high.first.weight', 'high.first.bias', 'high.final.weight', 'high.final.bias'], 'coordinate_indices': [0, 45, 199, 399], 'ok': True, 'error': ''}]

Data-term direction counts: PML=0 **{'PASS': 30, 'FAIL': 0, 'INCONCLUSIVE': 16}**; PML=6 **{'PASS': 1, 'FAIL': 32, 'INCONCLUSIVE': 13}**. The PML=0 checks have no confirmed stable failure; unresolved small conductivity directions and L1 finite-step kinks remain INCONCLUSIVE. In PML=6, stable resolved bias survives finer-step review in the FAIL directions. No case had None/NaN/Inf parameter gradients. The two direct mapping gradchecks and all eight checkpoint on/off parity comparisons pass.

The original eight-step run is retained as `pilot_step_sweep` in JSON. The final 18-step numerical run used the updated CPU native library; its hash was checked before and after execution. A separate deterministic classification review, using only saved curves, rejected coarse bias plateaus contradicted by finer resolved observations. Its prior status, changed-check list, timestamp and script hash are retained in `metadata.classification_review` and `logs/network_classification_review.log`. No reviewed case was promoted to PASS.

## Direction results

| Chain | PML | Mode | Loss | Checkpoint | Direction | Data | TV | Total | Data best relative error |
|---|---|---|---|---|---|---|---|---|---|
| Grid-FWI | 0 | joint | L1x1e7 | False | epsilon | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.000108 |
| Grid-FWI | 0 | joint | L1x1e7 | False | sigma | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.00351 |
| Grid-FWI | 0 | joint | L1x1e7 | False | joint | PASS | PASS | PASS | 0.000681 |
| SingleSiren | 0 | epsilon_only | L1x1e7 | False | all | PASS | PASS | PASS | 0.000439 |
| SingleSiren | 0 | sigma_only | L1x1e7 | False | all | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.000216 |
| SingleSiren | 0 | joint | L1x1e7 | False | all | PASS | PASS | PASS | 5.75e-05 |
| SingleSiren | 0 | joint | L1x1e7 | True | all | PASS | PASS | PASS | 5.75e-05 |
| DualSiren | 0 | epsilon_only | L1x1e7 | False | low | PASS | PASS | PASS | 8.21e-05 |
| DualSiren | 0 | epsilon_only | L1x1e7 | False | high | PASS | PASS | PASS | 0.000118 |
| DualSiren | 0 | epsilon_only | L1x1e7 | False | alpha | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.000144 |
| DualSiren | 0 | epsilon_only | L1x1e7 | False | all | PASS | PASS | PASS | 9.15e-05 |
| DualSiren | 0 | sigma_only | L1x1e7 | False | low | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.00132 |
| DualSiren | 0 | sigma_only | L1x1e7 | False | high | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.00734 |
| DualSiren | 0 | sigma_only | L1x1e7 | False | alpha | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.00458 |
| DualSiren | 0 | sigma_only | L1x1e7 | False | all | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.0095 |
| DualSiren | 0 | joint | L1x1e7 | False | low | PASS | PASS | PASS | 0.000177 |
| DualSiren | 0 | joint | L1x1e7 | False | high | PASS | PASS | PASS | 0.0001 |
| DualSiren | 0 | joint | L1x1e7 | False | alpha | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.00433 |
| DualSiren | 0 | joint | L1x1e7 | False | all | PASS | PASS | PASS | 0.000136 |
| DualSiren | 0 | joint | L1x1e7 | True | low | PASS | PASS | PASS | 0.000177 |
| DualSiren | 0 | joint | L1x1e7 | True | high | PASS | PASS | PASS | 0.0001 |
| DualSiren | 0 | joint | L1x1e7 | True | alpha | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.00433 |
| DualSiren | 0 | joint | L1x1e7 | True | all | PASS | PASS | PASS | 0.000136 |
| Grid-FWI | 0 | joint | MSEx1e4 | False | epsilon | PASS | PASS | PASS | 2.13e-05 |
| Grid-FWI | 0 | joint | MSEx1e4 | False | sigma | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.000275 |
| Grid-FWI | 0 | joint | MSEx1e4 | False | joint | PASS | PASS | PASS | 2.34e-05 |
| SingleSiren | 0 | epsilon_only | MSEx1e4 | False | all | PASS | PASS | PASS | 6.15e-05 |
| SingleSiren | 0 | sigma_only | MSEx1e4 | False | all | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.00423 |
| SingleSiren | 0 | joint | MSEx1e4 | False | all | PASS | PASS | PASS | 2.67e-05 |
| SingleSiren | 0 | joint | MSEx1e4 | True | all | PASS | PASS | PASS | 2.67e-05 |
| DualSiren | 0 | epsilon_only | MSEx1e4 | False | low | PASS | PASS | PASS | 3.97e-05 |
| DualSiren | 0 | epsilon_only | MSEx1e4 | False | high | PASS | PASS | PASS | 0.000157 |
| DualSiren | 0 | epsilon_only | MSEx1e4 | False | alpha | PASS | PASS | PASS | 0.000268 |
| DualSiren | 0 | epsilon_only | MSEx1e4 | False | all | PASS | PASS | PASS | 5.77e-05 |
| DualSiren | 0 | sigma_only | MSEx1e4 | False | low | INCONCLUSIVE | PASS | INCONCLUSIVE | 4.73e-05 |
| DualSiren | 0 | sigma_only | MSEx1e4 | False | high | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.128 |
| DualSiren | 0 | sigma_only | MSEx1e4 | False | alpha | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.00921 |
| DualSiren | 0 | sigma_only | MSEx1e4 | False | all | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.166 |
| DualSiren | 0 | joint | MSEx1e4 | False | low | PASS | PASS | PASS | 0.000931 |
| DualSiren | 0 | joint | MSEx1e4 | False | high | PASS | PASS | PASS | 0.00056 |
| DualSiren | 0 | joint | MSEx1e4 | False | alpha | PASS | PASS | PASS | 0.000148 |
| DualSiren | 0 | joint | MSEx1e4 | False | all | PASS | PASS | PASS | 3.55e-05 |
| DualSiren | 0 | joint | MSEx1e4 | True | low | PASS | PASS | PASS | 0.000931 |
| DualSiren | 0 | joint | MSEx1e4 | True | high | PASS | PASS | PASS | 0.00056 |
| DualSiren | 0 | joint | MSEx1e4 | True | alpha | PASS | PASS | PASS | 0.000148 |
| DualSiren | 0 | joint | MSEx1e4 | True | all | PASS | PASS | PASS | 3.55e-05 |
| Grid-FWI | 6 | joint | L1x1e7 | False | epsilon | FAIL | PASS | FAIL | 0.183 |
| Grid-FWI | 6 | joint | L1x1e7 | False | sigma | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.166 |
| Grid-FWI | 6 | joint | L1x1e7 | False | joint | FAIL | PASS | FAIL | 0.111 |
| SingleSiren | 6 | epsilon_only | L1x1e7 | False | all | FAIL | PASS | FAIL | 0.0156 |
| SingleSiren | 6 | sigma_only | L1x1e7 | False | all | FAIL | INCONCLUSIVE | FAIL | 0.000947 |
| SingleSiren | 6 | joint | L1x1e7 | False | all | FAIL | PASS | FAIL | 0.00132 |
| SingleSiren | 6 | joint | L1x1e7 | True | all | FAIL | PASS | FAIL | 0.00132 |
| DualSiren | 6 | epsilon_only | L1x1e7 | False | low | FAIL | PASS | FAIL | 0.00907 |
| DualSiren | 6 | epsilon_only | L1x1e7 | False | high | PASS | PASS | PASS | 0.000881 |
| DualSiren | 6 | epsilon_only | L1x1e7 | False | alpha | FAIL | PASS | FAIL | 0.00741 |
| DualSiren | 6 | epsilon_only | L1x1e7 | False | all | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.00572 |
| DualSiren | 6 | sigma_only | L1x1e7 | False | low | FAIL | INCONCLUSIVE | FAIL | 0.0524 |
| DualSiren | 6 | sigma_only | L1x1e7 | False | high | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.0473 |
| DualSiren | 6 | sigma_only | L1x1e7 | False | alpha | FAIL | INCONCLUSIVE | FAIL | 0.0325 |
| DualSiren | 6 | sigma_only | L1x1e7 | False | all | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 0.00774 |
| DualSiren | 6 | joint | L1x1e7 | False | low | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.0133 |
| DualSiren | 6 | joint | L1x1e7 | False | high | FAIL | PASS | FAIL | 0.0413 |
| DualSiren | 6 | joint | L1x1e7 | False | alpha | FAIL | PASS | FAIL | 0.0171 |
| DualSiren | 6 | joint | L1x1e7 | False | all | FAIL | PASS | FAIL | 0.00903 |
| DualSiren | 6 | joint | L1x1e7 | True | low | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.0133 |
| DualSiren | 6 | joint | L1x1e7 | True | high | FAIL | PASS | FAIL | 0.0413 |
| DualSiren | 6 | joint | L1x1e7 | True | alpha | FAIL | PASS | FAIL | 0.0171 |
| DualSiren | 6 | joint | L1x1e7 | True | all | FAIL | PASS | FAIL | 0.00903 |
| Grid-FWI | 6 | joint | MSEx1e4 | False | epsilon | FAIL | PASS | FAIL | 0.0501 |
| Grid-FWI | 6 | joint | MSEx1e4 | False | sigma | FAIL | PASS | FAIL | 0.0972 |
| Grid-FWI | 6 | joint | MSEx1e4 | False | joint | FAIL | PASS | FAIL | 0.00514 |
| SingleSiren | 6 | epsilon_only | MSEx1e4 | False | all | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.0257 |
| SingleSiren | 6 | sigma_only | MSEx1e4 | False | all | FAIL | PASS | FAIL | 0.0279 |
| SingleSiren | 6 | joint | MSEx1e4 | False | all | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.0134 |
| SingleSiren | 6 | joint | MSEx1e4 | True | all | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.0134 |
| DualSiren | 6 | epsilon_only | MSEx1e4 | False | low | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.013 |
| DualSiren | 6 | epsilon_only | MSEx1e4 | False | high | FAIL | PASS | FAIL | 0.0141 |
| DualSiren | 6 | epsilon_only | MSEx1e4 | False | alpha | FAIL | PASS | FAIL | 0.0369 |
| DualSiren | 6 | epsilon_only | MSEx1e4 | False | all | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.00238 |
| DualSiren | 6 | sigma_only | MSEx1e4 | False | low | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.931 |
| DualSiren | 6 | sigma_only | MSEx1e4 | False | high | INCONCLUSIVE | PASS | INCONCLUSIVE | 0.00864 |
| DualSiren | 6 | sigma_only | MSEx1e4 | False | alpha | FAIL | INCONCLUSIVE | FAIL | 0.0329 |
| DualSiren | 6 | sigma_only | MSEx1e4 | False | all | FAIL | PASS | FAIL | 0.0805 |
| DualSiren | 6 | joint | MSEx1e4 | False | low | FAIL | PASS | FAIL | 0.0139 |
| DualSiren | 6 | joint | MSEx1e4 | False | high | FAIL | PASS | FAIL | 0.00243 |
| DualSiren | 6 | joint | MSEx1e4 | False | alpha | FAIL | PASS | FAIL | 0.0384 |
| DualSiren | 6 | joint | MSEx1e4 | False | all | FAIL | PASS | FAIL | 0.0266 |
| DualSiren | 6 | joint | MSEx1e4 | True | low | FAIL | PASS | FAIL | 0.0139 |
| DualSiren | 6 | joint | MSEx1e4 | True | high | FAIL | PASS | FAIL | 0.00243 |
| DualSiren | 6 | joint | MSEx1e4 | True | alpha | FAIL | PASS | FAIL | 0.0384 |
| DualSiren | 6 | joint | MSEx1e4 | True | all | FAIL | PASS | FAIL | 0.0266 |

## Boundary interpretation

PML=0 and PML=6 results are both shown for physical directions that include edge cells. Forward compute replicates edge materials into external CPML and builds material-dependent damping coefficients. A mismatch specifically in PML=6 is evidence about the full public total derivative; an interior-only or frozen-boundary FD would conceal that dependency. FAIL remains FAIL in these tables. Numerical-resolution and cancellation limitations are shown by the full multi-step curves and do not justify labelling a mismatch PASS.

## Reproduce

```sh
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python tests/gradient_verification/codex_audit_20261010/network_verification.py
```

Run output: `logs/network_verification.log`; complete environment, source/native-library hashes, acquisition, random seeds, measured J and gradient checks: `data/network_results.json`. CPU only, OMP_NUM_THREADS=1. No optimization/convergence or external IFWI-product correctness is claimed.
