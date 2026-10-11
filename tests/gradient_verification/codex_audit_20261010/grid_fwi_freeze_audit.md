# Active/frozen physical-grid FWI audit

This independent addendum closes the difference between an epsilon-only/sigma-only **perturbation direction** in a joint model and an epsilon-only/sigma-only **optimization mode**. The preceding 36 network cases remain unchanged. This script executes 12 actual grid cases: epsilon_only, sigma_only and joint × PML 0/6 × L1×1e7+TV/MSE×1e4+TV.

It imports only the sibling audit's public CPU propagation, actual example loss and strict FD helpers; importing that module does not execute its network audit. The smooth physical fields, synthetic observations, 20×20 grid, nt=250, source/receivers, stored float32 wavefield histories and float64 loss accumulation match the existing chain audit. Every inactive material is a fixed tensor with requires_grad=False, omitted from the AD/optimizer parameter list. The fields function accepts an active parameter vector and always returns the same frozen inactive tensor.

Each case performs a fresh FDTD → combined data+TV loss → one backward. Its actual combined gradient drives independent forward-only central FD, with all 18 original steps and no physical-cell mask. Separate data/TV gradients are retained and compared with the combined result. The unchanged strict criteria require two neighboring stable, resolved FD values for PASS; measured +/- native material ULP response and float64 TV arithmetic bounds distinguish FAIL from INCONCLUSIVE. No best-single-step PASS is allowed. Each single-property case has one direction; each joint case has epsilon/sigma/joint directions, giving 20 directions and 60 term-direction checks.

Each case executes torch.optim.SGD.step() on cloned **active** physical parameters. Frozen clones are requires_grad=False and excluded from the optimizer, and remain bitwise identical. Active changes are checked against theta-lr*actual_gradient within float64 rounding tolerance; the step caps maximum proposed update at 1e-10 times max(1, active parameter scale). All original material tensors are unchanged. Loss decrease is neither evaluated nor used as a PASS criterion.

Case count: 12. Term-direction counts: {'PASS': 28, 'FAIL': 14, 'INCONCLUSIVE': 18}. Execution summary: {'actual_combined_backward_cases': 12, 'combined_vs_separate_gradient_agreement_cases': 12, 'active_gradients_present_finite_cases': 12, 'fixed_inactive_tensors_verified_cases': 12, 'actual_clone_SGD_only_active_updated_cases': 12, 'original_physical_tensors_unchanged_cases': 12, 'single_property_cases_with_frozen_tensor': 8, 'directions': 20, 'term_direction_checks': 60, 'loss_descent_evaluated': False}.

| Mode | PML | Loss | Direction | Data | TV | Total | Total AD |
|---|---|---|---|---|---|---|---|
| epsilon_only | 0 | L1x1e7 | epsilon | PASS | PASS | PASS | -59205616.6 |
| sigma_only | 0 | L1x1e7 | sigma | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | -720463.446 |
| joint | 0 | L1x1e7 | epsilon | PASS | PASS | PASS | -59201633.5 |
| joint | 0 | L1x1e7 | sigma | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | -2008424.17 |
| joint | 0 | L1x1e7 | joint | INCONCLUSIVE | PASS | INCONCLUSIVE | 4363151.93 |
| epsilon_only | 0 | MSEx1e4 | epsilon | PASS | PASS | PASS | -1392551.46 |
| sigma_only | 0 | MSEx1e4 | sigma | PASS | PASS | PASS | 23962.5924 |
| joint | 0 | MSEx1e4 | epsilon | INCONCLUSIVE | PASS | INCONCLUSIVE | -80129.8178 |
| joint | 0 | MSEx1e4 | sigma | PASS | PASS | PASS | -25248.111 |
| joint | 0 | MSEx1e4 | joint | PASS | PASS | PASS | 911300.698 |
| epsilon_only | 6 | L1x1e7 | epsilon | FAIL | PASS | FAIL | -6623023.12 |
| sigma_only | 6 | L1x1e7 | sigma | FAIL | INCONCLUSIVE | FAIL | 320661.834 |
| joint | 6 | L1x1e7 | epsilon | FAIL | PASS | FAIL | 4160522.4 |
| joint | 6 | L1x1e7 | sigma | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | 81073.5305 |
| joint | 6 | L1x1e7 | joint | FAIL | PASS | FAIL | -482428.299 |
| epsilon_only | 6 | MSEx1e4 | epsilon | INCONCLUSIVE | PASS | INCONCLUSIVE | 171012.855 |
| sigma_only | 6 | MSEx1e4 | sigma | FAIL | PASS | FAIL | -4022.03291 |
| joint | 6 | MSEx1e4 | epsilon | FAIL | PASS | FAIL | -347704.663 |
| joint | 6 | MSEx1e4 | sigma | INCONCLUSIVE | PASS | INCONCLUSIVE | 245.333077 |
| joint | 6 | MSEx1e4 | joint | FAIL | PASS | FAIL | -249047.992 |

PML failures and numerically unresolved conductivity directions remain labelled FAIL/INCONCLUSIVE. A frozen tensor working correctly does not prove the full PML propagation derivative correct. Full per-parameter records, all FD curves, empirical resolution probes, random seeds and source/library hashes are in data/grid_fwi_freeze.json.

## Reproduce

```sh
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python tests/gradient_verification/codex_audit_20261010/grid_fwi_freeze_verification.py
```

CPU only, OMP_NUM_THREADS=1. Run log: logs/grid_fwi_freeze.log. No external experiment/data import, production edits or GPU execution. No optimization convergence is claimed.
