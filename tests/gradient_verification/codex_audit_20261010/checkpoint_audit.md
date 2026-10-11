# Checkpoint and saved-wavefield audit, 2026-10-10

Checkpoint equivalence is confirmed on this synthetic CPU fixture. The full
public `compute()` material derivative with regenerated external PML is **not
confirmed**: three of four unmasked PML finite-difference directions fail.
Those failures remain in the result file. The documented fixed-PML partial
derivative explains the distinction; it does not turn the public derivative
into a passing result.

## Reproducible CPU execution

Run from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python tests/gradient_verification/codex_audit_20261010/checkpoint_verification.py > tests/gradient_verification/codex_audit_20261010/logs/checkpoint_cpu.log 2>&1
```

The independent output is `data/checkpoint_results.json`; it does not overwrite
the user-created t05 script, results, or logs. Environment: Python 3.10.16,
PyTorch 1.11.0+cu115, NumPy 1.26.4, one OpenMP/PyTorch thread, CPU only. Code
commit at execution: `3bfc1e998851dbe489cf3a56b3a7cb27d8f476ba` on
`test/gradient-verification`. Seed: `2026101019`. CPU library SHA256:
`a901019842ee3daefa05a4412d5e965c1141d3cd404585f4f1d65de216dd4a31`.

The fresh model is 3D `10 x 11 x 8`, order 2, 151 time steps, 2 synthetic shots,
`dx=0.02`, `dt=2e-11`, CPML thickness 3. Every one of the 30 initial E/H/CPML
tensors has nonzero seeded random values. Synthetic observations are freshly
computed from perturbed material parameters using the same initial state.
The unchanged objective is a float64 receiver misfit plus a fixed random linear
terminal functional on **all 30** state tensors. This also exercises nonzero
terminal CPML cotangents; the material inputs still use native float32.

## State mutation and checkpoint reconstruction

Production allocation preserves the identity of contiguous, matching supplied
E/H tensors (`solver/fields.py:39-53`); CPML phi tensors are likewise reused
(`solver/pml.py:363-390`). The native forward receives those memory addresses,
updates all 30 states in place, and returns them (`solver/autograd.py:570-589`
and `:619`). The checkpoint closure therefore must clone every boundary state
before calling `compute()`. The cloning must retain the autograd connection;
detaching a boundary would truncate the adjoint between segments.

The existing example and t05 closure correctly clone and return all 30 states.
The native bridge does not call `ctx.mark_dirty`, so callers should not rely on
PyTorch's version counter to catch unsafe native mutations automatically.
Material-gradient histories are released after backward; a second backward on
the same solver invocation is explicitly rejected (`autograd.py:641-649`,
`:725-728`). A new forward is required, even with `retain_graph=True`.

Fresh comparisons against an unsegmented stored-FP32 `S=1` solve:

| K | Segment lengths | Modes tested | Receiver / all 30 final states |
|---|---|---|---|
| 1 | 151 | plain, reentrant, nonreentrant | bitwise equal |
| 2 | 76, 75 | plain, reentrant, nonreentrant | bitwise equal |
| 3 | 51, 50, 50 | plain, reentrant, nonreentrant | bitwise equal |
| 5 | 31, 30, 30, 30, 30 | plain, reentrant, nonreentrant | bitwise equal |

Across all 12 cases the maximum relative L2 material-gradient differences are
`1.7059066205e-6` for epsilon and `3.8759356824e-7` for conductivity, below the
predeclared `1e-5` equivalence tolerance. Source and all 30 initial-state
gradients are identical. Initial boundary tensors remain unchanged after both
forward and backward; backward does not alter final states. The call record
shows K forward invocations for plain and 2K for both checkpoint modes, so
recomputation actually occurred rather than merely wrapping an ordinary graph.

A deliberately unsafe K=3 reentrant negative control omits the clones. It
overwrites the initial state and produces epsilon-gradient relative L2 error
`13.1456717388`. This is recorded as evidence for the required cloning contract.

## Independent state-adjoint identity

For a linear state map with fixed materials/CPML and source identically zero,
the LHS is obtained from a **forward-only** call and includes every terminal
E/H/CPML tensor plus receiver data. The RHS uses the native state adjoint to
compute the initial-state cotangent. All 30 input states are nonzero and all 30
output cotangents are supplied. No boundary or CPML array is omitted.

| nt | `<A x, y>` | `<x, A^T y>` | Relative error | Threshold |
|---|---|---|---|---|
| 1 | -0.0515411623819 | -0.0515411815916 | 3.7270505000e-7 | 2e-5 |
| 5 | -0.330030544386 | -0.330030517935 | 8.0147387010e-8 | 2e-5 |

Both pass; this identity tests the state transpose independently of a
material-gradient reference computed by the same backward implementation.

## Forward-only material finite differences: retained failures

Each centered difference calls only forward with history disabled and no
native backward. Fresh h values are `0.1, 0.03, 0.01, 0.003, 0.001, 0.0003`,
scaled by 4 for epsilon or 0.01 for conductivity. Directions are deterministic
random values over either every physical model cell or every outer physical
face. No `edge_mask`, data window, or sensitivity-selected direction is used.
The predeclared criterion is best relative error below 0.5%, with at least two
steps below 1%; all h values and objective values are preserved in JSON.

| External PML | Parameter / direction | Best relative error | Verdict |
|---|---|---|---|
| thickness 3 | epsilon, all physical cells | 11.9303% | FAIL |
| thickness 3 | epsilon, outer physical faces | 25.7330% | FAIL |
| thickness 3 | conductivity, all physical cells | 0.2211% | PASS |
| thickness 3 | conductivity, outer physical faces | 50.1189% | FAIL |

The epsilon all-cell discrepancy has a stable plateau around 12% for
h=0.03..0.003; it is not explained by an excessively small FD step alone.
`_ExtendModel.backward` crops rather than transposing replicated padding
(`preprocessing/model_setup.py:72-90`), and the CPML material averages are
detached (`solver/pml.py:245-246`). Every public forward nevertheless rebuilds
both from the perturbed model. `docs/API.md:131` explicitly documents the
fixed-boundary partial-gradient convention. These results do not validate the
complete derivative of the full public forward when model edges vary.

An additional **diagnostic**, keeping the same kind of objective/directions
and changing only PML thickness to zero, gives 0.0757%, 0.1086%, and 0.0255%
for epsilon all-cell, epsilon outer-face, and conductivity all-cell checks,
which pass. Conductivity outer-face has a small derivative (`-2.55566e-4`);
its best error is 0.9817%, with strong rounding noise at smaller h, and it
remains FAIL under the unchanged criterion. This diagnostic neither replaces
nor overrides any of the thickness-3 failures.

## Actual CPU storage support and independent precision measurements

All storage configurations preserve receiver data and all 30 terminal states
bitwise on this fixture. Gradient approximations are **measured**, not assigned
a universal PASS based on a permissive historical benchmark.

| Configuration | Epsilon rel L2 | Conductivity rel L2 | Interpretation |
|---|---|---|---|
| FP32 auto / reconstructed | 0 | 0 | equals stored FP32 |
| FP32 physical stored / reconstructed | 0 | 0 | equals extended stored FP32 |
| FP16 stored | 3.00068e-4 | 3.17435e-5 | measured approximation |
| BF16 stored | 4.66549e-3 | 7.62158e-5 | measured approximation |
| FP16 reconstructed | 1.37399e-3 | 3.22270e-5 | measured approximation |
| BF16 reconstructed | 9.23682e-3 | 8.00197e-5 | measured approximation |
| S=2 | 8.45088e-4 | 7.47985e-4 | measured approximation, above bound 1 |
| S=3 | 2.17465e-3 | 2.00137e-3 | measured approximation, above bound 1 |
| async requested on CPU | 0 | 0 | accepted but effectively a no-op |

CPU FP16/BF16 use the legacy conversion path and are actually exercised here.
The loaded CPU native library reports external-PML, RHS-reconstruction, and
physical-history capability 1. It has no INT8 capability symbol; CPU INT8 is
correctly rejected with `wavefield_compression='int8' is CUDA-only and
GPU-resident.` Native scalar/vector conversion choices are also CUDA only.
The effective async flag is `requested and device.type == 'cuda'`
(`autograd.py:526`), so this CPU result is not evidence of asynchronous offload.

CUDA noncompressed FP32/FP16/BF16 histories can reside on GPU or be offloaded
to pinned host tensors (`autograd.py:440-451`). INT8 requires CUDA, the
`wavefield_storage_dtype` argument must remain FP32, and INT8 plus async is
rejected (`modeling.py:113-125`). CUDA `native_scalar` / `native_vec2` conversion
backends are valid only with noncompressed FP16/BF16. BF16 auto resolves to
legacy, while FP16 auto resolves to native_scalar (`modeling.py:89-94`).
The optional ZFP mode is explicitly unimplemented, not a tested compression
backend (`modeling.py:95-101`). The CPU cases alone make no GPU claim; fresh
host-GPU evidence follows below.

## Why the existing t05 PASS labels are insufficient

The existing t05 files were read for audit only, not reused as new verification
data. t05's directional helper masks boundary-adjacent cells, so its material
FD cannot resolve the public-forward edge issue above. Its approximate branch
accepts **10 times** its reported `documented` number and, for E-only, first
inflates that number by another factor of **3.5** (`t05:188-194`). It checks
only one source/model/geometry family and has no universal error guarantee.

For example, existing t05 3D CUDA-general INT8 stored epsilon rel L2 is
`0.0530895411705` (5.31%) yet is labelled PASS, because its threshold is 0.32
(32%). Existing BF16 stored epsilon error is 0.9556% yet also gets the same
PASS wording. These numbers are evidence of conditional approximations and
permissive labels, not evidence of exact gradients or broadly guaranteed
accuracy. GPU INT8, BF16, sampling and reconstructed low-precision options
should retain quantitative errors and task-specific tolerances rather than
inherit those labels.

## Rerun after the native-library update

The old CPU result and logs are preserved. The independent rerun used the
same synthetic fixture, objective, directions, and h values:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python tests/gradient_verification/codex_audit_20261010/checkpoint_verification.py --tag after_native_fix > tests/gradient_verification/codex_audit_20261010/logs/checkpoint_cpu_after_native_fix.log 2>&1
```

Output: `data/checkpoint_results_after_native_fix.json`. New CPU library SHA256:
`2925ec1d49c36f4914f68442209e743d46ee34f764d3b4f5a7f986c0e3a39161`.
The 12 checkpoint cases, two independent state-dot cases, and exact-storage
comparisons retain their passing numerical results. The public-forward PML
finite differences retain **three FAILs**, and the additional no-PML small
conductivity outer-face derivative retains its separate FAIL. Both original
and new result sets are available; no old result was substituted for this
rerun. Native changes in another audit workstream therefore did not resolve
the full regenerated-PML material derivative.

## Fresh GPU storage and checkpoint execution

Run after the root agent atomically installed and verified the updated native
libraries, with exclusive GPU access for this script:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python tests/gradient_verification/codex_audit_20261010/gpu_storage_verification.py > tests/gradient_verification/codex_audit_20261010/logs/gpu_storage.log 2>&1
```

The command needs host GPU visibility; the default sandbox lacked the CUDA
driver. Its immediate unavailable-driver failure is preserved separately in
`logs/gpu_storage_unavailable.log`. The authorized host execution used an
RTX 4090, compute capability 8.9, driver 575.51.03, PyTorch 1.11.0+cu115,
one OpenMP/PyTorch thread, and seed `2026101021`. CUDA library SHA256:
`d2e458bd438ed0a050ad7e74b871e5c8ad5ce6633e65954020e28dc998401861`.
All queried native capabilities (external PML, reconstruction, physical
history, conversion backends, INT8, and INT8 reduction) report 1.

Fresh synthetic fixtures, with no external experimental data:

| Path | Shape | Time steps | PML | Order | dt |
|---|---|---|---|---|---|
| CUDA 2D TM fast | 40 x 40 | 400 | 6 | 2 | 2.5e-11 |
| CUDA 3D general | 14 x 14 x 10 | 200 | 3 | 2 | 2e-11 |

For each geometry, 35 storage configurations and four real checkpoint cases
were executed: **78 total**. All preserve receivers and all 30 final states
bitwise. Twenty-two exact-storage/checkpoint comparisons pass `1e-5` relative
gradient equivalence; 56 approximate comparisons are labelled
`MEASURED_APPROXIMATION`. There are **zero execution/storage failures**.
INT8+FP16, INT8+async, and ZFP validation are tested separately for each
geometry: all six are rejected with the expected explicit exception.

Representative relative L2 gradient errors against stored FP32 S=1:

| Geometry / configuration | Epsilon | Conductivity |
|---|---|---|
| 2D FP16 legacy stored | 2.84688e-5 | 3.38468e-5 |
| 2D BF16 legacy stored | 2.42934e-4 | 2.76117e-4 |
| 2D FP16 reconstructed | 3.42109e-5 | 3.41801e-5 |
| 2D BF16 reconstructed | 2.69631e-4 | 2.80405e-4 |
| 2D INT8 stored | 6.37767e-4 | 7.92878e-4 |
| 2D INT8 reconstructed | 6.53909e-4 | 7.93245e-4 |
| 2D S=3 | 2.44800e-6 | 4.19143e-6 |
| 2D S=5 | 3.59146e-6 | 4.94126e-6 |
| 3D FP16 legacy stored | 5.81266e-5 | 2.15475e-5 |
| 3D BF16 legacy stored | 7.91145e-4 | 1.96941e-4 |
| 3D FP16 reconstructed | 1.68584e-4 | 2.14012e-5 |
| 3D BF16 reconstructed | 7.86042e-4 | 1.98228e-4 |
| 3D INT8 stored | 1.96134e-2 | 7.47269e-3 |
| 3D INT8 reconstructed | 1.96054e-2 | 7.38268e-3 |
| 3D S=3 | 1.11444e-3 | 9.01298e-5 |
| 3D S=5 | 1.23512e-3 | 1.51384e-4 |

Legacy, native_scalar and native_vec2 are **actually executed** for both
FP16/BF16, with stored, explicit reconstructed, and auto RHS options. Full
epsilon/conductivity relative L2, cosine, norm, maximum absolute difference,
and outer-face error metrics for every configuration are saved in
`data/gpu_storage.json`; the table is a subset, not a substitute for that data.
The 3D INT8 epsilon error is about 1.96%, despite cosine about 0.99981. High
cosine alone must not be interpreted as small gradient error.

The actual autograd context is inspected before backward. FP32 auto uses
E-only reconstruction. FP16/BF16/INT8 auto use stored E+R; only explicit
`reconstructed` requests E-only for those lossy formats. Async FP32/FP16/BF16
rows have CPU-pinned E/R histories and `async_effective=True`; the E-only
final-E frame stays on GPU as designed. This verifies the effective backend,
not merely the requested configuration string.

GPU K=2 and K=3, both reentrant and nonreentrant, each execute 2K segment
calls. Their data and full 30-state outputs are bitwise equal to the full
solve; maximum material-gradient relative L2 is `2.42826e-7` epsilon and
`2.36506e-7` conductivity in 2D, and `5.95859e-7` / `3.77346e-7` in 3D.
Initial boundary inputs are checked to remain unchanged. These CUDA tests
use the ordinary zero initial state; the CPU independent dot/checkpoint test
above covers the all-30-nonzero case separately.

## GPU forward-only material finite differences retain full-edge failures

The objective is the unchanged float64 synthetic receiver misfit. Each
geometry receives a fresh forward-only sweep at h=`0.3, 0.1, 0.03, 0.01,
0.003, 0.001, 0.0003`; no native backward is used for any FD value. A compact
interior-support direction and **all outer physical faces** are reported side
by side for each parameter. Compact support is not a replacement for the
full-edge check. The same-run FD curves are reused only for configurations
whose receivers are bitwise identical; their corresponding gradients are
individually projected onto those directions. No saved t05 FD is reused.

Stored-FP32 baselines give:

| Geometry / parameter | Interior compact best error | All outer faces best error |
|---|---|---|
| 2D epsilon | 0.0368%, PASS | 55.7481%, FAIL |
| 2D conductivity | 0.00826%, PASS | 36.9718%, FAIL |
| 3D epsilon | 0.0440%, PASS | 47.8886%, FAIL |
| 3D conductivity | 0.0582%, PASS | 22.3600%, FAIL |

All h values, actual FD/AD directional derivatives, loss values, directions,
and per-configuration verdicts remain in JSON. The four baseline outer-face
FAILs are the complete-public-forward derivative status, even though exact
checkpoint/storage equivalence passes. The script deliberately exits 1 for
these unresolved derivative failures; this is distinct from zero execution
and zero storage-equivalence failures.

## Timing and memory scope

Forward/backward durations and `reset_peak_memory_stats()` allocated/reserved
peaks, plus baseline allocator usage, are recorded per case. These are
single-run measurements including model allocation and synchronization, not
claims of repeatable throughput or process-wide GPU memory. PyTorch peaks
exclude native `cudaMalloc`, the CUDA context, and driver allocations.

For illustration, 3D stored FP32 reports 61.96 MiB peak allocated, FP32
auto E-only 32.11 MiB, and async FP32 stored 1.96 MiB. Their measured total
times are 11.63 ms, 11.26 ms and 31.91 ms respectively. Reserved memory is
104 MiB throughout this 3D sequence because the allocator cache is retained;
it is explicitly recorded and must not be confused with reduced process
reservation or a total device-memory measurement.

## Resolution review of the weak no-PML conductivity face check

This is a derived review of the **new-library** result. Neither original CPU
JSON is edited, the original FD direction and objective are unchanged, and
the raw relative-test verdict remains **FAIL**. The derived review is
**INCONCLUSIVE_FLOAT32_RESOLUTION**, not PASS. The new-library gAD is
`-2.5549844297540814e-4`; its original sweep's best relative error is
`1.0079%` against the unchanged criterion, best <0.5% and at least two
steps <1%. The earlier 0.9817% value above belongs to the preserved old-library
result.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python tests/gradient_verification/codex_audit_20261010/checkpoint_fd_resolution_review.py > tests/gradient_verification/codex_audit_20261010/logs/checkpoint_fd_resolution_review.log 2>&1
```

The review reconstructs the same seeded nonzero 30-component initial state,
synthetic observation, fixed terminal weights, and sigma direction. It
preserves the float64 receiver misfit plus all-state terminal objective.
Fresh gAD matches the current new-library JSON exactly. At each of the four
h values below, three independent forward-only evaluations reproduce the
original J-minus and J-plus bitwise. Their repeated-input loss range is zero;
there is no measured run-to-run randomness.

Material-dependent float32 resolution is a separate issue. At each actual
FD endpoint, both signs of exactly one sigma ULP are tested with all-cell,
random-sign, and original-face-sign patterns. The observed residual is the
objective jump after subtracting the tiny linear material effect. The sum
of the largest measured endpoint residuals gives an **empirical resolution
scale**, not a certified roundoff bound or a new acceptance tolerance.

| h | Absolute derivative discrepancy | Objective numerator discrepancy | Sum of endpoint one-ULP residual maxima | Empirical derivative resolution |
|---|---|---|---|---|
| 0.1 | 3.59906e-6 | 7.19812e-7 | 1.08875e-6 | 5.44373e-6 |
| 0.03 | 2.60140e-6 | 1.56084e-7 | 7.62006e-7 | 1.27001e-5 |
| 0.01 | 2.00467e-5 | 4.00934e-7 | 7.37499e-7 | 3.68750e-5 |
| 0.003 | 5.83031e-5 | 3.49819e-7 | 7.63817e-7 | 1.27303e-4 |

Every displayed numerator discrepancy is smaller than its measured ULP
resolution scale. Baseline plus/minus one-ULP eps/sigma probes and additional
2..128-ULP scans are also saved separately, without using larger perturbations
to enlarge the primary error budget. Sigma ULPs are about 9.31e-10..1.86e-9;
an update coefficient can remain unchanged until a quantization boundary is
crossed. The pure face-sign one-ULP AD material effect is only 6.20e-11 in J,
yet its endpoint J changes can reach 3.96e-7.

The fixed-direction neighboring h curves do not exhibit a stable error
plateau: at h=0.024 and 0.036 the signed derivative errors are -2.18e-5 and
+1.28e-5; at 0.11 and 0.12 they are +4.61e-7 and -5.71e-7. Favorable
neighboring h values are diagnostic only and never replace the original
prespecified sweep or upgrade its FAIL.

These measurements support inadequate float32 forward resolution for this
weak derivative at the requested relative tolerance. They do **not** prove
that the production gradient is correct or that all discrepancy is rounding.
Higher forward precision or an independent analytical oracle is required to
settle it. All existing PML material derivative FAILs remain unresolved and
are unaffected by this additional no-PML review. The complete probes,
repetition values, neighboring curves, preserved raw criterion/verdict,
environment, script hash, and source-result hash are saved in
`data/checkpoint_fd_resolution_review.json`.
