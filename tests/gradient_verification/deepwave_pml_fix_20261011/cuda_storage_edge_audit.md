# CUDA storage and PML-edge material-gradient audit

Synthetic 20×20 2D Ez-TM, two shots, nt=220, PML=6, order=4, dx=.02, dt=3e-11. The PML reference speed is explicitly fixed at 299792458 m/s for AD and every FD. No external data/modules or old evidence were modified. This is a correctness audit; concurrent GPU activity makes elapsed time unsuitable for performance conclusions.

General and TM2D paths use the real public compute/backward with a test-only route predicate override. Six physical directions include every boundary face, corners, and all cells. FD uses fresh save_wavefield_history=False forwards at 10 steps; storage cannot change receiver forwards, so each path shares its forward-only curves across formats. Eight +/- one-native-ULP probes per direction and three repeats measure resolution. Strict PASS requires adjacent FD stability <=.005, signal >10× resolution/h, and both AD relative errors <=.002. Stable resolved bias > .005 plus 3× resolution/h is FAIL unless finer resolved counterevidence exists; otherwise INCONCLUSIVE. Lossy history discrepancies are retained, never relabelled exact PASS.

FP32 and FP16/BF16 legacy/native_scalar/native_vec2 each cover stored/reconstructed RHS and resident/synchronous history or asynchronous pinned-CPU offload. INT8 covers stored/reconstructed GPU-resident history; INT8 CPU offload is unsupported and checked for explicit rejection. All material histories cover the extended grid.

Cases: 60. Direction checks: {'PASS': 298, 'FAIL': 0, 'INCONCLUSIVE': 62}. Unsupported contracts: [{'path': 'tm2d', 'rhs': 'stored', 'int8_async_cpu': 'REJECTED_AS_UNSUPPORTED', 'error': 'use_async_offload=True is incompatible with GPU-resident INT8 wavefield compression.'}, {'path': 'tm2d', 'rhs': 'reconstructed', 'int8_async_cpu': 'REJECTED_AS_UNSUPPORTED', 'error': 'use_async_offload=True is incompatible with GPU-resident INT8 wavefield compression.'}, {'path': 'general', 'rhs': 'stored', 'int8_async_cpu': 'REJECTED_AS_UNSUPPORTED', 'error': 'use_async_offload=True is incompatible with GPU-resident INT8 wavefield compression.'}, {'path': 'general', 'rhs': 'reconstructed', 'int8_async_cpu': 'REJECTED_AS_UNSUPPORTED', 'error': 'use_async_offload=True is incompatible with GPU-resident INT8 wavefield compression.'}].

| Path | Storage | PASS | FAIL | INCONCLUSIVE | ε gradient relative to FP32 | σ gradient relative to FP32 |
|---|---|---|---|---|---|---|
| tm2d | fp32/stored/resident | 5 | 0 | 1 | 0 | 0 |
| tm2d | fp32/stored/async_cpu | 5 | 0 | 1 | 0 | 0 |
| tm2d | fp32/reconstructed/resident | 5 | 0 | 1 | 0 | 0 |
| tm2d | fp32/reconstructed/async_cpu | 5 | 0 | 1 | 0 | 0 |
| tm2d | fp16_legacy/stored/resident | 5 | 0 | 1 | 5.26241e-05 | 0.000152399 |
| tm2d | fp16_legacy/stored/async_cpu | 5 | 0 | 1 | 5.26241e-05 | 0.000152399 |
| tm2d | fp16_legacy/reconstructed/resident | 5 | 0 | 1 | 9.27818e-05 | 0.000145361 |
| tm2d | fp16_legacy/reconstructed/async_cpu | 5 | 0 | 1 | 9.27818e-05 | 0.000145361 |
| tm2d | fp16_native_scalar/stored/resident | 5 | 0 | 1 | 5.26241e-05 | 0.000152399 |
| tm2d | fp16_native_scalar/stored/async_cpu | 5 | 0 | 1 | 5.26241e-05 | 0.000152399 |
| tm2d | fp16_native_scalar/reconstructed/resident | 5 | 0 | 1 | 9.27818e-05 | 0.000145361 |
| tm2d | fp16_native_scalar/reconstructed/async_cpu | 5 | 0 | 1 | 9.27818e-05 | 0.000145361 |
| tm2d | fp16_native_vec2/stored/resident | 5 | 0 | 1 | 5.26241e-05 | 0.000152399 |
| tm2d | fp16_native_vec2/stored/async_cpu | 5 | 0 | 1 | 5.26241e-05 | 0.000152399 |
| tm2d | fp16_native_vec2/reconstructed/resident | 5 | 0 | 1 | 9.27818e-05 | 0.000145361 |
| tm2d | fp16_native_vec2/reconstructed/async_cpu | 5 | 0 | 1 | 9.27818e-05 | 0.000145361 |
| tm2d | bf16_legacy/stored/resident | 5 | 0 | 1 | 0.000269422 | 0.0013756 |
| tm2d | bf16_legacy/stored/async_cpu | 5 | 0 | 1 | 0.000269422 | 0.0013756 |
| tm2d | bf16_legacy/reconstructed/resident | 5 | 0 | 1 | 0.000670991 | 0.0012961 |
| tm2d | bf16_legacy/reconstructed/async_cpu | 5 | 0 | 1 | 0.000670991 | 0.0012961 |
| tm2d | bf16_native_scalar/stored/resident | 5 | 0 | 1 | 0.000269422 | 0.0013756 |
| tm2d | bf16_native_scalar/stored/async_cpu | 5 | 0 | 1 | 0.000269422 | 0.0013756 |
| tm2d | bf16_native_scalar/reconstructed/resident | 5 | 0 | 1 | 0.000670991 | 0.0012961 |
| tm2d | bf16_native_scalar/reconstructed/async_cpu | 5 | 0 | 1 | 0.000670991 | 0.0012961 |
| tm2d | bf16_native_vec2/stored/resident | 5 | 0 | 1 | 0.000269422 | 0.0013756 |
| tm2d | bf16_native_vec2/stored/async_cpu | 5 | 0 | 1 | 0.000269422 | 0.0013756 |
| tm2d | bf16_native_vec2/reconstructed/resident | 5 | 0 | 1 | 0.000670991 | 0.0012961 |
| tm2d | bf16_native_vec2/reconstructed/async_cpu | 5 | 0 | 1 | 0.000670991 | 0.0012961 |
| tm2d | int8/stored/resident | 5 | 0 | 1 | 0.000610717 | 0.00366832 |
| tm2d | int8/reconstructed/resident | 4 | 0 | 2 | 0.00220011 | 0.0033332 |
| general | fp32/stored/resident | 5 | 0 | 1 | 0 | 0 |
| general | fp32/stored/async_cpu | 5 | 0 | 1 | 1.446e-07 | 3.48628e-07 |
| general | fp32/reconstructed/resident | 5 | 0 | 1 | 1.36652e-07 | 4.25541e-07 |
| general | fp32/reconstructed/async_cpu | 5 | 0 | 1 | 1.52827e-07 | 3.48014e-07 |
| general | fp16_legacy/stored/resident | 5 | 0 | 1 | 5.26069e-05 | 0.00015228 |
| general | fp16_legacy/stored/async_cpu | 5 | 0 | 1 | 5.26118e-05 | 0.000152322 |
| general | fp16_legacy/reconstructed/resident | 5 | 0 | 1 | 9.27944e-05 | 0.000145328 |
| general | fp16_legacy/reconstructed/async_cpu | 5 | 0 | 1 | 9.2779e-05 | 0.000145308 |
| general | fp16_native_scalar/stored/resident | 5 | 0 | 1 | 5.26514e-05 | 0.000152351 |
| general | fp16_native_scalar/stored/async_cpu | 5 | 0 | 1 | 5.26631e-05 | 0.000152311 |
| general | fp16_native_scalar/reconstructed/resident | 5 | 0 | 1 | 9.27967e-05 | 0.000145254 |
| general | fp16_native_scalar/reconstructed/async_cpu | 5 | 0 | 1 | 9.2797e-05 | 0.000145248 |
| general | fp16_native_vec2/stored/resident | 5 | 0 | 1 | 5.26696e-05 | 0.00015233 |
| general | fp16_native_vec2/stored/async_cpu | 5 | 0 | 1 | 5.2636e-05 | 0.000152311 |
| general | fp16_native_vec2/reconstructed/resident | 5 | 0 | 1 | 9.2808e-05 | 0.000145217 |
| general | fp16_native_vec2/reconstructed/async_cpu | 5 | 0 | 1 | 9.28144e-05 | 0.000145246 |
| general | bf16_legacy/stored/resident | 5 | 0 | 1 | 0.000269347 | 0.00137562 |
| general | bf16_legacy/stored/async_cpu | 5 | 0 | 1 | 0.00026943 | 0.00137558 |
| general | bf16_legacy/reconstructed/resident | 5 | 0 | 1 | 0.000670958 | 0.00129611 |
| general | bf16_legacy/reconstructed/async_cpu | 5 | 0 | 1 | 0.000670941 | 0.00129615 |
| general | bf16_native_scalar/stored/resident | 5 | 0 | 1 | 0.000269376 | 0.00137557 |
| general | bf16_native_scalar/stored/async_cpu | 5 | 0 | 1 | 0.00026939 | 0.00137557 |
| general | bf16_native_scalar/reconstructed/resident | 5 | 0 | 1 | 0.00067093 | 0.00129615 |
| general | bf16_native_scalar/reconstructed/async_cpu | 5 | 0 | 1 | 0.000670978 | 0.00129614 |
| general | bf16_native_vec2/stored/resident | 5 | 0 | 1 | 0.00026939 | 0.00137568 |
| general | bf16_native_vec2/stored/async_cpu | 5 | 0 | 1 | 0.000269381 | 0.00137561 |
| general | bf16_native_vec2/reconstructed/resident | 5 | 0 | 1 | 0.000670952 | 0.00129607 |
| general | bf16_native_vec2/reconstructed/async_cpu | 5 | 0 | 1 | 0.000670943 | 0.00129611 |
| general | int8/stored/resident | 5 | 0 | 1 | 0.000610683 | 0.00366832 |
| general | int8/reconstructed/resident | 4 | 0 | 2 | 0.0022001 | 0.00333321 |

Complete shared raw J+/J− curves, resolution probes, case-specific AD/errors, history devices/bytes, corner gradients, hashes and environment are in data/cuda_storage_edge.json.

```sh
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/ll/anaconda3/envs/deepgpr/bin/python -u /home/ll/下载/DeepGPR-dev/tests/gradient_verification/deepwave_pml_fix_20261011/cuda_storage_edge_audit.py
```

Run log: logs/cuda_storage_edge.log.
