# CUDA async-offload resource reuse (audit N-8)

## Change and scope

Baseline: upstream `dd6a67ba0fd33808c1f939b79054f9fb1b9f23c7`.
Only the native async-offload resource lifetime changes: streams, events and
E/R staging buffers are leased to individual calls and reused after successful
synchronization. Events disable timing because the solver only uses them for
dependencies. Kernel arithmetic, history layout and the ABI 6 solver signatures
are unchanged. Exact-E and CPML scratch remain transient.

The cache holds one idle workspace per device, up to 64 MiB total E/R capacity.
Active calls never share a workspace. Failed or oversized workspaces are not
retained; partially constructed workspaces clean up through RAII. Cache locks
cover ownership transfers, not CUDA work or synchronization.
`DeepGPR.clear_cuda_resource_cache(device=None)` releases idle resources. Active
calls may refill the cache afterwards. Native cached bytes are outside
PyTorch's allocator accounting and `torch.cuda.empty_cache()`.

## Validation

Host: WSL2 Debian 13, Intel i7-14700KF, NVIDIA RTX 4070 (12 GiB), driver
617.14. Python 3.12.14, PyTorch 2.7.1+cu126, CUDA toolkit 12.9.86.
Baseline and candidate were rebuilt with identical `-std=c++17 -O3 -arch=sm_89
-shared -Xcompiler=-fPIC` flags and static cudart. The local GCC 14.2 compiler
used a private sysroot with glibc 2.36 headers to avoid CUDA 12.9's conflict
with Debian 13 math declarations. Packaged binaries were not replaced.

- Complete suite with CUDA enabled: **182 tests, no failures/errors,
  `OK (skipped=4)`**. Two skip entries require a second GPU; two existing CPU
  finite-difference subtests cannot resolve their weak joint direction at the
  specified tolerance. Skip entries include subtests, so subtraction does not
  give a passed-test count.
- CPU-only environment (PyTorch 2.14.1+cpu): 182 tests, no failures/errors,
  `OK (skipped=49)`.
- Six A/B configurations, each repeated twice with fresh processes per
  library: 2D/3D, varying shapes, fp32/fp16/bf16, sampled histories, stored-R
  and reconstructed-R histories, and source-only gradients. Receivers, fields,
  PML states, histories and 2D gradients are bitwise identical. General 3D
  gradients retain the existing atomic-order variation: relative L2 error
  must be at most `1e-6`, with elementwise `rtol=2e-5` and an absolute bound
  scaled by `1e-6 * max(abs(reference))`.
- `test_cuda_resource_cache.py` also compares async results with resident
  histories, exercises nondefault streams, and runs concurrent solves while
  clearing the cache. The two-GPU test is present but was skipped locally.
- `cuda_resource_cache_native.cpp` checks actual handle/buffer reuse,
  capacity growth, the retention limit, partial allocation failure, exclusive
  ownership across four host threads and one idle entry after they return.
- A second CUDA build with C++14, static libstdc++/libgcc and
  `_GLIBCXX_USE_CXX11_ABI=0` also passed the A/B comparison, covering the host
  language/linkage settings used by Linux CI.
- Ruff and native ABI-table checks pass. Windows/macOS execution and other
  GPU architectures were not tested locally.

## Full-call performance

The existing `benchmark_wavefield_compression.py` ran in separate baseline and
candidate processes. Three paired rounds alternated variant order; each
process used three warmups and eleven measured iterations, with PyTorch's
allocator cache retained. No profiler ran during timing. The table reports the
median of the three per-process wall-time medians, including `compute`, loss
construction, backward and final synchronization. Input preparation and GC are
outside the measured interval, as in the existing benchmark.

All cases use one shot, eight receivers, PML thickness 4, order 2, `dx=0.01`,
`dt=1.5e-11`, source frequency 400 MHz and peak time `1e-10`. The fp32 default
uses reconstructed-R history; fp16 uses stored-R history. The 3D case uses
gradient mode 3. Other cases use mode 2.

| Physical grid | Steps | History | Offload | Baseline ms | Candidate ms | Wall-time change |
|---|---:|---|---|---:|---:|---:|
| 24 x 32 | 24 | fp32 | Yes | 10.484 | 8.907 | -15.0% |
| 96 x 128 | 160 | fp32 | Yes | 23.231 | 22.410 | -3.5% |
| 24 x 26 x 22 | 40 | fp32 | Yes | 16.040 | 14.483 | -9.7% |
| 24 x 32 | 24 | fp16 | Yes | 10.512 | 8.937 | -15.0% |
| 24 x 32 | 24 | fp32 | No (control) | 8.081 | 8.198 | +1.4% |

This is a single workstation, also driving a display, and variation is
material. In particular, the three fp16 candidate medians were 14.460, 8.937
and 8.859 ms (baseline 11.271, 9.880 and 10.512 ms). The resident baseline
medians were 13.118, 8.081 and 8.035 ms. There is no established improvement
on the resident path, which does not acquire cached workspaces; its small
median change is within the observed variation. These measurements support
reuse for the tested async cases, not a universal speedup or a cross-GPU claim.

Captured receivers and material gradients from every benchmark pair were
also compared: exact in 2D and for 3D receivers, with the above tolerance for
3D atomic gradients. All captures were finite.

## Reproduction

Build independent baseline/candidate source trees with the same CUDA toolchain
and GPU target. For either tree, use the existing benchmark, for example:

```bash
python tests/benchmark_wavefield_compression.py \
  --src-root /path/to/candidate/src \
  --nx 24 --ny 32 --nt 24 --shots 1 --receivers 8 --pml 4 \
  --mode fp32 --async-offload --peak-time 1e-10 \
  --warmup 3 --repeat 11 --keep-allocator-cache \
  --output-dir validation/n8/candidate_short2d
```

Use `--nx 96 --ny 128 --nt 160` for the larger 2D case;
`--nx 24 --ny 26 --nz 22 --gradient-mode 3 --nt 40` for 3D;
`--mode fp16` for low precision; omit `--async-offload` for the control.
Alternate baseline/candidate process order across three rounds and keep
separate output directories. `--save-captures PATH` saves tensors for comparison.

Run the regression suite against a freshly rebuilt candidate library:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run the independent-library numerical comparison with:

```bash
python tests/verify_cuda_resource_cache.py \
  --baseline /path/to/baseline/src/DeepGPR/lib \
  --candidate /path/to/candidate/src/DeepGPR/lib \
  --output validation/n8/comparison
```

The native ownership test's CUDA-toolkit build command is in `tests/README.md`.
Generated libraries, tensor captures, raw timing files and environment setup
artifacts are local validation outputs and are not part of this change.
