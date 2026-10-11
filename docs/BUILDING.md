# Building the native backends

The Python package loads prebuilt shared libraries from `src/DeepGPR/lib`. Rebuild them after changing `deepgpr.cu` or `deepgpr_cpu.c` (CI does this automatically, see `.github/workflows/workflow.yaml`).

## CUDA backend

Build the CUDA shared library from the repository root on Linux. `-lineinfo`
preserves source correlation for Nsight; `-Xptxas=-v` prints registers and spill
stores/loads for each kernel.

```bash
nvcc -std=c++14 -O3 -lineinfo -Xptxas=-v \
  -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89 \
  --shared -Xcompiler -fPIC \
  -o src/DeepGPR/lib/deepgpr.so src/DeepGPR/lib/deepgpr.cu
```

The command above is the audited RTX 4090 build path. Select the matching
architecture when building for a different GPU. The PTX target provides a
forward-compatible fallback; the release workflow additionally builds multiple
SASS targets and retains its PTX fallback. Do not add `--use_fast_math`
unless a separate numerical and gradient validation explicitly permits it.

The new INT8 path deliberately keeps native ABI 6 because the forward/backward
C signatures are unchanged. A capability symbol prevents an older ABI-6 CUDA
library from accepting the packed storage code and corrupting memory. External PML requires `deepgpr_supports_external_pml`; material gradients with PML additionally require `deepgpr_supports_pml_material_gradients` on CPU and CUDA, including INT8. This capability ensures PML material sensitivities are retained and summed back to physical edges. Rebuild native libraries on each target platform; older binaries are rejected for PML material gradients. ABI 6 signatures are unchanged.

The history and adjoint changes in [`CHANGELOG.md`](CHANGELOG.md) [Unreleased] add three more
capability symbols (ABI stays 6): `deepgpr_supports_rhs_reconstruction`
(E-only history, `wavefield_rhs_history`), `deepgpr_supports_physical_history`
(`wavefield_history_region="physical"`) and `deepgpr_supports_tm2d_fast_path`
(CUDA 2D TM fast path). A library built before them still works: `"auto"`
falls back to the E+R history and the general CUDA kernels, while an explicit
`wavefield_rhs_history="reconstructed"` or `wavefield_history_region="physical"`
raises `NativeLibraryError`. The prebuilt Windows and macOS binaries in
`src/DeepGPR/lib` predate these symbols until they are rebuilt.

## CPU backend

The CPU backend is a plain C shared library and is built with OpenMP by default. Build it into `src/DeepGPR/lib` before running with `device='cpu'`. You can control CPU thread count with `OMP_NUM_THREADS`.

```bash
# Linux
cc -std=c99 -O3 -fopenmp -fPIC -shared -o src/DeepGPR/lib/deepgpr_cpu.so src/DeepGPR/lib/deepgpr_cpu.c

# macOS
brew install libomp
LIBOMP_PREFIX="$(brew --prefix libomp)"
LIBOMP_RUNTIME_NAME="/opt/llvm-openmp/lib/libomp.dylib"
cc -std=c99 -O3 -Xpreprocessor -fopenmp -DDEEPGPR_USE_OPENMP -I"$LIBOMP_PREFIX/include" -L"$LIBOMP_PREFIX/lib" -fPIC -shared -o src/DeepGPR/lib/deepgpr_cpu.dylib src/DeepGPR/lib/deepgpr_cpu.c -lomp
cp "$LIBOMP_PREFIX/lib/libomp.dylib" src/DeepGPR/lib/libomp.dylib
install_name_tool -id "$LIBOMP_RUNTIME_NAME" src/DeepGPR/lib/libomp.dylib
LIBOMP_DEP="$(otool -L src/DeepGPR/lib/deepgpr_cpu.dylib | awk '/libomp\.dylib/ {print $1; exit}')"
install_name_tool -change "$LIBOMP_DEP" "$LIBOMP_RUNTIME_NAME" src/DeepGPR/lib/deepgpr_cpu.dylib
codesign --force --sign - src/DeepGPR/lib/libomp.dylib
codesign --force --sign - src/DeepGPR/lib/deepgpr_cpu.dylib
```

On Windows, build `src\DeepGPR\lib\deepgpr_cpu.dll` with MSVC:

```powershell
cl /LD /O2 /openmp /Fe:src\DeepGPR\lib\deepgpr_cpu.dll src\DeepGPR\lib\deepgpr_cpu.c
```

After rebuilding, run `python -m unittest tests/test_native_abi.py` to confirm the library exports ABI 6 and matches the Python ABI table.

## Python package archives

Development dependencies live in the `dev` and `examples` extras of `pyproject.toml`:

```bash
pip install -e ".[dev,examples]"
```

The source-distribution template is `docs/MANIFEST.in`. The `egg_info` command
configured in `pyproject.toml` reads it through `docs/build_support.py`. Both files
are included in source archives so wheels can also be built from the sdist.
Native libraries and sources remain package data under `src/DeepGPR/lib`.

Website maintenance and validation tools are under `website/scripts` and
`website/tools`; film source files are under `website/promo_pixel` and
`website/tools/promo`. Run their documented commands from the repository root.
GitHub Pages publishes the pages and assets while excluding those source tools.
