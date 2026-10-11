"""Install the validated isolated Linux libraries using atomic replacement."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIB = ROOT / "src/DeepGPR/lib"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name in ("deepgpr_cpu.so", "deepgpr.so"):
    original = HERE / "native_libraries/before" / name
    assert sha(LIB / name) == sha(original), f"Workspace library changed externally: {name}"
    isolated = HERE / "native_libraries/after" / name
    temporary = LIB / (name + ".codex-native-fix.tmp")
    shutil.copy2(isolated, temporary)
    assert sha(temporary) == sha(isolated)
    os.replace(temporary, LIB / name)

manifest = {
    "installed_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": {name: sha(LIB / name) for name in ("deepgpr_cpu.c", "deepgpr.cu")},
    "binary_sha256": {phase: {name: sha(HERE / "native_libraries" / phase / name)
                              for name in ("deepgpr_cpu.so", "deepgpr.so")}
                      for phase in ("before", "after")},
    "installed_sha256": {name: sha(LIB / name) for name in ("deepgpr_cpu.so", "deepgpr.so")},
    "cpu_compile": "cc -std=c99 -O3 -fopenmp -fPIC -shared deepgpr_cpu.c -o deepgpr_cpu.so",
    "cuda_compile": "/usr/local/cuda-12.4/bin/nvcc -std=c++14 -O3 -arch=sm_89 --shared -Xcompiler -fPIC deepgpr.cu -o deepgpr.so",
    "platform": "Linux x86_64; CUDA binary built for sm_89; other OS precompiled binaries untouched",
    "complete_pml_material_derivative": "UNRESOLVED",
}
(HERE / "native_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
