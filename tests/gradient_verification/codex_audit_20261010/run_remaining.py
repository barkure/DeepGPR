"""Run the final native-library synthetic GPU phases serially."""
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
os.environ["OMP_NUM_THREADS"] = "1"
jobs = [
    ("three_d", ["3d", "--paths", "cpu,cuda_general"]),
    ("time_cuda", ["time", "--paths", "cuda_fast", "--tag", "cuda"]),
    ("pml_cuda", ["pml", "--paths", "cuda_fast", "--tag", "cuda"]),
    ("scatter", ["scatter", "--paths", "cuda_fast"]),
    ("basic_after_native_fix", ["basic", "--tag", "after_native_fix"]),
]
for name, arguments in jobs:
    print("START", name, flush=True)
    with (HERE / "logs" / (name + ".log")).open("w") as log:
        result = subprocess.run([sys.executable, "-u", str(HERE / "physical_verification.py"), *arguments],
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    print("END", name, "exit", result.returncode, flush=True)
    if result.returncode:
        sys.exit(result.returncode)
