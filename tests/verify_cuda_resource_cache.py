"""Compare freshly built CUDA libraries in separate processes (requires CUDA PyTorch)."""

import argparse
from pathlib import Path
import subprocess
import sys

import torch

from test_cuda_resource_cache import CudaResourceCacheTests, run_case
from DeepGPR.native import loader


CASES = [
    dict(shape=(17, 19), dtype=torch.float32, rhs="stored"),
    dict(shape=(23, 27), dtype=torch.float16, rhs="stored", interval=2),
    dict(shape=(11, 13, 9), dtype=torch.bfloat16, rhs="stored"),
    dict(shape=(13, 15), dtype=torch.float32, rhs="reconstructed"),
    dict(shape=(19, 21), dtype=torch.float16, rhs="reconstructed"),
    dict(shape=(17, 19), material=False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline library directory")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate library directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        loader.LIB_DIR = getattr(args, args.worker).resolve()
        results = [run_case(**case) for _ in range(2) for case in CASES]
        torch.save(results, args.output / f"{args.worker}.pt")
        print(f"{args.worker}: captured {len(results)} runs", flush=True)
        return
    for variant in ("baseline", "candidate"):
        subprocess.run([sys.executable, __file__, "--baseline", str(args.baseline),
                        "--candidate", str(args.candidate), "--output", str(args.output),
                        "--worker", variant], check=True)
    baseline = torch.load(args.output / "baseline.pt", weights_only=True)
    candidate = torch.load(args.output / "candidate.pt", weights_only=True)
    for index, (a, b) in enumerate(zip(baseline, candidate)):
        three_d = len(CASES[index % len(CASES)]["shape"]) == 3
        CudaResourceCacheTests().assert_results(a, b, three_d)
    print("A/B passed: exact forwards and 2D gradients; tolerant 3D atomic gradients")


if __name__ == "__main__":
    main()
