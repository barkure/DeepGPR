"""Old/new native comparison on zero-initial-state synthetic D problems."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gv_common as gv
import torch
from DeepGPR.native import loader


def differences(before, after):
    delta = after.double() - before.double()
    norm = before.double().norm()
    return {
        "shape": list(after.shape), "bitwise_identical": bool(torch.equal(before, after)),
        "max_abs_diff": float(delta.abs().max()),
        "relative_l2_diff": float(delta.norm() / norm) if norm else float(delta.norm()),
        "before_l2": float(norm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    loader.LIB_DIR = HERE / "native_libraries/before"
    loader._LOADED_LIBS.clear()
    problems = {2: gv.make_problem_2d("D"), 3: gv.make_problem_3d("D")}
    paths = [(2, "cpu"), (3, "cpu")] if args.device == "cpu" else [
        (2, "cuda_fast"), (2, "cuda_general"), (3, "cuda_general")]
    arrays = {}
    for phase in ("before", "after"):
        loader.LIB_DIR = HERE / "native_libraries" / phase
        loader._LOADED_LIBS.clear()
        arrays[phase] = {}
        for dims, path in paths:
            for rhs in ("stored", "reconstructed"):
                problem = problems[dims]
                result = gv.objective_ad(
                    problem, path, problem.eps, problem.sig, wrt=("eps", "sig", "src"),
                    model_gradient_sampling_interval=1, wavefield_storage_dtype=torch.float32,
                    wavefield_rhs_history=rhs)
                key = f"{dims}d_{path}_{rhs}"
                arrays[phase][key] = {
                    "data": result.data.detach().cpu(), "eps": result.grad_eps.detach().cpu(),
                    "sigma": result.grad_sig.detach().cpu(), "source": result.grad_src.detach().cpu(),
                }
    rows = {key: {name: differences(arrays["before"][key][name], arrays["after"][key][name])
                  for name in arrays["before"][key]} for key in arrays["before"]}
    output = {"device": args.device, "models": "synthetic D; zero initial states; full physical gradients",
              "comparison": rows, "all_bitwise_identical": all(
                  entry["bitwise_identical"] for row in rows.values() for entry in row.values())}
    (HERE / f"normal_native_{args.device}_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    torch.save(arrays, HERE / f"normal_native_{args.device}_arrays.pt")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
