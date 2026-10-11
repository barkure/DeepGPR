"""Compare isolated old/new native libraries and run focused regressions."""

import argparse
import io
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1]
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(HERE.parent))

import gv_common as gv  # noqa: E402
import test_unupdated_e_material_gradients as regression  # noqa: E402
import torch  # noqa: E402
from DeepGPR.native import loader  # noqa: E402


def select_libraries(phase):
    loader.LIB_DIR = HERE / "native_libraries" / phase
    loader._LOADED_LIBS.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = ["cpu"] if args.device == "cpu" else ["cuda_fast", "cuda_general"]
    outcomes = {}
    waveforms = {}
    for phase in ("before", "after"):
        select_libraries(phase)
        cases = []
        for path in paths:
            for rhs in ("stored", "reconstructed"):
                for source in (0.0, 0.0002):
                    cases.append(regression.material_case(path, 2, 2, rhs, source))
            if path != "cuda_fast":
                for rhs in ("stored", "reconstructed"):
                    cases.append(regression.material_case(path, 1, 3, rhs, cpml_correction=True))
            if args.device == "cuda":
                for rhs in ("stored", "reconstructed"):
                    for source in (0.0, 0.0002):
                        cases.append(regression.material_case(path, 2, 2, rhs, source, compression="int8"))
        outcomes[phase] = cases
        # The native forward implementation is unchanged by the patch.
        problem = gv.make_problem_2d("D", shape=(32, 32), nt=120, pml=4, obs_path="cpu")
        data = gv.forward_data(problem, paths[0], problem.eps, problem.sig).detach().cpu()
        waveforms[phase] = data
    select_libraries("after")
    cls = regression.UnupdatedElectricMaterialGradientTests
    names = [name for name in unittest.defaultTestLoader.getTestCaseNames(cls)
             if name.startswith("test_cpu") == (args.device == "cpu")]
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.TestSuite(cls(name) for name in names))
    log = output.getvalue()
    (HERE / f"native_fix_{args.device}_regression.log").write_text(log)
    print(log)
    difference = waveforms["after"] - waveforms["before"]
    payload = {
        "device": args.device, "cases": outcomes,
        "normal_interior_forward": {
            "shape": list(waveforms["after"].shape),
            "bitwise_identical": bool(torch.equal(waveforms["after"], waveforms["before"])),
            "max_abs_diff": float(difference.abs().max()),
            "relative_l2_diff": float(difference.double().norm() / waveforms["before"].double().norm()),
        },
        "regression_tests": {"ran": result.testsRun, "failures": len(result.failures),
                             "errors": len(result.errors), "skipped": len(result.skipped),
                             "successful": result.wasSuccessful()},
    }
    (HERE / f"native_fix_{args.device}_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, indent=2))
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
