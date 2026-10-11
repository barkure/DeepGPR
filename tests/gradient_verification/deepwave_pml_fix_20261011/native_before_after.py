"""Isolate the native PML sensitivity change under identical fixed profiles.

The baseline uses the saved, Yee-corrected libraries from the previous audit,
with the new Python preprocessing and fixed profiles. Only the new capability
check is bypassed for this explicitly labelled diagnostic baseline. This is
not the old public numerical function, whose absorption profiles were different.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402
from DeepGPR.native import loader  # noqa: E402
from DeepGPR.solver import autograd  # noqa: E402
from physical_verification import direction  # noqa: E402

torch.set_num_threads(1)


@contextmanager
def native_version(version):
    directory = loader.LIB_DIR
    cached = loader._LOADED_LIBS.copy()
    loader.LIB_DIR = HERE / "native_libraries" / version
    loader._LOADED_LIBS.clear()
    check = autograd._check_forward_capabilities

    def baseline_check(lib, config, needs_model_gradient=False):
        # Older ABI-6 libraries lack only the new material-gradient capability.
        check(lib, config, needs_model_gradient=False)

    try:
        with patch.object(autograd, "_check_forward_capabilities",
                          baseline_check if version == "before" else check):
            yield
    finally:
        loader.LIB_DIR = directory
        loader._LOADED_LIBS.clear()
        loader._LOADED_LIBS.update(cached)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="cpu,cuda_fast,cuda_general")
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()
    paths = args.paths.split(",")
    rows = []
    for dims in (2, 3):
        cpu_reference = None
        problem = (gv.make_problem_2d("G", shape=(40, 40), nt=400, pml=[3, 7, 5, 2])
                   if dims == 2 else gv.make_problem_3d("D", shape=(14, 14, 10), nt=200, pml=3))
        problem.options["pml_max_velocity"] = 299792458.0
        for path in paths:
            if dims == 3 and path == "cuda_fast":
                continue
            forwards, gradients = {}, {}
            for version in ("before", "after"):
                with native_version(version):
                    with torch.no_grad():
                        out = gv.compute(problem, path, problem.eps, problem.sig, history=False)
                        forwards[version] = [t.detach().cpu().clone() for t in
                                             (*out[1], *out[2], *out[3], out[-1])]
                    gradients[version] = gv.objective_ad(problem, path, problem.eps, problem.sig)
            equality = [torch.equal(a, b) for a, b in
                        zip(forwards["before"], forwards["after"])]
            assert all(equality), (dims, path, equality)
            if path == "cpu":
                cpu_reference = gradients["after"]
            cross_backend = None
            if cpu_reference is not None:
                cross_backend = dict(
                    receiver_relative_l2=gv.relative_l2(gradients["after"].data, cpu_reference.data),
                    epsilon_relative_l2=gv.relative_l2(gradients["after"].grad_eps, cpu_reference.grad_eps),
                    sigma_relative_l2=gv.relative_l2(gradients["after"].grad_sig, cpu_reference.grad_sig),
                    epsilon_cosine=gv.cosine(gradients["after"].grad_eps, cpu_reference.grad_eps),
                    sigma_cosine=gv.cosine(gradients["after"].grad_sig, cpu_reference.grad_sig),
                )
            cases = []
            for parameter, kind in (("eps", "edge"), ("sig", "edge"), ("joint", "random")):
                unit = direction(problem.eps.shape, kind, 104)
                de = unit if parameter in ("eps", "joint") else None
                ds = unit if parameter in ("sig", "joint") else None
                with native_version("after"):
                    result = gv.physical_check(problem, path, parameter, de, ds, gradients["after"])
                before_ad = 0.
                if de is not None:
                    before_ad += float((gradients["before"].grad_eps * de * problem.eps.mean()).sum())
                if ds is not None:
                    before_ad += float((gradients["before"].grad_sig * ds * problem.sig.mean()).sum())
                result["before_ad"] = before_ad
                result["before_best_relative_error"] = min(
                    abs(r["fd"] - before_ad) / max(abs(r["fd"]), abs(before_ad), 1.e-30)
                    for r in result["rows"]
                )
                result["parameter"] = parameter
                result["direction"] = kind
                cases.append(result)
                print(dims, path, parameter, kind, result["verdict"],
                      "before", result["before_best_relative_error"], "after", result["rel_err"],
                      flush=True)
            rows.append(dict(dims=dims, path=path, bitwise_equal_receiver_and_30_states=equality,
                             pml=problem.pml, pml_max_velocity=299792458.0,
                             after_relative_to_cpu=cross_backend,
                             epsilon_gradient_relative_change=gv.relative_l2(
                                 gradients["before"].grad_eps, gradients["after"].grad_eps),
                             sigma_gradient_relative_change=gv.relative_l2(
                                 gradients["before"].grad_sig, gradients["after"].grad_sig),
                             cases=cases))
            payload = dict(environment=gv.environment(),
                           purpose="Isolate native sensitivity changes at identical fixed profiles; not legacy public profiles",
                           capability_bypass="Only diagnostic before-library material probe; all other checks retained",
                           library_hashes={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in (HERE / "native_libraries").glob("*/*.so")},
                           results=rows)
            (HERE / "data" / f"native_before_after_{args.tag}.json").write_text(
                json.dumps(gv.to_jsonable(payload), indent=1) + "\n")


if __name__ == "__main__":
    main()
