"""Independent long-time lossless-3D material checks, including model corners.

Also records raw state cotangent overflow on unobservable outer PML cells;
zero material sensitivities are handled exactly, not as a general NaN filter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
import torch  # noqa: E402
import DeepGPR  # noqa: E402
from DeepGPR.solver import autograd  # noqa: E402

torch.set_num_threads(1)


def statistics(tensor):
    finite = torch.isfinite(tensor)
    return dict(shape=list(tensor.shape), nonfinite=int((~finite).sum()),
                maximum_finite_absolute=float(tensor[finite].abs().max()) if bool(finite.any()) else None)


def run(eps, options, interval=1, history=False):
    return DeepGPR.compute(eps_r=eps, sigma=torch.zeros_like(eps),
                           model_gradient_sampling_interval=interval,
                           save_wavefield_history=history, **options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = args.device
    base = torch.full((20, 24, 24), 4., device=device)
    options = dict(device=device, dx=.02, dt=2.e-11, pmlthick=5, pml_max_velocity=299792458.,
                   mode=3, fdtd_order=2, wavefield_storage_dtype=torch.float32,
                   source_amplitudes=DeepGPR.ricker(5.e8, 1500, 2.e-11, 2.e-9).to(device).reshape(1, 1500, 1),
                   source_location=torch.tensor([[[10, 7, 12]]], dtype=torch.int32, device=device),
                   receiver_location=torch.tensor([[[8, 7, 10], [12, 7, 14]]], dtype=torch.int32, device=device))
    with torch.no_grad():
        data = run(base, options)[-1].double()
        scale = data.abs().max()
        target = .9 * data

    def objective(eps, interval=1, history=False):
        result = run(eps, options, interval, history)
        return .5 * ((result[-1].double() - target) / scale).square().sum(), result

    probes = []
    reference = None
    for interval in (10, 1):
        leaf = base.clone().requires_grad_()
        loss, out = objective(leaf, interval, True)
        row = dict(device=device, interval=interval, epsilon_history=statistics(out[-1].grad_fn.E_saved),
                   receiver=statistics(out[-1]), J=float(loss.detach()))
        original = autograd.invoke

        def capture(lib, name, order, arguments):
            result = original(lib, name, order, arguments)
            if name == "backward":
                row["raw_state_cotangents"] = {key: statistics(value) for key, value in arguments.items()
                                               if key.startswith("lambda")}
            return result

        with patch.object(autograd, "invoke", capture):
            loss.backward()
        row["material_gradient"] = statistics(leaf.grad)
        assert bool(torch.isfinite(leaf.grad).all()), row
        if interval == 1:
            reference = leaf.grad.detach().double()
        else:
            approximate = leaf.grad.detach().double()
        probes.append(row)
    probes[0]["gradient_relative_to_interval_1"] = float((approximate - reference).norm() / reference.norm())
    checks = []
    for kind in ("edge", "corner"):
        direction = torch.zeros_like(base, dtype=torch.float64)
        if kind == "corner":
            direction[0, 0, 0] = .25
        else:
            for axis in range(3):
                direction.select(axis, 0).fill_(.25)
                direction.select(axis, base.shape[axis] - 1).fill_(.175)
        adjoint = float((reference * direction).sum())
        rows = []
        for h in (.3, .1, .03, .01, .003, .001):
            with torch.no_grad():
                plus = float(objective((base.double() + h * direction).float())[0])
                minus = float(objective((base.double() - h * direction).float())[0])
            fd = (plus - minus) / (2 * h)
            rows.append(dict(h=h, J_plus=plus, J_minus=minus, fd=fd,
                             relative_error=abs(fd - adjoint) / max(abs(fd), abs(adjoint), 1.e-30)))
        passed = any(max(a["relative_error"], b["relative_error"]) < .002
                     for a, b in zip(rows, rows[1:]))
        check = dict(direction=kind, adjoint=adjoint, rows=rows,
                     boundary_mask=False, verdict="PASS" if passed else "UNRESOLVED")
        checks.append(check)
        print(json.dumps(check), flush=True)
    output = dict(device=device, nt=1500, sigma=0., pml_max_velocity=299792458.,
                  material_checks=checks, backward_probes=probes,
                  limitation="Raw initial-state cotangents on unobservable outer PML corners can overflow at long times; no general state-gradient stability claim.")
    (HERE / "data" / f"long_time_{device}.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
