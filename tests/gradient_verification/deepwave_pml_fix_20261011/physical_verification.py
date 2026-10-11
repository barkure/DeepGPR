"""Synthetic, forward-only FD verification of the public discrete solver.

Reuses ONLY synthetic acquisition/model/solver helpers from gv_common. In
particular it does not use that module's edge_mask or frozen-PML objective.
All generated evidence goes into this independent audit directory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gv_common as gv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
gv.seed_all()
STEPS = (.3, .1, .03, .01, .003, .001, .0003, .0001)
RESULTS = []
FILE_TAG = ""
ENV = gv.environment()
ENV["command"] = "OMP_NUM_THREADS=1 /home/ll/anaconda3/envs/deepgpr/bin/python -u " + str(Path(__file__).resolve()) + " " + " ".join(sys.argv[1:])
ENV["hashes"] = {
    str(p.relative_to(gv.REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (gv.REPO / "src/DeepGPR/lib/deepgpr.cu", gv.REPO / "src/DeepGPR/lib/deepgpr_cpu.c",
              gv.REPO / "src/DeepGPR/lib/deepgpr.so", gv.REPO / "src/DeepGPR/lib/deepgpr_cpu.so")
}


def write(name, payload):
    if FILE_TAG:
        name = name.replace(".json", "_" + FILE_TAG + ".json")
    (HERE / "data" / name).write_text(json.dumps(gv.to_jsonable(payload), indent=1, ensure_ascii=False) + "\n")


def direction(shape, kind, seed=0, point=None):
    """Directions include physical edge cells unless explicitly interior_local."""
    rng = torch.Generator().manual_seed(gv.SEED + seed)
    if kind == "pixel":
        d = torch.zeros(shape, dtype=torch.float64)
        d[point or tuple(n // 2 for n in shape)] = 1.
    elif kind == "random":
        d = torch.randn(shape, generator=rng, dtype=torch.float64)
    elif kind in ("local", "interior_local"):
        centre = tuple(.5 * n for n in shape)
        d = gv.gaussian(shape, centre, .1 * min(shape))
        if kind == "interior_local":
            # An additional diagnostic with compact support, never the full test.
            for axis in range(len(shape)):
                indices = [slice(None)] * len(shape)
                indices[axis] = slice(0, 3)
                d[tuple(indices)] = 0
                indices[axis] = slice(shape[axis] - 3, shape[axis])
                d[tuple(indices)] = 0
    elif kind == "smooth":
        grids = torch.meshgrid(*[torch.linspace(0, 1, n, dtype=torch.float64) for n in shape], indexing="ij")
        d = .5 + sum(torch.cos(2 * math.pi * q + .2 * axis) for axis, q in enumerate(grids)) / len(shape)
    elif kind == "edge":
        d = torch.zeros(shape, dtype=torch.float64)
        for axis in range(len(shape)):
            lo = [slice(None)] * len(shape)
            lo[axis] = 0
            hi = list(lo)
            hi[axis] = -1
            d[tuple(lo)] = 1.
            d[tuple(hi)] = .7
    else:
        raise ValueError(kind)
    return d / d.abs().max().clamp_min(1e-30)


def check(p, path, ad, parameter, kind, label, seed=0, **options):
    d = direction(p.eps.shape, kind, seed)
    de = d if parameter in ("eps", "joint") else None
    ds = d if parameter in ("sig", "joint") else None
    r = gv.physical_check(p, path, parameter, de, ds, ad, steps=STEPS, **options)
    r.update(label=label, direction=kind, shape=list(p.eps.shape), nt=p.nt, dt=p.dt, dx=p.dx,
             pml=p.pml, order=p.order, source_direction=p.source_direction,
             receiver_component=p.receiver_component,
             eps_grad_norm=float(ad.grad_eps.norm()), sig_grad_norm=float(ad.grad_sig.norm()),
             perturbation_includes_boundary=kind != "interior_local", seed=gv.SEED + seed)
    RESULTS.append(r)
    print(f"{label} {path} {parameter}/{kind}: {r['verdict']} err={r['rel_err']:.3g} slope={r['fit_slope']:.3g}", flush=True)
    return r


def run_basic(paths):
    summary = []
    for model in "ABCDEFG":
        p = gv.make_problem_2d(model, shape=(40, 40), nt=400, pml=6)
        reference = None
        for path in paths:
            t0 = time.perf_counter()
            ad = gv.objective_ad(p, path, p.eps, p.sig)
            if reference is None:
                reference = ad
            summary.append(dict(model=model, path=path, J=ad.J, seconds=time.perf_counter()-t0,
                                waveform_relative_to_first=gv.relative_l2(ad.data, reference.data),
                                epsilon_gradient_relative_to_first=gv.relative_l2(ad.grad_eps, reference.grad_eps),
                                sigma_gradient_relative_to_first=gv.relative_l2(ad.grad_sig, reference.grad_sig),
                                epsilon_cosine=gv.cosine(ad.grad_eps, reference.grad_eps),
                                sigma_cosine=gv.cosine(ad.grad_sig, reference.grad_sig),
                                tm2d_fast=ad.tm2d_fast_path))
            for parameter in ("eps", "sig"):
                for kind in ("pixel", "random", "local", "smooth"):
                    check(p, path, ad, parameter, kind, "2d_" + model, seed=ord(model))
            for kind in ("random", "smooth", "interior_local"):
                check(p, path, ad, "joint", kind, "2d_" + model, seed=ord(model))
            if model == "G":
                for parameter in ("eps", "sig"):
                    check(p, path, ad, parameter, "edge", "2d_G_boundary")
            write("basic.json", dict(environment=ENV, results=RESULTS, solver_comparison=summary))


def run_time(path):
    for nt in (50, 100, 200, 400, 800):
        p = gv.make_problem_2d("D", shape=(40, 40), nt=nt, pml=6)
        ad = gv.objective_ad(p, path, p.eps, p.sig)
        for parameter in ("eps", "sig", "joint"):
            check(p, path, ad, parameter, "interior_local", "nt_series")
        # Preserve the full boundary direction also at long times.
        check(p, path, ad, "joint", "smooth", "nt_series_full")
        write("time.json", dict(environment=ENV, results=RESULTS))
    base = gv.make_problem_2d("D", shape=(40, 40), nt=800, pml=6)
    window_gradients = []
    for start, end in ((0, 100), (0, 200), (0, 400), (0, 800), (200, 400), (400, 800)):
        mask = torch.zeros(base.nt, dtype=torch.float64)
        mask[start:end] = 1.
        p = replace(base, window=mask)
        ad = gv.objective_ad(p, path, p.eps, p.sig)
        window_gradients.append((start, end, ad))
        for parameter in ("eps", "sig", "joint"):
            r = check(p, path, ad, parameter, "interior_local", "window_series")
            r.update(window_start=start, window_end=end)
        write("time.json", dict(environment=ENV, results=RESULTS))
    cosines = []
    for start, end, ad in window_gradients:
        full = window_gradients[3][2]
        cosines.append(dict(start=start, end=end, epsilon_cosine_to_full=gv.cosine(ad.grad_eps, full.grad_eps),
                            sigma_cosine_to_full=gv.cosine(ad.grad_sig, full.grad_sig)))
    # Keep total physical duration exactly 10 ns; only dt and nt change.
    for dt, nt in ((5e-11, 200), (2.5e-11, 400), (1.25e-11, 800)):
        p = gv.make_problem_2d("D", shape=(40, 40), nt=nt, dt=dt, pml=6)
        ad = gv.objective_ad(p, path, p.eps, p.sig)
        for parameter in ("eps", "sig", "joint"):
            r = check(p, path, ad, parameter, "interior_local", "fixed_duration_dt")
            r["duration"] = dt * nt
    write("time.json", dict(environment=ENV, results=RESULTS, window_cosines=cosines))


def run_pml(path):
    for pml in (0, 3, 6, 10, [3, 7, 5, 2, 0, 0]):
        for nt in (100, 400):
            p = gv.make_problem_2d("G", shape=(40, 40), nt=nt, pml=pml)
            ad = gv.objective_ad(p, path, p.eps, p.sig)
            for parameter in ("eps", "sig"):
                for kind in ("interior_local", "edge"):
                    check(p, path, ad, parameter, kind, "pml_sweep")
            write("pml.json", dict(environment=ENV, results=RESULTS))
    for order in (2, 4, 8):
        p = gv.make_problem_2d("D", shape=(40, 40), nt=400, pml=6, order=order)
        ad = gv.objective_ad(p, path, p.eps, p.sig)
        for parameter in ("eps", "sig", "joint"):
            check(p, path, ad, parameter, "interior_local", "stencil_order")
        write("pml.json", dict(environment=ENV, results=RESULTS))


def model3d(name):
    p = gv.make_problem_3d("D" if name == "D" else name, shape=(14, 14, 10), nt=200, pml=3)
    if name in ("F", "G"):
        x, y, z = torch.meshgrid(torch.arange(14), torch.arange(14), torch.arange(10), indexing="ij")
        if name == "F":
            anomaly = (x > 4 + .4 * y + .2 * z).double()
        else:
            anomaly = gv.gaussian(p.eps.shape, (.8, 12.7, 4.), 1.5)
        p.eps = 4 + 2 * anomaly
        p.sig = .01 + .015 * anomaly
        gv.attach_observed(p, *gv.observed_perturbation_3d(p.eps.shape))
    return p


def run_3d(paths):
    comparison = []
    for model in "ABCDEFG":
        p = model3d(model)
        first = None
        for path in paths:
            ad = gv.objective_ad(p, path, p.eps, p.sig)
            if first is None:
                first = ad
            comparison.append(dict(model=model, path=path, waveform_error=gv.relative_l2(ad.data, first.data),
                                   epsilon_error=gv.relative_l2(ad.grad_eps, first.grad_eps),
                                   sigma_error=gv.relative_l2(ad.grad_sig, first.grad_sig)))
            for parameter in ("eps", "sig"):
                for kind in ("pixel", "random", "local", "smooth"):
                    check(p, path, ad, parameter, kind, "3d_" + model, seed=ord(model))
            for kind in ("interior_local", "smooth"):
                check(p, path, ad, "joint", kind, "3d_" + model)
            if model == "G":
                for parameter in ("eps", "sig"):
                    check(p, path, ad, parameter, "edge", "3d_G_boundary")
            write("three_d.json", dict(environment=ENV, results=RESULTS, comparison=comparison))


def run_scatter(path):
    p = gv.make_problem_2d("D", shape=(40, 40), nt=400, pml=6)
    ad = gv.objective_ad(p, path, p.eps, p.sig)
    points = [(x, y) for x in (6, 14, 22, 30) for y in (8, 16, 24, 32)] + [(0, 10), (0, 20), (39, 15), (20, 0), (20, 39)]
    samples = []
    for parameter in ("eps", "sig"):
        for point in points:
            d = direction(p.eps.shape, "pixel", point=point)
            r = gv.physical_check(p, path, parameter, d if parameter == "eps" else None,
                                  d if parameter == "sig" else None, ad, steps=STEPS)
            scale = float((p.eps if parameter == "eps" else p.sig).mean())
            r.update(point=point, parameter=parameter, ad_component=r["g_ad"] / scale,
                     fd_component=r["g_fd"] / scale, boundary=0 in point or 39 in point)
            samples.append(r)
            print("sample", parameter, point, r["verdict"], r["rel_err"], flush=True)
            write("scatter.json", dict(environment=ENV, results=samples))
    cosines = []
    for parameter in ("eps", "sig"):
        for region in ("all", "interior"):
            selected = [r for r in samples if r["parameter"] == parameter and (region == "all" or not r["boundary"])]
            av = torch.tensor([r["ad_component"] for r in selected], dtype=torch.float64)
            fv = torch.tensor([r["fd_component"] for r in selected], dtype=torch.float64)
            cosines.append(dict(parameter=parameter, region=region, n=len(selected), cosine=gv.cosine(av, fv),
                                relative_norm_error=gv.relative_l2(av, fv), ad_norm=float(av.norm()), fd_norm=float(fv.norm()),
                                note="sampled pixel gradient, not full-grid FD gradient"))
    write("scatter.json", dict(environment=ENV, results=samples, sampled_cosines=cosines))


def main():
    global FILE_TAG
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("basic", "time", "pml", "3d", "scatter"))
    parser.add_argument("--paths", default="cpu,cuda_fast,cuda_general")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    paths = args.paths.split(",")
    FILE_TAG = args.tag
    start = time.perf_counter()
    print(json.dumps(ENV, indent=1), flush=True)
    if args.phase == "basic":
        run_basic(paths)
    elif args.phase == "time":
        run_time(paths[-1])
    elif args.phase == "pml":
        run_pml(paths[-1])
    elif args.phase == "3d":
        run_3d(paths)
    elif args.phase == "scatter":
        run_scatter(paths[-1])
    print("completed seconds", time.perf_counter()-start, flush=True)


if __name__ == "__main__":
    main()
