"""Follow-up of the INCONCLUSIVE items of t01 (model E, sigma / joint random directions).

The same directions as in t01 are rebuilt from the same random stream and
checked with (1) a dense geometric step grid (13 steps) and a noise-weighted
extrapolation fd(h) -> g0 with a standard error, and (2) for sigma, the
relative (multiplicative) perturbation sigma * (1 + h p), which allows steps
up to 0.9 without leaving sigma >= 0, for every model A-G.

Usage: python t01b_resolve.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import t01_fdtd2d_basic as t1  # noqa: E402
import torch  # noqa: E402


def t01_random_directions(problem, model):
    rng = torch.Generator().manual_seed(gv.SEED + ord(model))
    out = {}
    for k_index, kind in enumerate(t1.KINDS):
        pixel = t1.pixel_for(problem, k_index) if kind in ("pixel", "local") else None
        d_eps = gv.directions(problem.eps.shape, problem.pml, kind, rng, pixel=pixel)
        d_sig = gv.directions(problem.eps.shape, problem.pml, kind, rng, pixel=pixel)
        out[kind] = (d_eps, d_sig)
    return out


def main():
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    rows = []
    problem = gv.make_problem_2d("E")
    d_eps, d_sig = t01_random_directions(problem, "E")["random"]
    s_eps, s_sig = float(problem.eps.mean()), float(problem.sig.mean())
    for path in paths:
        ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
        for label, pe, ps, hmax in (("sig", torch.zeros_like(problem.eps), d_sig * s_sig, 0.3),
                                    ("joint", d_eps * s_eps, d_sig * s_sig, 3e-2)):
            g = float((ad.grad_eps * pe).sum() + (ad.grad_sig * ps).sum())
            J_at = lambda h: gv.objective_forward(problem, path, problem.eps + h * pe, problem.sig + h * ps)
            r = gv.dense_extrapolation(J_at, g, hmax, hmax / 30, n=13)
            verdict = "PASS" if r["z"] <= 3.0 and r["rel_se"] <= 2e-3 else (
                "FAIL" if r["z"] > 3.0 and r["rel_se"] <= 2e-3 else "INCONCLUSIVE")
            r.update(model="E", path=path, label=label, kind="random (t01 direction)", verdict=verdict)
            rows.append(r)
            print(f"E {path} {label}: AD={g:.6e} g0={r['g0']:.6e}+-{r['se']:.1e} rel={r['rel_err']:.2e} "
                  f"z={r['z']:.2f} -> {verdict}", flush=True)
    # Multiplicative sigma directions, every model.
    for model in "ABCDEFG":
        problem = gv.make_problem_2d(model)
        rng = torch.Generator().manual_seed(gv.SEED + 1000 + ord(model))
        p = gv.directions(problem.eps.shape, problem.pml, "random", rng)
        for path in paths:
            ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
            ps = p * problem.sig          # relative perturbation sigma * (1 + h p)
            g = float((ad.grad_sig * ps).sum())
            cs = float(ad.grad_sig.norm() * ps.norm())
            J_at = lambda h: gv.objective_forward(problem, path, problem.eps, problem.sig + h * ps)
            r = gv.taylor_check(ad.J, g, J_at, (9e-1, 6e-1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3), cs_scale=cs)
            r.update(model=model, path=path, label="sig_relative", kind="random")
            rows.append(r)
            print(model, gv.summarise(r), flush=True)
    gv.save_json("t01b_resolve.json", dict(results=rows))


if __name__ == "__main__":
    main()
