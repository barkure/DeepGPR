"""Section 6: CPML and the material gradient.

1. Short propagation (no physical reflection yet, nt = 100) with and without
   CPML: both gradients verified and compared.
2. CPML thickness 0, 2, 5, 10, 20 (nt = 400, waves enter the CPML).
3. Asymmetric and partly disabled faces.
4. Anomaly at the boundary (model G):
   a. fixed-boundary rule (edge layer masked): AD vs FD;
   b. FD through compute() including the edge layer (CPML materials and
      coefficients rebuilt) vs AD: the design-level difference between the
      fixed-boundary gradient and the total derivative, quantified;
   c. FD with frozen extended materials and coefficients (edge layer
      included) vs AD.
5. Every edge cell class (first/last physical cell on each face, second
   layer, corner) by single-pixel FD with frozen CPML: truncation and sign.

Usage: python t04_pml.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402


def standard_checks(problem, path, rng, tag, out):
    ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
    d_eps = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    d_sig = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    for label, de, ds, steps in (("eps", d_eps, None, gv.STEPS), ("sig", None, d_sig, gv.SIGMA_STEPS),
                                 ("joint", d_eps, d_sig, gv.STEPS)):
        r = gv.physical_check(problem, path, "random", de, ds, ad, steps=steps)
        r.update(dict(tag), label=label, pml=problem.pml, tm2d_fast_path=ad.tm2d_fast_path)
        out.append(r)
        print(tag, gv.summarise(r), flush=True)
    return ad


def frozen_check(problem, path, ad, direction_eps, direction_sig, tag, out, steps=gv.STEPS):
    """Directional derivative with frozen extended materials and CPML coefficients."""
    s_eps, s_sig = float(problem.eps.mean()), float(problem.sig.mean())
    pe = direction_eps * s_eps if direction_eps is not None else torch.zeros_like(problem.eps)
    ps = direction_sig * s_sig if direction_sig is not None else torch.zeros_like(problem.sig)
    g = float((ad.grad_eps * pe).sum() + (ad.grad_sig * ps).sum())
    cs = float(((ad.grad_eps.norm() * pe.norm()) ** 2 + (ad.grad_sig.norm() * ps.norm()) ** 2) ** 0.5)
    eps_full = gv.pad_replicate(problem.eps, problem.pml)
    sig_full = gv.pad_replicate(problem.sig, problem.pml)
    J0, data0 = gv.objective_frozen_pml(problem, path, eps_full, sig_full)
    ep, sp = gv.embed_physical(pe, problem.pml), gv.embed_physical(ps, problem.pml)
    J_at = lambda h: gv.objective_frozen_pml(problem, path, eps_full + h * ep, sig_full + h * sp)[0]
    r = gv.taylor_check(J0, g, J_at, steps, cs_scale=cs)
    r.update(dict(tag), path=path, approach="frozen_pml", J0_matches_compute=(J0 == ad.J))
    out.append(r)
    print(tag, "frozen", gv.summarise(r), "J0==compute:", J0 == ad.J, flush=True)
    return r


def main():
    gv.seed_all()
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    out = []
    # 1. Short propagation: centre source, nearby receivers, nt = 100.
    short = {}
    for pml in (0, 10):
        problem = gv.make_problem_2d("D", nt=100, pml=pml)
        problem.src_loc = torch.tensor([[[24, 20, 0]], [[24, 28, 0]]], dtype=torch.int32)
        problem.rec_loc = torch.tensor([[[24 + dx, 24 + dy, 0] for dx in (-6, 0, 6) for dy in (-6, 0, 6)]] * 2,
                                       dtype=torch.int32)
        gv.attach_observed(problem, *gv.observed_perturbation_2d((48, 48)))
        for path in paths:
            ad = standard_checks(problem, path, torch.Generator().manual_seed(1), dict(series="short", pml=pml), out)
            short[(pml, path)] = ad
    for path in paths:
        a, b = short[(0, path)], short[(10, path)]
        row = dict(series="short_compare", path=path,
                   data_rel_diff=gv.relative_l2(b.data, a.data),
                   eps_grad_rel_diff=gv.relative_l2(b.grad_eps, a.grad_eps),
                   sig_grad_rel_diff=gv.relative_l2(b.grad_sig, a.grad_sig),
                   eps_grad_cos=gv.cosine(b.grad_eps, a.grad_eps))
        out.append(row)
        print("short propagation pml 0 vs 10", row, flush=True)
    # 2. Thickness.
    for pml in (0, 2, 5, 10, 20):
        problem = gv.make_problem_2d("D", nt=400, pml=pml)
        for path in paths:
            standard_checks(problem, path, torch.Generator().manual_seed(2), dict(series="thickness"), out)
    # 3. Asymmetric / partly disabled faces.
    for pml in ([3, 12, 0, 7], [0, 10, 10, 0], [6, 0, 0, 0], [1, 1, 15, 2]):
        problem = gv.make_problem_2d("D", nt=400, pml=pml)
        for path in paths:
            standard_checks(problem, path, torch.Generator().manual_seed(3), dict(series="faces"), out)
    # 4. Anomalies at the boundary (model G).
    problem = gv.make_problem_2d("G", nt=400, pml=10)
    shape = problem.eps.shape
    for path in paths:
        rng = torch.Generator().manual_seed(4)
        ad = standard_checks(problem, path, rng, dict(series="G_masked"), out)
        # Directions that include the edge layer (no mask).
        unmasked_eps = torch.randn(shape, generator=rng, dtype=torch.float64)
        unmasked_eps /= unmasked_eps.abs().max()
        unmasked_sig = torch.randn(shape, generator=rng, dtype=torch.float64)
        unmasked_sig /= unmasked_sig.abs().max()
        edge_only = 1.0 - gv.edge_mask(shape, problem.pml)
        for label, de, ds in (("eps", unmasked_eps, None), ("sig", None, unmasked_sig),
                              ("eps_edge_layer_only", unmasked_eps * edge_only, None)):
            steps = gv.SIGMA_STEPS if label == "sig" else gv.STEPS
            # b. total derivative through compute() (CPML rebuilt from the perturbed edge).
            r = gv.physical_check(problem, path, "random_unmasked", de, ds, ad, steps=steps)
            r.update(series="G_total_derivative", label=label,
                     note="expected to differ from AD by design (CPML materials/coefficients rebuilt)")
            out.append(r)
            print("G total-derivative FD vs fixed-boundary AD", gv.summarise(r), flush=True)
            # c. frozen CPML.
            frozen_check(problem, path, ad, de, ds, dict(series="G_frozen", label=label), out, steps=steps)
    # 5. Edge cells, single pixels, frozen CPML (CPU and CUDA fast path).
    problem = gv.make_problem_2d("D", nt=400, pml=[4, 7, 5, 6])
    nx, ny = problem.eps.shape
    cells = {"x0_first": (0, 20), "x0_second": (1, 20), "xm_last": (nx - 1, 30), "xm_second": (nx - 2, 30),
             "y0_first": (25, 0), "y0_second": (25, 1), "ym_last": (18, ny - 1), "ym_second": (18, ny - 2),
             "corner_x0y0": (0, 0), "corner_xmym": (nx - 1, ny - 1), "interior": (24, 24)}
    for path in paths[:2]:
        ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
        for name, cell in cells.items():
            for label in ("eps", "sig"):
                d = torch.zeros(problem.eps.shape, dtype=torch.float64)
                d[cell] = 1.0
                frozen_check(problem, path, ad, d if label == "eps" else None, d if label == "sig" else None,
                             dict(series="edge_cells", cell=name, label=label), out,
                             steps=gv.STEPS if label == "eps" else gv.SIGMA_STEPS)
    gv.save_json("t04_pml.json", dict(results=out))
    verdicts = {}
    for r in out:
        if "verdict" in r:
            key = (r.get("series"), r["verdict"])
            verdicts[key] = verdicts.get(key, 0) + 1
    print("verdicts", verdicts)


if __name__ == "__main__":
    main()
