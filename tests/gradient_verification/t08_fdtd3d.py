"""3D FDTD gradients (mode = 3), CPU and CUDA general path.

Models A-E on a 16 x 16 x 12 physical grid with CPML 4 on all six faces, two
shots, 18 receivers on two planes, nt = 150, order 2 (model D also order 4 and
8), eps / sigma / joint / source directions of the four kinds, plus
asymmetric and disabled CPML faces, CPML thickness 0 / 2 / 6, x- and
y-polarised sources, and frozen-CPML single-cell checks on the z faces.
(3D initial-state and dot-product tests are in t02, 3D history options in t05.)

Usage: python t08_fdtd3d.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import t04_pml as t4  # noqa: E402
import torch  # noqa: E402

KINDS = ("pixel", "random", "local", "smooth")


def run(problem, path, model, rng, results, tag=""):
    ad = gv.objective_ad(problem, path, problem.eps, problem.sig, wrt=("eps", "sig", "src"))
    shape = problem.eps.shape
    centre = [int(0.45 * shape[0]), int(0.4 * shape[1]), int(0.5 * shape[2])]
    for kind in KINDS:
        pixel = centre if kind in ("pixel", "local") else None
        d_eps = gv.directions(shape, problem.pml, kind, rng, pixel=pixel)
        d_sig = gv.directions(shape, problem.pml, kind, rng, pixel=pixel)
        for label, de, ds, steps in (("eps", d_eps, None, gv.STEPS), ("sig", None, d_sig, gv.SIGMA_STEPS),
                                     ("joint", d_eps, d_sig, gv.STEPS)):
            r = gv.physical_check(problem, path, kind, de, ds, ad, steps=steps)
            r.update(label=label, model=model, tag=tag, dims=3)
            results.append(r)
            print(tag, model, gv.summarise(r), flush=True)
    q = gv.smooth_noise((problem.nt,), rng, passes=8).reshape(1, problem.nt, 1)
    q = q / float(q.abs().max()) * float(problem.wavelet.abs().max())
    g_src = float((ad.grad_src * q).sum())
    J_at = lambda h: gv.objective_forward(problem, path, problem.eps, problem.sig,
                                          source=(problem.wavelet.double() + h * q).float())
    r = gv.taylor_check(ad.J, g_src, J_at, gv.STEPS)
    r.update(path=path, kind="smooth", label="source", model=model, tag=tag, dims=3)
    results.append(r)
    print(tag, model, gv.summarise(r), flush=True)
    return ad


def random_only(problem, path, model, rng, results, tag):
    ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
    d_eps = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    d_sig = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    for label, de, ds, steps in (("eps", d_eps, None, gv.STEPS), ("sig", None, d_sig, gv.SIGMA_STEPS),
                                 ("joint", d_eps, d_sig, gv.STEPS)):
        r = gv.physical_check(problem, path, "random", de, ds, ad, steps=steps)
        r.update(label=label, model=model, tag=tag, dims=3)
        results.append(r)
        print(tag, model, gv.summarise(r), flush=True)
    return ad


def main():
    gv.seed_all()
    paths = ["cpu"] + (["cuda_general"] if gv.cuda_available() else [])
    results = []
    for model in "ABCDE":
        problem = gv.make_problem_3d(model)
        for path in paths:
            ad = run(problem, path, model, torch.Generator().manual_seed(gv.SEED + ord(model)), results, "base")
            if path != "cpu" and ad.tm2d_fast_path:
                raise RuntimeError("3D run reported the 2D TM fast path")
    for order in (4, 8):
        problem = gv.make_problem_3d("D", order=order)
        for path in paths:
            random_only(problem, path, f"D-order{order}", torch.Generator().manual_seed(order), results, "order")
    for pml in ([2, 5, 0, 3, 4, 1], 0, 2, 6):
        problem = gv.make_problem_3d("D", pml=pml)
        for path in paths:
            random_only(problem, path, f"D-pml{pml}", torch.Generator().manual_seed(11), results, "pml")
    for direction in (0, 1):
        problem = gv.make_problem_3d("D", obs_path="cpu")
        problem.source_direction = direction
        problem.receiver_component = direction
        gv.attach_observed(problem, *gv.observed_perturbation_3d(problem.eps.shape))
        for path in paths:
            random_only(problem, path, f"D-pol{direction}", torch.Generator().manual_seed(13), results, "polarisation")
    # Frozen-CPML single cells on the z faces (first / last physical layer) and a corner.
    problem = gv.make_problem_3d("D", pml=[2, 3, 2, 3, 3, 2])
    nx, ny, nz = problem.eps.shape
    cells = {"z0_first": (8, 8, 0), "zm_last": (8, 8, nz - 1), "corner": (0, 0, 0), "x0_first": (0, 7, 6)}
    for path in paths:
        ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
        for name, cell in cells.items():
            d = torch.zeros(problem.eps.shape, dtype=torch.float64)
            d[cell] = 1.0
            for label in ("eps", "sig"):
                t4.frozen_check(problem, path, ad, d if label == "eps" else None, d if label == "sig" else None,
                                dict(series="3d_edge_cells", cell=name, label=label), results,
                                steps=gv.STEPS if label == "eps" else gv.SIGMA_STEPS)
    gv.save_json("t08_fdtd3d.json", dict(results=results, shape=(16, 16, 12), nt=150, pml=4))
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("verdicts", counts)


if __name__ == "__main__":
    main()
