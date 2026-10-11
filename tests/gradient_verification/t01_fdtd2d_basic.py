"""Phase 1: 2D FDTD material / source gradients, every model, direction and path.

For models A-G (48 x 48 physical cells, CPML 10 on the four 2D faces, two
shots, 18 receivers, nt = 400, order 2) and the three solver paths, check
the directional derivative of the float64 misfit along

* eps_r, sigma and joint (dimensionless, scaled by the mean parameter)
  directions of four kinds: pixel, random, local, global smooth,
* a smooth random perturbation of the source waveform,

with central differences and Taylor remainders from forward-only runs.
Model D additionally runs orders 4 and 8, and a pixel-wise FD gradient field
(48 pixels) for the AD/FD scatter plots and the cosine similarity.

Usage: python t01_fdtd2d_basic.py [--quick]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402

MODELS = "ABCDEFG"
KINDS = ("pixel", "random", "local", "smooth")


def pixel_for(problem, kind_index):
    nx, ny = problem.eps.shape
    src = problem.src_loc[0, 0].tolist()[:2]
    candidates = [
        (int(0.42 * nx), int(0.38 * ny)),        # eps anomaly of D
        tuple(src),                               # source cell (source-term derivative)
        (int(0.62 * nx), int(0.66 * ny)),        # sigma anomaly of D
        (1, ny // 2),                             # first cell inside the masked edge layer
    ]
    return candidates[kind_index % len(candidates)]


def run_case(problem, path, model, rng, results, label_prefix=""):
    t0 = time.time()
    ad = gv.objective_ad(problem, path, problem.eps, problem.sig, wrt=("eps", "sig", "src"))
    expected_fast = path == "cuda_fast"
    if path != "cpu" and ad.tm2d_fast_path != expected_fast:
        raise RuntimeError(f"{path}: tm2d_fast_path={ad.tm2d_fast_path}")
    shape = problem.eps.shape
    for k_index, kind in enumerate(KINDS):
        pixel = pixel_for(problem, k_index) if kind in ("pixel", "local") else None
        d_eps = gv.directions(shape, problem.pml, kind, rng, pixel=pixel)
        d_sig = gv.directions(shape, problem.pml, kind, rng, pixel=pixel)
        for label, de, ds, steps in (
            ("eps", d_eps, None, gv.STEPS),
            ("sig", None, d_sig, gv.SIGMA_STEPS),
            ("joint", d_eps, d_sig, gv.STEPS),
        ):
            r = gv.physical_check(problem, path, kind, de, ds, ad, steps=steps)
            r.update(label=label_prefix + label, model=model, pixel=pixel,
                     tm2d_fast_path=ad.tm2d_fast_path, grad_eps_norm=float(ad.grad_eps.norm()),
                     grad_sig_norm=float(ad.grad_sig.norm()))
            results.append(r)
            print(gv.summarise(r), flush=True)
        # Joint = eps + sigma (linearity of the AD directional derivative).
    # Source waveform direction: smooth random perturbation, peak = peak of the wavelet.
    q = gv.smooth_noise((problem.nt,), rng, passes=8).reshape(1, problem.nt, 1)
    q = q / float(q.abs().max()) * float(problem.wavelet.abs().max())
    g_src = float((ad.grad_src * q).sum())
    J_at = lambda h: gv.objective_forward(problem, path, problem.eps, problem.sig,
                                          source=(problem.wavelet.double() + h * q).float())
    r = gv.taylor_check(ad.J, g_src, J_at, gv.STEPS)
    r.update(path=path, kind="smooth", label=label_prefix + "source", model=model,
             tm2d_fast_path=ad.tm2d_fast_path, grad_src_norm=float(ad.grad_src.norm()))
    results.append(r)
    print(gv.summarise(r), f"({time.time() - t0:.1f}s)", flush=True)
    return ad


def pixel_field(problem, path, ad, rng, n_pixels, results):
    """FD gradient at individual pixels (Richardson extrapolated central differences)."""
    shape = problem.eps.shape
    mask = gv.edge_mask(shape, problem.pml)
    candidates = torch.nonzero(mask > 0)
    pick = candidates[torch.randperm(len(candidates), generator=rng)[:n_pixels]]
    rows = []
    for pixel in pick.tolist():
        for label, scale, h1 in (("eps", float(problem.eps.mean()), 3e-2), ("sig", float(problem.sig.mean()), 1e-1)):
            unit = torch.zeros(shape, dtype=torch.float64)
            unit[tuple(pixel)] = scale
            fds = []
            for h in (h1, h1 / 2):
                if label == "eps":
                    jp = gv.objective_forward(problem, path, problem.eps + h * unit, problem.sig)
                    jm = gv.objective_forward(problem, path, problem.eps - h * unit, problem.sig)
                else:
                    jp = gv.objective_forward(problem, path, problem.eps, problem.sig + h * unit)
                    jm = gv.objective_forward(problem, path, problem.eps, problem.sig - h * unit)
                fds.append((jp - jm) / (2 * h))
            fd = (4 * fds[1] - fds[0]) / 3   # Richardson (h^2 term removed)
            grad = ad.grad_eps if label == "eps" else ad.grad_sig
            rows.append(dict(path=path, label=label, pixel=pixel, ad=float(grad[tuple(pixel)]) * scale,
                             fd=fd, fd_h=fds[0], fd_h2=fds[1]))
    for label in ("eps", "sig"):
        a = torch.tensor([r["ad"] for r in rows if r["label"] == label])
        f = torch.tensor([r["fd"] for r in rows if r["label"] == label])
        summary = dict(path=path, label=label, n=len(a), cosine=gv.cosine(a, f),
                       rel_l2=gv.relative_l2(a, f), max_abs_ad=float(a.abs().max()))
        print("pixel field", summary, flush=True)
        results.append(dict(pixel_field_summary=summary))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    gv.seed_all()
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    results, pixel_rows = [], []
    models = "D" if args.quick else MODELS
    for model in models:
        problem = gv.make_problem_2d(model)
        for path in paths:
            rng = torch.Generator().manual_seed(gv.SEED + ord(model))
            run_case(problem, path, model, rng, results)
    # Higher stencil orders (model D, random directions only).
    for order in (4, 8):
        problem = gv.make_problem_2d("D", order=order)
        for path in paths:
            rng = torch.Generator().manual_seed(gv.SEED + order)
            ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
            d_eps = gv.directions(problem.eps.shape, problem.pml, "random", rng)
            d_sig = gv.directions(problem.eps.shape, problem.pml, "random", rng)
            for label, de, ds, steps in (("eps", d_eps, None, gv.STEPS), ("sig", None, d_sig, gv.SIGMA_STEPS),
                                         ("joint", d_eps, d_sig, gv.STEPS)):
                r = gv.physical_check(problem, path, "random", de, ds, ad, steps=steps)
                r.update(label=label, model=f"D-order{order}", tm2d_fast_path=ad.tm2d_fast_path)
                results.append(r)
                print(gv.summarise(r), flush=True)
    # Pixel-wise FD gradient field for the scatter plots (model D).
    problem = gv.make_problem_2d("D")
    for path in paths[:2]:
        ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
        rng = torch.Generator().manual_seed(gv.SEED + 7)
        pixel_rows += pixel_field(problem, path, ad, rng, 12 if args.quick else 48, results)
    config = dict(shape=(48, 48), nt=400, dt=2.5e-11, dx=0.02, pml=10, order=2, freq=4e8, shots=2,
                  receivers=18, steps=gv.STEPS, sigma_steps=gv.SIGMA_STEPS, seed=gv.SEED,
                  misfit="0.5*sum(((d - d_obs)/max|d_obs|)^2), float64",
                  observed="d_obs from the model plus a separate eps/sigma perturbation")
    gv.save_json("t01_fdtd2d_basic.json", dict(config=config, results=results, pixel_rows=pixel_rows))
    counts = {}
    for r in results:
        if "verdict" in r:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("verdicts:", counts)


if __name__ == "__main__":
    main()
