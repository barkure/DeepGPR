"""Section 5: gradient accuracy versus propagation time, loss window and dt.

Three separate factors (the time-segment count is in t05_checkpoint_storage.py):

(a) total number of steps nt = 50 ... 1600 (dt fixed; d_obs regenerated per nt),
(b) loss time window [t0, t1) inside a fixed nt = 800 simulation,
(c) fixed total time 10 ns with dt = 2.5e-11, 1.25e-11, 6.25e-12 (each dt
    is its own discrete problem: its AD is compared only with its own FD;
    the differences between the dt are discretisation errors and should
    shrink like dt^2).

Each configuration checks eps / sigma / joint random directions with Taylor
remainders, and a 16-pixel FD gradient sample for the cosine similarity.

Usage: python t03_time.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import ref_fdtd2d as ref  # noqa: E402
import torch  # noqa: E402

SHAPE = (40, 40)


def pixel_sample(problem, path, ad, rng, n=16, h_eps=3e-2, h_sig=1e-1):
    mask = gv.edge_mask(problem.eps.shape, problem.pml)
    cand = torch.nonzero(mask > 0)
    pick = cand[torch.randperm(len(cand), generator=rng)[:n]].tolist()
    out = {}
    for label in ("eps", "sig"):
        scale = float(problem.eps.mean() if label == "eps" else problem.sig.mean())
        h = h_eps if label == "eps" else h_sig
        ads, fds = [], []
        for pixel in pick:
            unit = torch.zeros(problem.eps.shape, dtype=torch.float64)
            unit[tuple(pixel)] = scale
            vals = []
            for hh in (h, h / 2):
                e1, s1 = (problem.eps + hh * unit, problem.sig) if label == "eps" else (problem.eps, problem.sig + hh * unit)
                e2, s2 = (problem.eps - hh * unit, problem.sig) if label == "eps" else (problem.eps, problem.sig - hh * unit)
                vals.append((gv.objective_forward(problem, path, e1, s1) - gv.objective_forward(problem, path, e2, s2)) / (2 * hh))
            fds.append((4 * vals[1] - vals[0]) / 3)
            grad = ad.grad_eps if label == "eps" else ad.grad_sig
            ads.append(float(grad[tuple(pixel)]) * scale)
        a, f = torch.tensor(ads), torch.tensor(fds)
        out[label] = dict(cosine=gv.cosine(a, f), rel_l2=gv.relative_l2(a, f), n=n)
    return out


def reference(problem):
    """float64 reference gradient (and float32 reference for the rounding level)."""
    out = {}
    for dtype in (torch.float64, torch.float32):
        eps = problem.eps.clone().to(dtype).requires_grad_(True)
        sig = problem.sig.clone().to(dtype).requires_grad_(True)
        J = gv.misfit(problem, ref.simulate(problem, eps, sig, dtype=dtype))
        J.backward()
        out[dtype] = (eps.grad.double(), sig.grad.double())
    return dict(eps=out[torch.float64][0], sig=out[torch.float64][1],
                eps_f32_level=gv.relative_l2(out[torch.float32][0], out[torch.float64][0]),
                sig_f32_level=gv.relative_l2(out[torch.float32][1], out[torch.float64][1]))


def checks(problem, path, rng, tag, results, refgrad=None):
    ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
    d_eps = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    d_sig = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    d_smooth = gv.directions(problem.eps.shape, problem.pml, "smooth", rng)
    row = dict(tag, path=path, J=ad.J, grad_eps_norm=float(ad.grad_eps.norm()),
               grad_sig_norm=float(ad.grad_sig.norm()), checks={})
    if refgrad is not None:
        row["vs_reference"] = dict(eps_rel_l2=gv.relative_l2(ad.grad_eps, refgrad["eps"]),
                                   sig_rel_l2=gv.relative_l2(ad.grad_sig, refgrad["sig"]),
                                   eps_cos=gv.cosine(ad.grad_eps, refgrad["eps"]),
                                   sig_cos=gv.cosine(ad.grad_sig, refgrad["sig"]),
                                   eps_f32_level=refgrad["eps_f32_level"], sig_f32_level=refgrad["sig_f32_level"])
        print(tag, path, "vs float64 reference", {k: f"{v:.2e}" for k, v in row["vs_reference"].items()}, flush=True)
    for label, de, ds, steps in (("eps", d_eps, None, gv.STEPS), ("sig", None, d_sig, gv.SIGMA_STEPS),
                                 ("joint", d_eps, d_sig, gv.STEPS), ("eps_smooth", d_smooth, None, gv.STEPS)):
        r = gv.physical_check(problem, path, "random", de, ds, ad, steps=steps)
        r.update(label=label)
        row["checks"][label] = r
        print(tag, gv.summarise(r), flush=True)
    if path in ("cpu", "cuda_fast"):
        row["pixels"] = pixel_sample(problem, path, ad, torch.Generator().manual_seed(11))
        print(tag, path, "pixel cosine", {k: round(v["cosine"], 8) for k, v in row["pixels"].items()}, flush=True)
    row["grad_eps"] = ad.grad_eps
    row["grad_sig"] = ad.grad_sig
    results.append(row)
    return ad


def main():
    gv.seed_all()
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    nt_rows, window_rows, dt_rows = [], [], []
    # (a) total number of time steps
    for nt in (50, 100, 200, 400, 800, 1600):
        problem = gv.make_problem_2d("D", shape=SHAPE, nt=nt)
        refgrad = reference(problem)
        for path in paths:
            checks(problem, path, torch.Generator().manual_seed(100 + nt), dict(series="nt", nt=nt), nt_rows,
                   refgrad)
    # (b) loss window inside nt = 800
    base = gv.make_problem_2d("D", shape=SHAPE, nt=800)
    windows = [(0, 100), (0, 200), (0, 300), (0, 400), (0, 600), (0, 800),
               (100, 800), (200, 800), (400, 800), (600, 800), (200, 400)]
    for t0, t1 in windows:
        problem = gv.Problem(**{**base.__dict__})
        window = torch.zeros(800, dtype=torch.float64)
        window[t0:t1] = 1.0
        problem.window = window
        refgrad = reference(problem)
        for path in paths[:2]:
            checks(problem, path, torch.Generator().manual_seed(300 + t0 + t1),
                   dict(series="window", t0=t0, t1=t1), window_rows, refgrad)
    # (c) fixed total time 10 ns, dt halved twice
    for dt in (2.5e-11, 1.25e-11, 6.25e-12):
        nt = int(round(1.0e-8 / dt))
        problem = gv.make_problem_2d("D", shape=SHAPE, nt=nt, dt=dt)
        # Misfit as a time integral: scale by dt / dt0 so J and the gradients are comparable.
        problem.scale = problem.scale / (dt / 2.5e-11) ** 0.5
        refgrad = reference(problem)
        for path in paths[:2]:
            checks(problem, path, torch.Generator().manual_seed(500), dict(series="dt", dt=dt, nt=nt), dt_rows,
                   refgrad)
    # Discretisation differences between dt (same path), using the dt=6.25e-12 run as reference.
    convergence = []
    for path in paths[:2]:
        rows = [r for r in dt_rows if r["path"] == path]
        ref = rows[-1]
        for r in rows:
            convergence.append(dict(path=path, dt=r["dt"],
                                    eps_rel_diff_to_finest=gv.relative_l2(r["grad_eps"], ref["grad_eps"]),
                                    sig_rel_diff_to_finest=gv.relative_l2(r["grad_sig"], ref["grad_sig"]),
                                    eps_cos_to_finest=gv.cosine(r["grad_eps"], ref["grad_eps"])))
        print("dt convergence", path, convergence[-3:], flush=True)
    # Window gradients: cosine to the full-window gradient (same path).
    for path in paths[:2]:
        full = [r for r in window_rows if r["path"] == path and (r["t0"], r["t1"]) == (0, 800)][0]
        for r in window_rows:
            if r["path"] == path:
                r["eps_cos_to_full_window"] = gv.cosine(r["grad_eps"], full["grad_eps"])
    for rows in (nt_rows, window_rows, dt_rows):
        for r in rows:
            r.pop("grad_eps"), r.pop("grad_sig")
    gv.save_json("t03_time.json", dict(shape=SHAPE, nt_series=nt_rows, window_series=window_rows,
                                       dt_series=dt_rows, dt_convergence=convergence))
    verdicts = {}
    for rows in (nt_rows, window_rows, dt_rows):
        for r in rows:
            for c in r["checks"].values():
                verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
    print("verdicts", verdicts)


if __name__ == "__main__":
    main()
