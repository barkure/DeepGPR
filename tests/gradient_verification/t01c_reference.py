"""2D gradients against the independent float64 reference (ref_fdtd2d.py).

For models A-G (order 2, the t01 configuration, plus asymmetric/disabled CPML
faces and anisotropic spacing) compare, on every path:

* the receiver data with the float64 forward,
* the whole eps / sigma gradient field with the float64 autograd gradient of
  the same discrete problem (cell by cell: relative L2, cosine, max error),
* every t01 directional derivative <grad, p> with the exact float64 value
  (this resolves the cancellation-limited directions of t01 without FD noise).

The float32 rounding level is measured by running the same reference in
float32: DeepGPR passes if its deviation from float64 is within 3x that level
(or 1e-4). The reference's own autograd is checked against float64 central
differences with tiny steps.

Usage: python t01c_reference.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import ref_fdtd2d as ref  # noqa: E402
import t01_fdtd2d_basic as t1  # noqa: E402
import torch  # noqa: E402


def reference_gradients(problem, dtype=torch.float64):
    eps = problem.eps.clone().to(dtype).requires_grad_(True)
    sig = problem.sig.clone().to(dtype).requires_grad_(True)
    data = ref.simulate(problem, eps, sig, dtype=dtype)
    J = gv.misfit(problem, data)
    J.backward()
    return dict(J=float(J), data=data.detach().double(), eps=eps.grad.double(), sig=sig.grad.double())


def field_metrics(a, b):
    return dict(rel_l2=gv.relative_l2(a, b), cosine=gv.cosine(a, b),
                max_abs_err_over_max=float((a - b).abs().max() / b.abs().max()))


def t01_directions(problem, model):
    rng = torch.Generator().manual_seed(gv.SEED + ord(model))
    out = []
    for k_index, kind in enumerate(t1.KINDS):
        pixel = t1.pixel_for(problem, k_index) if kind in ("pixel", "local") else None
        d_eps = gv.directions(problem.eps.shape, problem.pml, kind, rng, pixel=pixel)
        d_sig = gv.directions(problem.eps.shape, problem.pml, kind, rng, pixel=pixel)
        out.append((kind, d_eps, d_sig))
    return out


def main():
    gv.seed_all()
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    fields, directional, sanity, scatter = [], [], [], {}
    cases = [(m, {}) for m in "ABCDEFG"] + [
        ("D", dict(pml=[3, 12, 0, 7])), ("D", dict(pml=0)), ("D", dict(dx=(0.02, 0.015, 0.02))),
        ("D", dict(nt=1200)),
    ]
    for model, kw in cases:
        problem = gv.make_problem_2d(model, **kw)
        label = model + ("" if not kw else " " + ",".join(f"{k}={v}" for k, v in kw.items()))
        r64 = reference_gradients(problem, torch.float64)
        r32 = reference_gradients(problem, torch.float32)
        rounding = dict(data=gv.relative_l2(r32["data"], r64["data"]),
                        eps=gv.relative_l2(r32["eps"], r64["eps"]), sig=gv.relative_l2(r32["sig"], r64["sig"]))
        for path in paths:
            ad = gv.objective_ad(problem, path, problem.eps, problem.sig)
            row = dict(case=label, path=path, tm2d_fast_path=ad.tm2d_fast_path, ref_float32_rounding=rounding,
                       data_rel_l2=gv.relative_l2(ad.data, r64["data"]),
                       J_rel=abs(ad.J - r64["J"]) / abs(r64["J"]),
                       eps=field_metrics(ad.grad_eps, r64["eps"]), sig=field_metrics(ad.grad_sig, r64["sig"]))
            limit = {k: max(3.0 * rounding[k], 1e-4) for k in ("eps", "sig")}
            row["limit"] = limit
            row["verdict"] = "PASS" if (row["eps"]["rel_l2"] <= limit["eps"] and row["sig"]["rel_l2"] <= limit["sig"]) else "FAIL"
            fields.append(row)
            print(f"{label:28s} {path:12s} data {row['data_rel_l2']:.1e} eps {row['eps']['rel_l2']:.2e} "
                  f"(f32 ref {rounding['eps']:.1e}) sig {row['sig']['rel_l2']:.2e} (f32 ref {rounding['sig']:.1e}) "
                  f"cos {row['eps']['cosine']:.10f} -> {row['verdict']}", flush=True)
            if not kw:
                for kind, d_eps, d_sig in t01_directions(problem, model):
                    s_eps, s_sig = float(problem.eps.mean()), float(problem.sig.mean())
                    for name, pe, ps in (("eps", d_eps * s_eps, 0 * d_sig), ("sig", 0 * d_eps, d_sig * s_sig),
                                         ("joint", d_eps * s_eps, d_sig * s_sig)):
                        g_ref = float((r64["eps"] * pe).sum() + (r64["sig"] * ps).sum())
                        g_ad = float((ad.grad_eps * pe).sum() + (ad.grad_sig * ps).sum())
                        cs = float(((r64["eps"].norm() * pe.norm()) ** 2 + (r64["sig"].norm() * ps.norm()) ** 2) ** 0.5)
                        directional.append(dict(model=model, path=path, kind=kind, label=name, g_ad=g_ad, g_ref=g_ref,
                                                rel_err=abs(g_ad - g_ref) / abs(g_ref), rel_err_cs=abs(g_ad - g_ref) / cs,
                                                cancellation=abs(g_ref) / cs))
            if model == "D" and not kw:
                sub = torch.randperm(problem.eps.numel(), generator=torch.Generator().manual_seed(5))[:600]
                scatter[path] = dict(eps_ad=ad.grad_eps.flatten()[sub], eps_ref=r64["eps"].flatten()[sub],
                                     sig_ad=ad.grad_sig.flatten()[sub], sig_ref=r64["sig"].flatten()[sub])
        # Sanity of the reference autograd: float64 central differences, tiny steps.
        if not kw:
            rng = torch.Generator().manual_seed(9)
            for name in ("eps", "sig"):
                d = gv.directions(problem.eps.shape, problem.pml, "random", rng)
                scale = float(problem.eps.mean() if name == "eps" else problem.sig.mean())
                p = d * scale
                g = float(((r64["eps"] if name == "eps" else r64["sig"]) * p).sum())
                vals = []
                for h in (1e-6, -1e-6):
                    with torch.no_grad():
                        e = problem.eps + (h * p if name == "eps" else 0)
                        s = problem.sig + (h * p if name == "sig" else 0)
                        vals.append(float(gv.misfit(problem, ref.simulate(problem, e, s))))
                fd = (vals[0] - vals[1]) / 2e-6
                sanity.append(dict(model=model, label=name, g_autograd=g, g_fd64=fd, rel_err=abs(fd - g) / abs(g)))
                print(f"reference autograd vs float64 FD ({model} {name}): {abs(fd - g) / abs(g):.2e}", flush=True)
    worst = max(directional, key=lambda r: r["rel_err"])
    print("directional derivatives vs float64 reference: worst rel", worst["rel_err"], worst)
    gv.save_json("t01c_reference.json", dict(fields=fields, directional=directional, reference_sanity=sanity,
                                             scatter=scatter))
    print("field verdicts", {v: sum(r["verdict"] == v for r in fields) for v in ("PASS", "FAIL")})


if __name__ == "__main__":
    main()
