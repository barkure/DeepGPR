"""Fixed-boundary gradient vs total derivative, exactly (float64 reference).

DeepGPR rebuilds the CPML on every call by replicating the current edge of the
physical model and computing the CPML coefficients from the boundary-plane
averages, but its gradient treats both as fixed (docs/API.md 4.2). With the
float64 reference (ref_fdtd2d.py) both derivatives are available exactly:

* fixed  = autograd through the default reference (DeepGPR's definition),
* total  = autograd through replicate padding with summed halo sensitivities
           and differentiable CPML coefficients.

Checks: float64 central differences (h = 1e-6) of the frozen forward equal the
fixed gradient, and of the plain forward equal the total gradient (so each is
the exact derivative of its forward); DeepGPR's AD equals the fixed gradient
at float32 level; the fixed - total difference is located on the edge layer
only, and its size is reported per model (interior anomaly, anomaly at the
boundary, air layer touching the top face as in the landfill geometry).

Usage: python t04b_boundary_reference.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import ref_fdtd2d as ref  # noqa: E402
import torch  # noqa: E402


def grads(problem, **mode):
    eps = problem.eps.clone().requires_grad_(True)
    sig = problem.sig.clone().requires_grad_(True)
    J = gv.misfit(problem, ref.simulate(problem, eps, sig, **mode))
    J.backward()
    return float(J), eps.grad.clone(), sig.grad.clone()


def landfill_like():
    p = gv.make_problem_2d("D", nt=400)
    p.eps[:8] = 1.0          # air rows at the top face (they touch the x0 CPML)
    p.sig[:8] = 0.0
    p.src_loc[:, 0, 0] = 8
    p.rec_loc[:, : p.rec_loc.shape[1] // 2, 0] = 8
    gv.attach_observed(p, *gv.observed_perturbation_2d((48, 48)))
    return p


def main():
    gv.seed_all()
    cases = {"D (interior anomalies)": gv.make_problem_2d("D", nt=400),
             "G (anomalies at the boundary)": gv.make_problem_2d("G", nt=400),
             "air layer on the x0 face": landfill_like(),
             "G, CPML 20": gv.make_problem_2d("G", nt=400, pml=20),
             "G, CPML 5": gv.make_problem_2d("G", nt=400, pml=5)}
    rows = []
    for name, problem in cases.items():
        J, ge_f, gs_f = grads(problem)
        _, ge_t, gs_t = grads(problem, total=True)
        edge = 1.0 - gv.edge_mask(problem.eps.shape, problem.pml)
        interior = 1.0 - edge
        row = dict(case=name, J=J)
        for label, gf, gt in (("eps", ge_f, ge_t), ("sig", gs_f, gs_t)):
            diff = gf - gt
            row[label] = dict(
                rel_l2_fixed_vs_total=gv.relative_l2(gf, gt), cosine=gv.cosine(gf, gt),
                interior_max_abs_diff=float((diff * interior).abs().max()),
                edge_rel_l2=float((diff * edge).norm() / (gt * edge).norm()),
                edge_sign_flips=int(((gf * gt < 0) & (edge > 0)).sum()), edge_cells=int(edge.sum()),
                edge_share_of_total_norm=float((gt * edge).norm() / gt.norm()))
        # Exactness of both derivatives: float64 FD with tiny steps.
        rng = torch.Generator().manual_seed(3)
        d = torch.randn(problem.eps.shape, generator=rng, dtype=torch.float64)
        d = d / d.abs().max()
        for label in ("eps", "sig"):
            scale = float(problem.eps.mean() if label == "eps" else problem.sig.mean())
            pdir = d * scale
            h = 1e-6
            vals = {}
            for kind in ("frozen", "plain"):
                v = []
                for s in (1, -1):
                    with torch.no_grad():
                        e = problem.eps + (s * h * pdir if label == "eps" else 0)
                        sg = problem.sig + (s * h * pdir if label == "sig" else 0)
                        # "plain" = the forward that rebuilds the CPML from the current edge; the
                        # total=True reference computes its CPML coefficients in float64 (DeepGPR's
                        # build_pml_coeffs works in float32, which would quantise a 1e-6 step).
                        mode = dict(frozen=(problem.eps, problem.sig)) if kind == "frozen" else dict(total=True)
                        v.append(float(gv.misfit(problem, ref.simulate(problem, e, sg, **mode))))
                vals[kind] = (v[0] - v[1]) / (2 * h)
            gfix = float(((ge_f if label == "eps" else gs_f) * pdir).sum())
            gtot = float(((ge_t if label == "eps" else gs_t) * pdir).sum())
            row[label].update(fd_frozen=vals["frozen"], ad_fixed=gfix,
                              frozen_fd_vs_fixed_rel=abs(vals["frozen"] - gfix) / abs(gfix),
                              fd_plain=vals["plain"], ad_total=gtot,
                              plain_fd_vs_total_rel=abs(vals["plain"] - gtot) / abs(gtot),
                              fixed_vs_total_directional_rel=abs(gfix - gtot) / abs(gtot))
        # DeepGPR vs fixed reference.
        ad = gv.objective_ad(problem, "cpu", problem.eps, problem.sig)
        row["deepgpr_cpu_vs_fixed"] = dict(eps=gv.relative_l2(ad.grad_eps, ge_f), sig=gv.relative_l2(ad.grad_sig, gs_f))
        rows.append(row)
        print(name, {k: (v if not isinstance(v, dict) else {kk: (f"{vv:.2e}" if isinstance(vv, float) else vv)
                                                              for kk, vv in v.items()}) for k, v in row.items()},
              flush=True)
    gv.save_json("t04b_boundary_reference.json", dict(results=rows))


if __name__ == "__main__":
    main()
