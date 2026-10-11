"""Minimal reproduction: CPML corrections on cells without a base update.

The CPML correction loops update field components at positions where the
base FDTD update is skipped (outer boundary lines/planes, the high-side halo,
and in 2D the TE components Ex, Ey, Hz that are never base-updated). These
positions form closed CPML-only subsystems. From zero fields they stay zero
(receiver data unaffected), but
  1. the forward grows without bound from non-zero initial states, and
  2. the exact adjoint is driven there by the interior, so the gradients with
     respect to the initial E / H / CPML states grow exponentially backward in
     time and become inf / NaN for long runs (all three paths).
Material and source gradients stay finite (the subsystems do not feed back
into the interior adjoint).

Usage: python t09_cpml_degenerate.py TAG     (TAG = before_fix / after_fix)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402
import DeepGPR  # noqa: E402
from DeepGPR.solver.autograd import STATE_NAMES  # noqa: E402


def initial_states(problem, path):
    device = gv.path_device(path)
    E, H, PML = DeepGPR.checkpoint_initial_field(
        device=device, dx=problem.dx, dt=problem.dt, source_amplitudes=problem.wavelet.to(device),
        source_location=problem.src_loc.to(device), receiver_location=problem.rec_loc.to(device),
        er=problem.eps.float(), se=problem.sig.float(), pmlthick=problem.pml, fdtd_order=problem.order)
    return [t for t in (*E, *H, *PML)]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    paths2 = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    paths3 = ["cpu"] + (["cuda_general"] if gv.cuda_available() else [])
    rows = []
    for dims, paths in (("2d", paths2), ("3d", paths3)):
        for path in paths:
            for nt in (100, 200, 400, 800, 1200, 1600):
                problem = (gv.make_problem_2d("D", shape=(16, 16), nt=nt, pml=4) if dims == "2d"
                           else gv.make_problem_3d("D", shape=(10, 10, 8), nt=nt, pml=3))
                ad = gv.objective_ad(problem, path, problem.eps, problem.sig, wrt=("eps", "sig", "states"),
                                     states=initial_states(problem, path))
                per_state = {}
                for name, g in zip(STATE_NAMES, ad.grad_states):
                    if g is not None and g.numel():
                        finite = bool(torch.isfinite(g).all())
                        per_state[name] = float(g.abs().max()) if finite else float("nan")
                worst = max(per_state.items(), key=lambda kv: (kv[1] != kv[1], kv[1]))
                rows.append(dict(test="state_gradient_growth", dims=dims, path=path, nt=nt,
                                 grad_eps_finite=bool(torch.isfinite(ad.grad_eps).all()),
                                 grad_eps_max=float(ad.grad_eps.abs().max()),
                                 state_grad_max=per_state, worst_state=worst[0], worst_value=worst[1]))
                print(f"{tag} {dims} {path:12s} nt={nt:5d} material grad finite {rows[-1]['grad_eps_finite']} "
                      f"| largest initial-state gradient {worst[0]} = {worst[1]:.2e}", flush=True)
            # Forward from small random initial states (zero source).
            problem = (gv.make_problem_2d("D", shape=(16, 16), nt=1000, pml=4) if dims == "2d"
                       else gv.make_problem_3d("D", shape=(10, 10, 8), nt=1000, pml=3))
            gen = torch.Generator().manual_seed(0)
            states = [torch.randn(t.shape, generator=gen).to(t.device) * 1e-3 for t in initial_states(problem, path)]
            if path == "cuda_fast":
                from DeepGPR.solver.autograd import TM2D_INACTIVE_STATE_INDICES
                for i in TM2D_INACTIVE_STATE_INDICES:
                    states[i] = torch.zeros_like(states[i])
            for nt in (100, 300, 1000):
                st = [s.clone() for s in states]
                out = gv.compute(problem, path, problem.eps, problem.sig,
                                 source=torch.zeros((1, nt, 1)), states=st)
                fields = [*out[1], *out[2], *out[3]]
                mx = max((float(f.abs().max()) if bool(torch.isfinite(f).all()) else float("inf")) for f in fields if f.numel())
                rows.append(dict(test="forward_from_random_states", dims=dims, path=path, nt=nt, max_field=mx))
                print(f"{tag} {dims} {path:12s} forward from 1e-3 random states, nt={nt}: max |field| = {mx:.2e}", flush=True)
    gv.save_json(f"t09_cpml_degenerate_{tag}.json", dict(tag=tag, rows=rows))


if __name__ == "__main__":
    main()
