"""Initial-state gradients and multi-step operator transpose (dot-product) tests.

Part A, dot-product tests <A x, y> = <x, A^T y> of the full state map over n
time steps (both half steps, CPML on every listed face, heterogeneous
eps/sigma/mu, zero source): A maps the 30 initial state tensors (E, H, 24
CPML arrays) to the 30 final states and the receiver data; A^T y is the native
adjoint (gradients of <A x, y> with respect to the initial states).
n = 1 reproduces the existing single-step tests; n up to 300 is new.

Part B, directional derivatives of the misfit with respect to the initial
E / H / CPML states of a second time segment (states taken from a first
segment), against central differences of forward-only runs.

Usage: python t02_states_dot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402
import DeepGPR  # noqa: E402

# Ex, Ey, Hz and the CPML arrays that only couple them (see TM2D_INACTIVE_STATE_INDICES).
from DeepGPR.solver.autograd import TM2D_INACTIVE_STATE_INDICES  # noqa: E402

THRESHOLD = {"cpu": 3.0e-5, "cuda_fast": 2.0e-4, "cuda_general": 2.0e-4}   # tests/test_discrete_adjoint.py


def materials(shape):
    grids = torch.meshgrid(*[torch.linspace(0.0, 1.0, n, dtype=torch.float64) for n in shape], indexing="ij")
    eps = 3.5 + 0.7 * grids[0] + 0.4 * grids[1] + (0.2 * grids[2] if len(shape) == 3 else 0.0)
    sig = 1.0e-2 + 1.0e-2 * sum(grids) / len(grids)
    mu = 1.0 + 0.15 * grids[0] + 0.05 * grids[1]
    eps = eps + 1.5 * gv.gaussian(shape, [0.4 * n for n in shape], 0.15 * shape[0])
    return eps.float(), sig.float(), mu.float()


def problem_for(shape, pml, order, nt, n_rx=3):
    dims = len(shape)
    eps, sig, mu = materials(shape)
    if dims == 2:
        loc = torch.tensor([[[shape[0] // 2, shape[1] // 2, 0]]], dtype=torch.int32)
        rec = torch.tensor([[[2 + i, 2 + 3 * i, 0] for i in range(n_rx)]], dtype=torch.int32)
    else:
        loc = torch.tensor([[[shape[0] // 2, shape[1] // 2, shape[2] // 2]]], dtype=torch.int32)
        rec = torch.tensor([[[2 + i, 2 + 2 * i, 3] for i in range(n_rx)]], dtype=torch.int32)
    return dict(eps=eps, sig=sig, mu=mu, loc=loc, rec=rec, pml=pml, order=order, nt=nt,
                dx=(0.020, 0.017, 0.013), dt=1.0e-11, mode=2 if dims == 2 else 3)


def run(prob, path, states, source):
    device = gv.path_device(path)
    with gv.solver_path(path):
        return DeepGPR.compute(
            device=device, dx=prob["dx"], dt=prob["dt"], source_amplitudes=source.to(device),
            source_location=prob["loc"].to(device), receiver_location=prob["rec"].to(device),
            eps_r=prob["eps"].to(device), sigma=prob["sig"].to(device), mu_r=prob["mu"].to(device),
            E=tuple(states[:3]), H=tuple(states[3:6]), PML=tuple(states[6:]),
            pmlthick=prob["pml"], fdtd_order=prob["order"], mode=prob["mode"],
            model_gradient_sampling_interval=1, wavefield_storage_dtype=torch.float32,
        )


def zero_states(prob, path):
    device = gv.path_device(path)
    source = torch.zeros((1, 1, 1))
    return [t.to(device) for t in sum(map(list, DeepGPR.checkpoint_initial_field(
        device=device, dx=prob["dx"], dt=prob["dt"], source_amplitudes=source,
        source_location=prob["loc"].to(device), receiver_location=prob["rec"].to(device),
        er=prob["eps"].to(device), se=prob["sig"].to(device), mr=prob["mu"].to(device),
        pmlthick=prob["pml"], fdtd_order=prob["order"])), [])]


def dot_test(prob, path, seed):
    gen = torch.Generator().manual_seed(seed)
    base = zero_states(prob, path)
    x = [torch.randn(t.shape, generator=gen).to(t.device) * 1e-3 for t in base]
    active_only = path == "cuda_fast"
    if active_only:
        for i in TM2D_INACTIVE_STATE_INDICES:
            x[i] = torch.zeros_like(x[i])
    leaves = [t.clone().requires_grad_(True) for t in x]
    out = run(prob, path, leaves, torch.zeros((1, prob["nt"], 1)))
    flag = bool(getattr(out[-1].grad_fn.config, "tm2d_fast_path", False))
    outputs = [*out[1], *out[2], *out[3], out[-1]]
    y = [torch.randn(t.shape, generator=gen).to(t.device) for t in outputs]
    lhs = sum((o.double() * w.double()).sum() for o, w in zip(outputs, y))
    lhs.backward()
    rhs = sum((xi.double() * li.grad.double()).sum()
              for i, (xi, li) in enumerate(zip(x, leaves)) if not (active_only and i in TM2D_INACTIVE_STATE_INDICES))
    inactive_max = max(float(leaves[i].grad.abs().max()) if leaves[i].numel() else 0.0
                       for i in TM2D_INACTIVE_STATE_INDICES) if active_only else None
    lhs, rhs = float(lhs.detach()), float(rhs)
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
    return dict(path=path, nt=prob["nt"], order=prob["order"], pml=prob["pml"], dims=len(prob["eps"].shape),
                lhs=lhs, rhs=rhs, rel_err=rel, threshold=THRESHOLD[path], tm2d_fast_path=flag,
                inactive_cotangent_max=inactive_max,
                verdict="PASS" if rel < THRESHOLD[path] else "FAIL")


def state_fd(path, dims, seed, results):
    """Misfit derivative w.r.t. the initial E/H/CPML state of a second segment."""
    if dims == 2:
        problem = gv.make_problem_2d("D", shape=(32, 32), nt=300)
    else:
        problem = gv.make_problem_3d("D", shape=(14, 14, 10), nt=160)
    device = gv.path_device(path)
    split = problem.nt // 3
    # States after the first segment (forward only).
    first = gv.compute(problem, path, problem.eps, problem.sig, source=problem.wavelet[:, :split])
    states0 = [t.detach().clone() for t in (*first[1], *first[2], *first[3])]
    second = gv.Problem(**{**problem.__dict__})
    second.wavelet = problem.wavelet[:, split:].contiguous()
    second.nt = problem.nt - split
    second.d_obs = problem.d_obs[:, split:].contiguous()
    gen = torch.Generator().manual_seed(seed)
    groups = {"E": range(0, 3), "H": range(3, 6), "CPML": range(6, 30), "all": range(0, 30)}
    ad = gv.objective_ad(second, path, second.eps, second.sig, wrt=("states",), states=states0)
    for name, indices in groups.items():
        q = [torch.zeros_like(t) for t in states0]
        for i in indices:
            if path == "cuda_fast" and i in TM2D_INACTIVE_STATE_INDICES:
                continue
            if states0[i].numel():
                scale = float(states0[i].abs().max()) or 1.0
                q[i] = (torch.randn(states0[i].shape, generator=gen) * scale).to(device)
        if all(float(t.abs().sum()) == 0.0 for t in q):
            continue
        g = sum(float((gi * qi.double().cpu()).sum()) for gi, qi in zip(ad.grad_states, q) if gi is not None)
        J_at = lambda h: gv.objective_forward(second, path, second.eps, second.sig,
                                              states=[s + h * qi for s, qi in zip(states0, q)])
        # The data are affine in the initial state, so J is exactly quadratic in h:
        # the central difference has no truncation error and large steps beat the noise.
        r = gv.taylor_check(ad.J, g, J_at, (1e2, 3e1, 1e1, 3.0, 1.0, 3e-1, 1e-1, 3e-2))
        r.update(path=path, dims=dims, label=f"state:{name}", kind="random", tm2d_fast_path=ad.tm2d_fast_path)
        results.append(r)
        print(gv.summarise(r), flush=True)
    if path == "cuda_fast":
        inactive = max(float(ad.grad_states[i].abs().max()) for i in TM2D_INACTIVE_STATE_INDICES
                       if ad.grad_states[i].numel())
        results.append(dict(path=path, dims=dims, label="state:inactive_cotangents_fast_path",
                            max_abs=inactive, verdict="PASS" if inactive == 0.0 else "FAIL"))
        print("fast path inactive cotangents max", inactive)


def main():
    gv.seed_all()
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    dots = []
    shape2, pml2 = (20, 22), [3, 5, 4, 2]
    for nt in (1, 2, 5, 20, 100, 300):
        prob = problem_for(shape2, pml2, 4, nt)
        for path in paths:
            r = dot_test(prob, path, 4000 + nt)
            dots.append(r)
            print("dot 2D", path, nt, "%.2e" % r["rel_err"], r["verdict"], "fast" if r["tm2d_fast_path"] else "", flush=True)
    for order in (2, 8):
        for pml in ([3, 5, 4, 2], [0, 6, 2, 0], 0):
            prob = problem_for(shape2, pml, order, 50)
            for path in paths:
                r = dot_test(prob, path, 5000 + order)
                dots.append(r)
                print("dot 2D order", order, pml, path, "%.2e" % r["rel_err"], r["verdict"], flush=True)
    shape3, pml3 = (10, 11, 9), [2, 1, 3, 2, 1, 2]
    for nt in (1, 5, 20, 100):
        prob = problem_for(shape3, pml3, 4, nt)
        for path in ["cpu"] + (["cuda_general"] if gv.cuda_available() else []):
            r = dot_test(prob, path, 6000 + nt)
            dots.append(r)
            print("dot 3D", path, nt, "%.2e" % r["rel_err"], r["verdict"], flush=True)
    states = []
    for path in paths:
        state_fd(path, 2, 7000, states)
    for path in ["cpu"] + (["cuda_general"] if gv.cuda_available() else []):
        state_fd(path, 3, 7100, states)
    gv.save_json("t02_states_dot.json", dict(dot_tests=dots, state_fd=states,
                                             thresholds=THRESHOLD))
    print("dot verdicts", {v: sum(r["verdict"] == v for r in dots) for v in ("PASS", "FAIL")})
    print("state verdicts", {v: sum(r["verdict"] == v for r in states) for v in ("PASS", "FAIL", "INCONCLUSIVE")})


if __name__ == "__main__":
    main()
