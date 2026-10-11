"""Section 9: full FWI / IFWI chains, network-parameter gradients.

Objects under test (imported read-only):
* 2D IFWI ~/桌面/文件/3experiments/surface/landfill_experiment.py: map_eps, map_sig,
  DualSiren / IFWIModel (shared and separate layouts, z = z_L + alpha z_H),
  LogitFWIModel, GridFWIModel(blur), with_air, simulate(), make_misfit(), and
  the training pattern of run_inversion (material gradient of the solver
  first, then one vector-Jacobian product through the network);
* 3D IFWI ~/桌面/文件/3experiments/remote_3d_1010_epsilon_study: SingleSiren3D,
  DualSiren3D (alpha = 1), EpsilonDual (alpha schedule), init_high_freq_to_zero,
  predict_full_grid_chunked (activation checkpoint), replay_material_gradients.

Every chain J(theta) = misfit(FDTD(map(net_theta(x)))) is checked along random
parameter directions (all parameters, low branch only, high branch only, and
d/d alpha) with forward-only finite differences. The network outputs every
physical cell, including the layer DeepGPR replicates into the CPML, so the
FD that is consistent with DeepGPR's fixed-boundary gradient keeps the
extended materials and CPML coefficients of the unperturbed model frozen
(gv.objective_frozen_pml); the plain FD through compute() (total derivative,
CPML rebuilt) is reported separately as the design-level difference.

Precision: networks float32 (and the 3D engine's BF16 autocast, compared
with its FP32 gradient), FDTD float32, misfit float64 where possible.

Usage: python t07_ifwi.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import copy  # noqa: E402

import gv_common as gv  # noqa: E402
import ref_fdtd2d as ref  # noqa: E402
import t06_losses as t6  # noqa: E402
import torch  # noqa: E402

STEPS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)   # wide: SIREN layers are strongly nonlinear


def param_direction(params, gen, scale_floor=1e-2):
    """Random direction with per-tensor scale = RMS of the tensor (scale_floor for zero tensors)."""
    out = []
    for p in params:
        rms = float(p.detach().pow(2).mean().sqrt())
        v = torch.randn(p.shape, generator=gen) * max(rms, scale_floor)
        out.append(v.to(p.device, p.dtype))
    return out


def apply(params, base, direction, h):
    with torch.no_grad():
        for p, b, v in zip(params, base, direction):
            p.copy_(b + h * v)


class Chain:
    """A network/parameterisation on a small DeepGPR problem."""

    def __init__(self, problem, path, fields, misfit=None):
        self.problem, self.path, self.fields = problem, path, fields
        self.misfit = misfit or (lambda obs, sim: gv.misfit(problem, sim))

    def loss_from_data(self, data):
        return self.misfit(self.problem.d_obs.to(data.device).to(data.dtype), data)

    def J_plain(self):
        with torch.no_grad():
            eps, sig = self.fields()
            data = gv.forward_data(self.problem, self.path, eps.detach().double(), sig.detach().double())
            return float(self.loss_from_data(data.double() if data.dtype != torch.float64 else data))

    def J_frozen(self, eps0, sig0):
        with torch.no_grad():
            eps, sig = self.fields()
            pml = self.problem.pml
            eps_full = gv.pad_replicate(eps0.double(), pml) + gv.embed_physical(eps.double() - eps0.double(), pml)
            sig_full = gv.pad_replicate(sig0.double(), pml) + gv.embed_physical(sig.double() - sig0.double(), pml)
            original = self.problem.eps, self.problem.sig
            self.problem.eps, self.problem.sig = eps0.double().cpu(), sig0.double().cpu()
            try:
                _, data = gv.objective_frozen_pml(self.problem, self.path, eps_full, sig_full)
            finally:
                self.problem.eps, self.problem.sig = original
            return float(self.loss_from_data(data.double()))

    def ad(self, params):
        for p in params:
            p.grad = None
        eps, sig = self.fields()
        out = gv.compute(self.problem, self.path, eps, sig, history=True)
        data = out[-1].double()
        J = self.loss_from_data(data)
        J.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        return float(J), grads

    def ad_replay(self, params):
        """run_inversion / replay_material_gradients pattern: solver gradient first, then one VJP."""
        for p in params:
            p.grad = None
        with torch.no_grad():
            eps0, sig0 = self.fields()
        e_leaf = eps0.detach().clone().requires_grad_(True)
        s_leaf = sig0.detach().clone().requires_grad_(True)
        data = gv.compute(self.problem, self.path, e_leaf, s_leaf, history=True)[-1].double()
        J = self.loss_from_data(data)
        J.backward()
        eps, sig = self.fields()
        torch.autograd.backward([eps, sig], [e_leaf.grad, s_leaf.grad])
        return float(J), [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def flat(tensors):
    return torch.cat([t.detach().double().cpu().reshape(-1) for t in tensors])


def reference_chain(problem, module, factory, misfit64, params_names, frozen_steps=(1e-6,), dtype=torch.float64):
    """float64 copy of the network/parameterisation + float64 reference FDTD (2D order 2).

    Returns the exact float64 parameter gradient of the fixed-boundary objective and a
    float64 frozen-CPML central difference along one random direction (sanity)."""
    m64 = copy.deepcopy(module).to(dtype).cpu()
    named = dict(m64.named_parameters())
    params = [named[n] for n in params_names]
    for q in params:
        q.grad = None
    e, sg = factory(m64)
    data = ref.simulate(problem, e, sg, dtype=dtype)
    J = misfit64(problem.d_obs, data.double())
    J.backward()
    grads = [q.grad.detach().clone().double() if q.grad is not None else torch.zeros_like(q).double() for q in params]
    if dtype != torch.float64:
        return float(J), grads, None
    with torch.no_grad():
        e0, s0 = factory(m64)
    gen = torch.Generator().manual_seed(17)
    v = [torch.randn(q.shape, generator=gen, dtype=torch.float64) * max(float(q.detach().pow(2).mean().sqrt()), 1e-2)
         for q in params]
    g = sum(float((a * b).sum()) for a, b in zip(grads, v))
    base = [q.detach().clone() for q in params]
    fds = []
    for h in frozen_steps:
        vals = []
        for sgn in (1, -1):
            with torch.no_grad():
                for q, b, d in zip(params, base, v):
                    q.copy_(b + sgn * h * d)
                ee, ss = factory(m64)
                vals.append(float(misfit64(problem.d_obs, ref.simulate(problem, ee, ss, frozen=(e0, s0)))))
                for q, b in zip(params, base):
                    q.copy_(b)
        fds.append((vals[0] - vals[1]) / (2 * h))
    return float(J), grads, dict(g_ad64=g, g_fd64=fds[0], rel=abs(fds[0] - g) / max(abs(g), 1e-300))


def check_chain(name, chain, params, groups, gen, results, alpha_holder=None):
    with torch.no_grad():
        eps0, sig0 = chain.fields()
    eps0, sig0 = eps0.detach().clone(), sig0.detach().clone()
    if float(sig0.max()) >= 100.0 or float(eps0.min()) < 1.0:
        raise RuntimeError("network output outside the physical range")
    J0, grads = chain.ad(params)
    bad = [i for i, g in enumerate(grads) if not torch.isfinite(g).all()]
    J0r, grads_r = chain.ad_replay(params)
    replay_rel = math.sqrt(sum(float((a - b).pow(2).sum()) for a, b in zip(grads, grads_r)) /
                           max(sum(float(b.pow(2).sum()) for b in grads), 1e-300))
    base = [p.detach().clone() for p in params]
    summary = dict(chain=name, path=chain.path, J0=J0, nonfinite_grad_tensors=bad,
                   zero_grad_tensors=[i for i, g in enumerate(grads) if float(g.abs().max()) == 0.0],
                   grad_norm=math.sqrt(sum(float(g.pow(2).sum()) for g in grads)),
                   replay_vs_direct_rel_l2=replay_rel, checks=[])
    for group_name, mask in groups.items():
        direction = [v if keep else torch.zeros_like(v) for v, keep in zip(param_direction(params, gen), mask)]
        if all(float(v.abs().max()) == 0.0 for v in direction):
            continue
        g = sum(float((gr * v).sum()) for gr, v in zip(grads, direction))
        cs = math.sqrt(sum(float(gr.pow(2).sum()) for gr, keep in zip(grads, mask) if keep)) * \
            math.sqrt(sum(float(v.pow(2).sum()) for v in direction))
        for kind, J_eval in (("frozen", lambda: chain.J_frozen(eps0, sig0)), ("total", chain.J_plain)):
            def J_at(h):
                apply(params, base, direction, h)
                try:
                    return J_eval()
                finally:
                    apply(params, base, direction, 0.0)
            r = gv.taylor_check(J0, g, J_at, STEPS, cs_scale=cs)
            r.update(group=group_name, fd_kind=kind)
            if kind == "total":
                r["note"] = "FD through compute(): CPML rebuilt from the perturbed edge (not DeepGPR's gradient definition)"
            summary["checks"].append(r)
            print(f"{name:40s} {chain.path:12s} {group_name:10s} {kind:6s}: AD={g:.6e} FD={r['g_fd']:.6e} "
                  f"rel={r['rel_err']:.2e} pair={r['pair_rel_err']:.1e} slope={r['fit_slope']:.2f} -> {r['verdict']}",
                  flush=True)
    if alpha_holder is not None:
        # d J / d alpha through the fusion z = z_L + alpha z_H (alpha as a tensor).
        a0 = float(alpha_holder["alpha"])
        alpha_t = torch.tensor(a0, requires_grad=True)
        alpha_holder["alpha"] = alpha_t
        Ja, _ = chain.ad([])
        g_alpha = float(alpha_t.grad) if alpha_t.grad is not None else 0.0
        alpha_holder["alpha"] = a0

        def J_alpha(h):
            alpha_holder["alpha"] = a0 + h
            try:
                return chain.J_frozen(eps0, sig0)
            finally:
                alpha_holder["alpha"] = a0
        r = gv.taylor_check(J0, g_alpha, J_alpha, (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4))
        r.update(group="alpha", fd_kind="frozen")
        summary["checks"].append(r)
        print(f"{name:40s} {chain.path:12s} d/dalpha   frozen: AD={g_alpha:.6e} FD={r['g_fd']:.6e} "
              f"rel={r['rel_err']:.2e} -> {r['verdict']}", flush=True)
    results.append(summary)
    print(f"  replay vs direct gradient rel L2 {replay_rel:.2e}; non-finite tensors {bad}; "
          f"all-zero tensors {summary['zero_grad_tensors']}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------
def problem_2d_landfill(landfill, device_path, nx=24, nt=300):
    """Landfill geometry in miniature: 10 air rows + 30 ground rows, antennas on the first ground row."""
    rows = landfill.AIR + 30
    eps = torch.full((rows, nx), 9.0, dtype=torch.float64)
    eps[: landfill.AIR] = 1.0
    sig = torch.full((rows, nx), 2.0e-3, dtype=torch.float64)
    sig[: landfill.AIR] = 0.0
    src = torch.tensor([[[landfill.ANTENNA_ROW, 6, 0]], [[landfill.ANTENNA_ROW, 17, 0]]], dtype=torch.int32)
    rec = torch.tensor([[[landfill.ANTENNA_ROW, c, 0] for c in range(2, nx - 2, 3)]] * 2, dtype=torch.int32)
    p = gv.Problem(eps=eps, sig=sig, dx=landfill.DX, dt=landfill.DT, nt=nt, src_loc=src, rec_loc=rec,
                   wavelet=gv.ricker(nt, landfill.DT, landfill.FREQ), pml=10, order=2, mode=2)
    deps = 3.0 * gv.gaussian((rows, nx), (landfill.AIR + 15, 12), 3.0)
    deps[: landfill.AIR] = 0
    dsig = 6.0e-3 * gv.gaussian((rows, nx), (landfill.AIR + 20, 8), 3.0)
    dsig[: landfill.AIR] = 0
    gv.attach_observed(p, deps, dsig, path=device_path)
    return p


def problem_3d_engine(shape=(12, 12, 10), nt=120):
    p = gv.make_problem_3d("D", shape=shape, nt=nt, pml=3)
    return p


def main():
    gv.seed_all()
    # PyTorch 1.11 enables TF32 for CUDA float32 matmuls by default (Ampere/Ada GPUs);
    # --no-tf32 runs the same tests with true float32 network arithmetic.
    no_tf32 = "--no-tf32" in sys.argv
    if no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tf32_state = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    print("TF32 state:", tf32_state, flush=True)
    landfill, engine, Misfit_M_SSIM = t6.import_ifwi()
    import epsilon_components as eps_comp   # noqa: E402  (3D epsilon study, alpha-capable dual SIREN)
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    gen = torch.Generator().manual_seed(gv.SEED)
    results = []

    # ---------------- 2D, landfill parameterisations -------------------------------------
    for path in ["cpu", "cuda_fast"] if gv.cuda_available() else ["cpu"]:
        device = gv.path_device(path)
        problem = problem_2d_landfill(landfill, path)
        air = landfill.AIR
        nx = problem.eps.shape[1]
        ground_shape = (problem.eps.shape[0] - air, nx)
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, ground_shape[0]), torch.linspace(-1, 1, nx),
                                            indexing="ij"), -1).reshape(-1, 2)

        def to_full(eg, sg):
            e, s = landfill.with_air(eg, sg)
            return e[..., 0], s[..., 0]

        default64 = lambda obs, data: gv.misfit(problem, data)
        mssim64 = Misfit_M_SSIM(is_3d=False)
        mssim64.g_masks = mssim64.g_masks.double()
        mssim32 = Misfit_M_SSIM(is_3d=False)
        l2 = landfill.make_misfit("l2", str(t6.MSSIM_PARENT))
        cases = []
        logit = landfill.LogitFWIModel(torch.full(ground_shape, 9.0) + 0.3 * torch.rand(ground_shape, generator=gen),
                                       torch.full(ground_shape, 2.0e-3))
        cases.append(("landfill LogitFWIModel", logit, lambda m: to_full(*m.fields(None)),
                      {"eps": ["z_eps"], "sigma": ["z_sig"], "joint": ["z_eps", "z_sig"]}, None, None, None))
        grid = landfill.GridFWIModel(torch.full(ground_shape, 9.0), torch.full(ground_shape, 2.0e-3), blur=1.5)
        cases.append(("landfill GridFWIModel blur=1.5", grid, lambda m: to_full(*m.fields(None)),
                      {"eps": ["eps"], "sigma": ["sig"], "joint": ["eps", "sig"]}, None, None, None))
        for layout in ("shared", "separate"):
            torch.manual_seed(gv.SEED + (0 if layout == "shared" else 1))
            model = landfill.IFWIModel(nx, torch.device("cpu"), layout=layout)
            model.coords = coords.clone()
            model.shape = ground_shape
            with torch.no_grad():   # keep the start near the observed model's background
                for net in ([model.net] if layout == "shared" else [model.net_eps, model.net_sig]):
                    for branch in (net.low, net.high):
                        branch[-1].weight.mul_(0.3)
            holder = {"alpha": 0.7}
            names = [n for n, _ in model.named_parameters()]
            cases.append((f"landfill IFWIModel {layout} (alpha 0.7)", model,
                          lambda m, holder=holder: to_full(*m.fields(holder["alpha"])),
                          {"all": names, "low": [n for n in names if ".low." in n],
                           "high": [n for n in names if ".high." in n]}, holder, None, None))
        torch.manual_seed(gv.SEED + 2)
        homog = landfill.IFWIModel(nx, torch.device("cpu"), layout="shared")
        homog.coords = coords.clone()
        homog.shape = ground_shape
        homog.set_homogeneous("eps", 9.0)
        homog.set_homogeneous("sig", 2.0e-3)
        names = [n for n, _ in homog.named_parameters()]
        for alpha in (0.0, 1.0):
            cases.append((f"landfill IFWIModel homogeneous start alpha={alpha}", homog,
                          lambda m, alpha=alpha: to_full(*m.fields(alpha)),
                          {"all": names, "high": [n for n in names if ".high." in n]}, None, None, None))
        torch.manual_seed(gv.SEED + 4)
        model2 = landfill.IFWIModel(nx, torch.device("cpu"), layout="shared")
        model2.coords = coords.clone()
        model2.shape = ground_shape
        with torch.no_grad():
            for branch in (model2.net.low, model2.net.high):
                branch[-1].weight.mul_(0.3)
        names = [n for n, _ in model2.named_parameters()]
        cases.append(("landfill IFWI + landfill L2", model2, lambda m: to_full(*m.fields(0.7)), {"all": names},
                      None, l2, l2))
        cases.append(("landfill IFWI + M-SSIM", model2, lambda m: to_full(*m.fields(0.7)), {"all": names}, None,
                      lambda obs, sim: mssim32.forward(obs.float(), sim.float()).double(),
                      lambda obs, sim: mssim64.forward(obs, sim)))
        for name, module, factory, groups_by_name, holder, misfit32, misfit64 in cases:
            module.to(device)
            named = list(module.named_parameters())
            params = [q for _, q in named]
            groups = {g: [n in members for n, _ in named] for g, members in groups_by_name.items()}
            chain = Chain(problem, path, lambda module=module, factory=factory: factory(module), misfit=misfit32)
            summary = check_chain(name, chain, params, groups, gen, results, alpha_holder=holder)
            summary["named_zero_grad"] = [named[i][0] for i in summary["zero_grad_tensors"]]
            if summary["named_zero_grad"]:
                print("   zero-gradient tensors:", summary["named_zero_grad"], flush=True)
            # float64 reference chain (exact fixed-boundary gradient).
            _, g32 = chain.ad(params)
            module.cpu()
            Jr, g64, sanity = reference_chain(problem, module, factory, misfit64 or default64, [n for n, _ in named])
            module.to(device)
            _, g_ref32, _ = reference_chain(problem, module, factory, misfit64 or default64, [n for n, _ in named],
                                            dtype=torch.float32)
            level = gv.relative_l2(flat(g_ref32), flat(g64))
            rel = gv.relative_l2(flat(g32), flat(g64))
            cos = gv.cosine(flat(g32), flat(g64))
            limit = max(3.0 * level, 1e-4)
            summary["vs_reference"] = dict(rel_l2=rel, cosine=cos, float32_level=level, limit=limit,
                                           reference_fd_sanity=sanity,
                                           verdict="PASS" if rel <= limit and sanity["rel"] <= 1e-5 else "FAIL")
            print(f"   vs float64 reference chain: grad rel L2 {rel:.2e} (float32 reference level {level:.1e}) "
                  f"cosine {cos:.10f}; reference frozen FD {sanity['rel']:.1e} -> {summary['vs_reference']['verdict']}",
                  flush=True)
            module.cpu()

    # ---------------- 3D engine networks ---------------------------------------------------
    for path in ["cpu"] + (["cuda_general"] if gv.cuda_available() else []):
        device = gv.path_device(path)
        problem = problem_3d_engine()
        shape = tuple(problem.eps.shape)
        coords = engine.get_mgrid_3d(shape).to(device)
        sig_range = (1e-3, 0.05)
        nets = []
        torch.manual_seed(gv.SEED + 3)
        single = engine.SingleSiren3D(hidden_features=48, omega_0=30.0, er_range=(1.0, 10.0), sig_range=sig_range).to(device)
        nets.append(("3D SingleSiren3D (width 48)", single, None))
        dual = engine.DualSiren3D(hidden_features=48, er_range=(1.0, 10.0), sig_range=sig_range).to(device)
        nets.append(("3D DualSiren3D (width 48, alpha 1)", dual, None))
        epsdual = eps_comp.EpsilonDual(sig_range).to(device)   # width 600, high branch output zeroed
        nets.append(("3D EpsilonDual (width 600, zero high head)", epsdual, 0.5))
        for name, net, alpha in nets:
            holder = {"alpha": alpha}
            kind = "single" if isinstance(net, engine.SingleSiren3D) else "dual"

            def fields(net=net, holder=holder, kind=kind):
                if kind == "single":
                    e, s = net.get_params(coords)
                elif holder["alpha"] is None:
                    e, s = net.get_params(coords, alpha=1.0)
                else:
                    e, s = net.get_params(coords, holder["alpha"])
                return e.reshape(shape), s.reshape(shape)

            chain = Chain(problem, path, fields)
            named = list(net.named_parameters())
            params = [p for _, p in named]
            groups = {"all": [1] * len(params)}
            if kind == "dual":
                groups["low"] = [n.startswith("net_low") for n, _ in named]
                groups["high"] = [n.startswith("net_high") for n, _ in named]
            s = check_chain(name, chain, params, groups, gen, results,
                            alpha_holder=holder if alpha is not None else None)
            s["named_zero_grad"] = [named[i][0] for i in s["zero_grad_tensors"]]
            print("   zero-gradient tensors:", s["named_zero_grad"], flush=True)
            # Network VJP in float64 (exact): the replay of the solver's material gradient
            # through the network, float32 vs float64 network, and float64 FD of
            # phi(theta) = <eps(theta), g_eps> + <sigma(theta), g_sigma>.
            with torch.no_grad():
                e0, s0 = fields()
            e_leaf = e0.detach().clone().requires_grad_(True)
            s_leaf = s0.detach().clone().requires_grad_(True)
            gv.misfit(problem, gv.compute(problem, path, e_leaf, s_leaf, history=True)[-1]).backward()
            g_mat = (e_leaf.grad.detach(), s_leaf.grad.detach())
            for q in params:
                q.grad = None
            torch.autograd.backward(list(fields()), list(g_mat))
            vjp32 = [q.grad.detach().clone() if q.grad is not None else torch.zeros_like(q) for q in params]
            net64 = copy.deepcopy(net).double().cpu()
            coords64 = coords.double().cpu()
            p64 = [q for _, q in net64.named_parameters()]

            def fields64(alpha=alpha):
                if kind == "single":
                    e, sg = net64.get_params(coords64)
                elif alpha is None:
                    e, sg = net64.get_params(coords64, alpha=1.0)
                else:
                    e, sg = net64.get_params(coords64, alpha)
                return e.reshape(shape), sg.reshape(shape)

            gm64 = [t.double().cpu() for t in g_mat]
            torch.autograd.backward(list(fields64()), gm64)
            vjp64 = [q.grad.detach().clone() if q.grad is not None else torch.zeros_like(q) for q in p64]
            v = [torch.randn(q.shape, generator=gen, dtype=torch.float64) * max(float(q.detach().pow(2).mean().sqrt()), 1e-2)
                 for q in p64]
            g_lin = sum(float((a * b).sum()) for a, b in zip(vjp64, v))
            base = [q.detach().clone() for q in p64]

            def phi(h):
                with torch.no_grad():
                    for q, b, d in zip(p64, base, v):
                        q.copy_(b + h * d)
                    e, sg = fields64()
                    out = float((e.double() * gm64[0]).sum() + (sg.double() * gm64[1]).sum())
                    for q, b in zip(p64, base):
                        q.copy_(b)
                    return out
            # _map_logits casts the logits to float32 (by design, for AMP), so phi is float32-
            # quantised: steps from 1e-2 down, not 1e-7.
            lin = gv.taylor_check(phi(0.0), g_lin, phi, (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4))
            s["network_vjp"] = dict(float32_vs_float64_rel_l2=gv.relative_l2(flat(vjp32), flat(vjp64)),
                                    fd_check=lin, verdict=lin["verdict"])
            print(f"   network VJP: float32 vs float64 {s['network_vjp']['float32_vs_float64_rel_l2']:.2e}; "
                  f"float64 FD: {gv.summarise(lin)}", flush=True)
            # Activation checkpoint (predict_full_grid_chunked) on/off, FP32.
            engine.USE_BF16_AUTOCAST = False
            grads = {}
            for use_ckpt in (False, True):
                for p in params:
                    p.grad = None
                e, sg = engine.predict_full_grid_chunked(net, coords, kind, 1.0 if alpha is None else alpha,
                                                         batch_size=300, use_checkpoint=use_ckpt)
                data = gv.compute(problem, path, e.reshape(shape), sg.reshape(shape), history=True)[-1]
                gv.misfit(problem, data).backward()
                grads[use_ckpt] = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            ck_rel = math.sqrt(sum(float((a - b).pow(2).sum()) for a, b in zip(grads[True], grads[False])) /
                               max(sum(float(b.pow(2).sum()) for b in grads[False]), 1e-300))
            ck_bitwise = all(torch.equal(a, b) for a, b in zip(grads[True], grads[False]))
            s["activation_checkpoint_rel_l2"] = ck_rel
            s["activation_checkpoint_bitwise"] = ck_bitwise
            print(f"   activation checkpoint on/off: rel {ck_rel:.2e} bitwise {ck_bitwise}", flush=True)
            # BF16 autocast (CUDA only, as trained) vs FP32 gradient.
            if device.type == "cuda" and alpha is None:
                engine.USE_BF16_AUTOCAST = True
                for p in params:
                    p.grad = None
                e, sg = engine.predict_full_grid_chunked(net, coords, kind, 1.0, batch_size=300, use_checkpoint=False)
                data = gv.compute(problem, path, e.reshape(shape).float(), sg.reshape(shape).float(), history=True)[-1]
                gv.misfit(problem, data).backward()
                g16 = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                rel16 = math.sqrt(sum(float((a - b).pow(2).sum()) for a, b in zip(g16, grads[False])) /
                                  max(sum(float(b.pow(2).sum()) for b in grads[False]), 1e-300))
                cos16 = sum(float((a * b).sum()) for a, b in zip(g16, grads[False])) / max(
                    math.sqrt(sum(float(a.pow(2).sum()) for a in g16) * sum(float(b.pow(2).sum()) for b in grads[False])), 1e-300)
                s["bf16_autocast_vs_fp32_rel_l2"] = rel16
                s["bf16_autocast_vs_fp32_cosine"] = cos16
                print(f"   BF16 autocast vs FP32 gradient: rel {rel16:.2e} cosine {cos16:.6f}", flush=True)
                engine.USE_BF16_AUTOCAST = False
    gv.save_json("t07_ifwi_no_tf32.json" if no_tf32 else "t07_ifwi.json",
                 dict(results=results, tf32=tf32_state,
                      modules=dict(landfill=str(t6.LANDFILL), engine=str(t6.ENGINE_DIR))))
    counts = {}
    for s in results:
        for c in s["checks"]:
            key = (c["fd_kind"], c["verdict"])
            counts[key] = counts.get(key, 0) + 1
    print("verdicts", counts)


if __name__ == "__main__":
    main()
