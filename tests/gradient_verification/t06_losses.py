"""Section 8: gradients of the misfits with respect to the simulated waveform (no FDTD).

Tested objects (imported read-only from where they live):

* DeepGPR:   hilbert_transform (envelope), apply_filter (low-pass FIR), TVRegularization;
* 2D IFWI (~/桌面/文件/3experiments/surface/landfill_experiment.py):
             make_misfit("l2"), make_misfit("l1"), make_misfit("mssim"), NoiseAwareMSSIM,
             bandlimit (Kaiser FIR, reflect padding), mute_mask;
* 3D IFWI (~/桌面/文件/3experiments/remote_3d_1010_epsilon_study/baseline_engine.py):
             build_loss("l2") = nn.MSELoss, the M-SSIM criterion Misfit_M_SSIM(is_3d=True)
             fed (B, T, R) data, and loss_forward (autocast off, float32);
* M-SSIM:    ~/桌面/MSSIM/MS_SSIM.py (sha256 identical to the copy in the 3D study).

OT / Wasserstein, cross-correlation and convolution-type misfits are not part
of either IFWI code (MSSIM/__init__.py has its Wasserstein import commented
out and the source file does not exist); they are reported as not tested.

Each misfit: torch.autograd.gradcheck in float64 on several input families
(random, Ricker gathers, time-shifted, amplitude-scaled, low-amplitude) and an
independent float64 central-difference / Taylor check of directional
derivatives (forward evaluations only). Special inputs probe the
non-differentiable points (L1 at sim == obs, per-trace max ties, zero traces).

Usage: python t06_losses.py
"""
import contextlib
import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402
import DeepGPR  # noqa: E402

HOME = Path.home()
LANDFILL = HOME / "桌面" / "文件" / "3experiments" / "surface" / "landfill_experiment.py"
ENGINE_DIR = HOME / "桌面" / "文件" / "3experiments" / "remote_3d_1010_epsilon_study"
MSSIM_PARENT = HOME / "桌面"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_ifwi():
    sys.dont_write_bytecode = True
    landfill = load_module("landfill_experiment_under_test", LANDFILL)
    landfill.DG = DeepGPR                      # the module sets DG in import_deepgpr()
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))
    import baseline_engine as engine           # noqa: E402
    sys.path.insert(0, str(MSSIM_PARENT))
    from MSSIM.MS_SSIM import Misfit_M_SSIM    # noqa: E402
    return landfill, engine, Misfit_M_SSIM


@contextlib.contextmanager
def default_dtype(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def gathers(kind, gen, shape=(2, 64, 6)):
    B, T, R = shape
    t = torch.arange(T, dtype=torch.float64)
    if kind == "random":
        sim = torch.randn(shape, generator=gen, dtype=torch.float64)
        obs = torch.randn(shape, generator=gen, dtype=torch.float64)
        return sim, obs
    # Ricker traces with receiver-dependent delay (a small synthetic radargram).
    def ricker(delay, f=0.08):
        a = (math.pi * f * (t - delay)) ** 2
        return (1 - 2 * a) * torch.exp(-a)
    base = torch.stack([torch.stack([ricker(18 + 3 * r + 2 * b) for r in range(R)], 1) for b in range(B)])
    noise = 0.02 * torch.randn(shape, generator=gen, dtype=torch.float64)
    if kind == "ricker":
        return base + noise, 0.8 * base
    if kind == "shifted":
        obs = torch.stack([torch.stack([ricker(21 + 3 * r + 2 * b) for r in range(R)], 1) for b in range(B)])
        return base + noise, obs
    if kind == "scaled":
        return 2.5 * base + noise, base
    if kind == "low_amplitude":
        return 1e-6 * (base + noise), 0.8e-6 * base
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def gradcheck(fn, sim):
    """gradcheck of u -> fn(a u) at u = sim / a, a = max|sim|.

    The same derivative (times a), evaluated with gradcheck's absolute step
    1e-6 at unit amplitude: for the 1e-6-amplitude inputs an absolute step
    1e-6 would itself be a 100 % perturbation."""
    a = float(sim.abs().max())
    x = (sim / a).clone().requires_grad_(True)
    scaled = lambda u: fn(a * u)
    try:
        ok = torch.autograd.gradcheck(scaled, (x,), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True)
        return bool(ok), ""
    except Exception as exc:                    # noqa: BLE001 - report the failure text
        return False, str(exc).split("\n")[0][:300]


def fd_check(fn, sim, gen, steps=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)):
    x = sim.clone().requires_grad_(True)
    value = fn(x)
    value.backward()
    grad = x.grad.detach()
    if not torch.isfinite(grad).all():
        return dict(verdict="FAIL", reason="non-finite gradient", grad_norm=float("nan"))
    p = torch.randn(sim.shape, generator=gen, dtype=torch.float64)
    p = p / p.abs().max() * float(sim.abs().max())
    g = float((grad * p).sum())
    with torch.no_grad():
        J0 = float(fn(sim))
        J_at = lambda h: float(fn(sim + h * p))
        rows = []
        for h in steps:
            jp, jm = J_at(h), J_at(-h)
            rows.append(dict(h=h, fd=(jp - jm) / (2 * h), r1=abs(jp - J0 - h * g)))
    best = min(rows, key=lambda r: abs(r["fd"] - g))
    rel = abs(best["fd"] - g) / max(abs(g), 1e-300)
    valid = [r for r in rows if r["r1"] > 1e-12 * max(abs(J0), 1e-300)]
    slope = float("nan")
    if len(valid) >= 2:
        import numpy as np
        slope = float(np.polyfit(np.log([r["h"] for r in valid[:4]]), np.log([r["r1"] for r in valid[:4]]), 1)[0])
    verdict = "PASS" if rel < 1e-6 or (rel < 1e-4 and slope > 1.7) else "FAIL"
    return dict(J0=J0, g_ad=g, g_fd=best["fd"], rel_err=rel, slope=slope, rows=rows, verdict=verdict,
                grad_norm=float(grad.norm()), grad_max=float(grad.abs().max()))


def main():
    gv.seed_all()
    landfill, engine, Misfit_M_SSIM = import_ifwi()
    results = []
    mssim_2d = Misfit_M_SSIM(is_3d=False)
    mssim_2d.g_masks = mssim_2d.g_masks.double()

    def mssim_3d_engine(obs, syn):
        # The 3D engine builds Misfit_M_SSIM(is_3d=True) and feeds (B, T, R) gathers;
        # the module then rebuilds 2D kernels on every call (default dtype).
        with default_dtype(torch.float64):
            crit = Misfit_M_SSIM(is_3d=True)
            return crit.forward(obs, syn)

    noise_aware_module = Misfit_M_SSIM(is_3d=False)
    noise_aware_module.g_masks = noise_aware_module.g_masks.double()
    noise_aware = landfill.NoiseAwareMSSIM(noise_aware_module, kappa=100.0)
    noise_aware.set_noise(1e-3)
    tv_aniso = DeepGPR.TVRegularization(weight_ep=1.0, weight_sigma=1e-3, method="anisotropic")
    tv_iso = DeepGPR.TVRegularization(weight_ep=1.0, weight_sigma=1e-3, method="isotropic")
    fs = 1.0 / 3e-11

    misfits = {
        # name: (callable(sim, obs) -> scalar, source)
        "L2 (landfill make_misfit l2)": (landfill.make_misfit("l2", str(MSSIM_PARENT)), "landfill"),
        "L2 (3D engine nn.MSELoss)": (engine.build_loss("l2"), "3D engine"),
        "L1 (landfill make_misfit l1)": (landfill.make_misfit("l1", str(MSSIM_PARENT)), "landfill"),
        "M-SSIM 2D (MSSIM, landfill)": (lambda obs, sim: mssim_2d.forward(obs, sim), "MSSIM"),
        "M-SSIM is_3d=True on (B,T,R) (3D engine)": (mssim_3d_engine, "3D engine + MSSIM"),
        "NoiseAwareMSSIM kappa=100 (landfill)": (lambda obs, sim: noise_aware(obs, sim), "landfill"),
        "Envelope L2 (DeepGPR.hilbert_transform)": (
            lambda obs, sim: 0.5 * (DeepGPR.hilbert_transform(sim) - DeepGPR.hilbert_transform(obs)).square().sum(),
            "DeepGPR"),
        "Envelope^2 L2 (hilbert_transform p=2)": (
            lambda obs, sim: 0.5 * (DeepGPR.hilbert_transform(sim, 2) - DeepGPR.hilbert_transform(obs, 2)).square().sum(),
            "DeepGPR"),
        "L2 after DeepGPR.apply_filter (low-pass)": (
            lambda obs, sim: 0.5 * (DeepGPR.apply_filter(sim, fs, 8e8) - DeepGPR.apply_filter(obs, fs, 8e8)).square().sum(),
            "DeepGPR"),
        "L2 after landfill bandlimit (FIR, reflect pad)": (
            lambda obs, sim: torch.mean((landfill.bandlimit(sim, 1.2e9) - landfill.bandlimit(obs, 1.2e9)) ** 2),
            "landfill"),
    }
    gen = torch.Generator().manual_seed(gv.SEED)
    for name, (fn, source) in misfits.items():
        for kind in ("random", "ricker", "shifted", "scaled", "low_amplitude"):
            sim, obs = gathers(kind, gen)
            # The 1.2 GHz bandlimit kernel spans ~2*6*fs/fc taps; keep T long enough for reflect padding.
            if "bandlimit" in name:
                sim = torch.nn.functional.pad(sim, (0, 0, 80, 80))
                obs = torch.nn.functional.pad(obs, (0, 0, 80, 80))
            wrapped = lambda x, obs=obs, fn=fn: fn(obs, x)
            ok, msg = gradcheck(wrapped, sim)
            fd = fd_check(wrapped, sim, gen)
            row = dict(misfit=name, source=source, input=kind, gradcheck=ok, gradcheck_msg=msg, **fd)
            row["verdict"] = "PASS" if ok and fd["verdict"] == "PASS" else "FAIL"
            results.append(row)
            print(f"{name:46s} {kind:14s} gradcheck {ok} FD rel {fd.get('rel_err', float('nan')):.1e} "
                  f"slope {fd.get('slope', float('nan')):.2f} |grad| {fd.get('grad_norm', float('nan')):.2e} "
                  f"-> {row['verdict']} {msg[:80]}", flush=True)

    # Special points ------------------------------------------------------------
    special = []
    sim, obs = gathers("ricker", gen)
    l1 = misfits["L1 (landfill make_misfit l1)"][0]
    tie = obs.clone()
    tie[0, :, 0] = sim[0, :, 0]        # sim == obs on one trace: |.| kink of L1
    ok, msg = gradcheck(lambda x: l1(tie, x), sim)
    special.append(dict(case="L1 with sim == obs on one trace (kink)", gradcheck=ok, msg=msg,
                        note="expected: not differentiable there; torch uses sign(0) = 0"))
    zero = sim.clone()
    zero[0, :, 2] = 0.0                # an all-zero simulated trace
    x = zero.clone().requires_grad_(True)
    mssim_2d.forward(obs, x).backward()
    special.append(dict(case="M-SSIM, all-zero simulated trace", grad_max_zero_trace=float(x.grad[0, :, 2].abs().max()),
                        grad_max_other=float(x.grad[0, :, 3].abs().max()),
                        note="trace / (1e-10 + max|trace|): the normalised trace jumps from 0 to O(1) for any "
                             "perturbation, the loss is discontinuous there and AD returns ~1e10-scaled values"))
    for amp in (1.0, 1e-3, 1e-6, 1e-9, 1e-11):
        x = (amp * sim).clone().requires_grad_(True)
        v = mssim_2d.forward(obs, x)
        v.backward()
        special.append(dict(case=f"M-SSIM amplitude scale {amp:g}", loss=float(v), grad_norm=float(x.grad.norm()),
                            note="per-trace max normalisation: loss scale-invariant until amp ~ 1e-10 (epsilon)"))
    peak = sim.clone()
    idx = int(peak[0, :, 1].abs().argmax())
    peak[0, idx + 1, 1] = peak[0, idx, 1]   # tie of the per-trace maximum
    ok, msg = gradcheck(lambda x: mssim_2d.forward(obs, x), peak)
    special.append(dict(case="M-SSIM with a tie of the per-trace maximum", gradcheck=ok, msg=msg,
                        note="max(|x|) is not differentiable at ties"))
    # 3D engine loss_forward in float32 (as trained) vs float64 gradient.
    sim, obs = gathers("ricker", gen)
    x64 = sim.clone().requires_grad_(True)
    mssim_3d_engine(obs, x64).backward()
    crit32 = Misfit_M_SSIM(is_3d=True)
    x32 = sim.float().clone().requires_grad_(True)
    # baseline_engine.load_mssim_loss returns the bound method criterion.forward.
    engine.loss_forward(crit32.forward, x32, obs.float()).backward()
    special.append(dict(case="3D engine M-SSIM via loss_forward (float32) vs float64",
                        grad_rel_l2=gv.relative_l2(x32.grad, x64.grad)))
    # TV regularisers (model space).
    m = torch.randn((12, 14), generator=gen, dtype=torch.float64)
    s = 1e-2 * torch.rand((12, 14), generator=gen, dtype=torch.float64)
    for name, tv in (("TVRegularization anisotropic", tv_aniso), ("TVRegularization isotropic", tv_iso)):
        ok, msg = gradcheck(lambda e: tv(ep=e, sigma=s), m)
        fd = fd_check(lambda e: tv(ep=e, sigma=s), m, gen)
        row = dict(misfit=name, source="DeepGPR", input="random model", gradcheck=ok, gradcheck_msg=msg, **fd)
        row["fd_verdict"] = row.pop("verdict")
        row["verdict"] = "PASS" if ok and row["fd_verdict"] == "PASS" else "FAIL"
        results.append(row)
        print(f"{name:46s} gradcheck {ok} FD rel {fd['rel_err']:.1e} -> {results[-1]['verdict']} {msg[:80]}", flush=True)
    # Mute mask (landfill): elementwise mask of the residual.
    src, rec = landfill.acquisition(torch.device("cpu"), nx=60)
    mask = landfill.mute_mask(src[:2], rec[:2], 1.0e8, 2e-9, 64).double()
    special.append(dict(case="landfill mute_mask", shape=list(mask.shape), requires_grad=bool(mask.requires_grad),
                        note="constant mask; gradient = mask * dL/dsim"))
    for item in special:
        print("special:", item, flush=True)
    not_tested = ["OT / Wasserstein (Misfit_wasserstein_sinkhorn: import commented out, no source file)",
                  "cross-correlation / normalised cross-correlation misfit (not present)",
                  "convolution-type (source-independent convolved-wavefield) misfit (not present)"]
    gv.save_json("t06_losses.json", dict(results=results, special=special, not_tested=not_tested,
                                         modules=dict(landfill=str(LANDFILL), engine=str(ENGINE_DIR),
                                                      mssim=str(MSSIM_PARENT / "MSSIM" / "MS_SSIM.py"))))
    print("verdicts", {v: sum(r["verdict"] == v for r in results) for v in ("PASS", "FAIL")})


if __name__ == "__main__":
    main()
