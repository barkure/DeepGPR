"""Shared infrastructure of the DeepGPR gradient verification.

Every test script imports this module first. It puts the repository's
``src`` first on ``sys.path`` (the dev DeepGPR, not the installed package),
disables bytecode writing (the IFWI modules are imported read-only from
directories outside the repository) and provides

* synthetic 2D/3D models A-G, acquisition, wavelets and observed data,
* the three solver paths (``cpu``, ``cuda_fast``, ``cuda_general``),
* a float64 misfit evaluated from forward-only runs (finite differences
  never call the native backward) and the AD gradient,
* the Taylor-remainder / central-difference check and its verdict,
* JSON helpers and the environment record.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import DeepGPR  # noqa: E402
from DeepGPR.solver import modeling as _modeling  # noqa: E402

DATA = HERE / "data"
FIG = HERE / "figures"
LOGS = HERE / "logs"
for _d in (DATA, FIG, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20261010
FLOAT32_EPS = float(np.finfo(np.float32).eps)

# Relative steps (times the parameter scale); the default sweep of section 4.
STEPS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
SIGMA_STEPS = (9e-1, 6e-1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3)


# ---------------------------------------------------------------------------
# Environment and I/O
# ---------------------------------------------------------------------------
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict:
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "deepgpr_file": DeepGPR.__file__,
        "deepgpr_version": getattr(DeepGPR, "__version__", None),
        "git_commit": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_threads": torch.get_num_threads(),
        "seed": SEED,
        "solver_precision": "float32 (native CPU and CUDA solvers have no float64 path)",
        "misfit_precision": "float64 accumulation of float32 receiver data",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        info["nvidia_smi"] = _run([
            "nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader"])
        info["nvidia_smi_apps"] = _run([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    return info


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_json(name: str, payload: dict) -> Path:
    path = DATA / name
    payload = dict(payload)
    payload.setdefault("environment", environment())
    path.write_text(json.dumps(to_jsonable(payload), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    return path


def seed_all(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))


# ---------------------------------------------------------------------------
# Solver paths
# ---------------------------------------------------------------------------
PATHS = ("cpu", "cuda_fast", "cuda_general")


def path_device(path: str) -> torch.device:
    return torch.device("cpu" if path == "cpu" else "cuda")


@contextlib.contextmanager
def solver_path(path: str):
    """Force the CUDA general path on 2D models (as tests/test_discrete_adjoint.py)."""
    if path != "cuda_general":
        yield
        return
    original = _modeling._tm2d_fast_path_applies
    _modeling._tm2d_fast_path_applies = lambda *args, **kwargs: False
    try:
        yield
    finally:
        _modeling._tm2d_fast_path_applies = original


def cuda_available() -> bool:
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Synthetic models
# ---------------------------------------------------------------------------
def gaussian(shape, center, radius):
    grids = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in shape], indexing="ij")
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return torch.exp(-r2 / (2.0 * radius ** 2))


EPS_BG, SIG_BG = 4.0, 1.0e-2


def model_2d(name: str, shape=(48, 48)):
    """Models A-G of section 3 (float64 masters; cast to float32 for the solver)."""
    nx, ny = shape
    eps = torch.full(shape, EPS_BG, dtype=torch.float64)
    sig = torch.full(shape, SIG_BG, dtype=torch.float64)
    x = torch.arange(nx, dtype=torch.float64)[:, None]
    y = torch.arange(ny, dtype=torch.float64)[None, :]
    if name == "A":
        pass
    elif name == "B":
        eps = eps + 2.0 * gaussian(shape, (0.45 * nx, 0.40 * ny), 0.09 * nx)
    elif name == "C":
        sig = sig + 2.0e-2 * gaussian(shape, (0.55 * nx, 0.62 * ny), 0.09 * nx)
    elif name == "D":
        eps = eps + 2.0 * gaussian(shape, (0.42 * nx, 0.38 * ny), 0.09 * nx)
        sig = sig + 2.0e-2 * gaussian(shape, (0.62 * nx, 0.66 * ny), 0.09 * nx)
    elif name == "E":
        eps = torch.where(x < 0.35 * nx, torch.tensor(4.0, dtype=torch.float64),
                          torch.where(x < 0.7 * nx, torch.tensor(6.0, dtype=torch.float64),
                                      torch.tensor(9.0, dtype=torch.float64))).expand(shape).clone()
        sig = torch.where(x < 0.35 * nx, torch.tensor(5.0e-3, dtype=torch.float64),
                          torch.where(x < 0.7 * nx, torch.tensor(1.0e-2, dtype=torch.float64),
                                      torch.tensor(2.0e-2, dtype=torch.float64))).expand(shape).clone()
    elif name == "F":
        below = (x > 0.35 * nx + 0.4 * y).to(torch.float64)
        eps = 4.0 + 3.0 * below
        sig = 1.0e-2 + 1.5e-2 * below
    elif name == "G":
        # Anomalies touching the low-x/high-y corner and the high-x face.
        eps = eps + 2.5 * gaussian(shape, (1.5, ny - 2.5), 0.08 * nx)
        sig = sig + 2.0e-2 * gaussian(shape, (nx - 1.5, 0.5 * ny), 0.08 * nx)
    else:
        raise ValueError(name)
    return eps, sig


def observed_perturbation_2d(shape):
    """Difference between the model used for d_obs and the test model."""
    nx, ny = shape
    deps = 0.8 * gaussian(shape, (0.66 * nx, 0.30 * ny), 0.08 * nx)
    dsig = 1.0e-2 * gaussian(shape, (0.36 * nx, 0.72 * ny), 0.08 * nx)
    return deps, dsig


def model_3d(name: str, shape=(16, 16, 12)):
    nx, ny, nz = shape
    eps = torch.full(shape, EPS_BG, dtype=torch.float64)
    sig = torch.full(shape, SIG_BG, dtype=torch.float64)
    if name in ("B", "D"):
        eps = eps + 2.0 * gaussian(shape, (0.45 * nx, 0.4 * ny, 0.5 * nz), 0.12 * nx)
    if name in ("C", "D"):
        sig = sig + 2.0e-2 * gaussian(shape, (0.6 * nx, 0.65 * ny, 0.45 * nz), 0.12 * nx)
    if name == "E":
        z = torch.arange(nx, dtype=torch.float64)[:, None, None]
        eps = (4.0 + 3.0 * (z > 0.5 * nx)).expand(shape).clone()
        sig = (1.0e-2 + 1.0e-2 * (z > 0.5 * nx)).expand(shape).clone()
    return eps, sig


def observed_perturbation_3d(shape):
    nx, ny, nz = shape
    deps = 0.8 * gaussian(shape, (0.65 * nx, 0.3 * ny, 0.55 * nz), 0.1 * nx)
    dsig = 1.0e-2 * gaussian(shape, (0.35 * nx, 0.7 * ny, 0.5 * nz), 0.1 * nx)
    return deps, dsig


# ---------------------------------------------------------------------------
# Problem description
# ---------------------------------------------------------------------------
@dataclass
class Problem:
    eps: torch.Tensor                  # physical model, float64 master
    sig: torch.Tensor
    dx: float
    dt: float
    nt: int
    src_loc: torch.Tensor              # (nstep, nsr, 3) int32
    rec_loc: torch.Tensor              # (nstep, nrx, 3) int32
    wavelet: torch.Tensor              # (nwaveforms, nt, 1) float32
    pml: object = 10
    order: int = 2
    mode: int = 2
    source_direction: int = 2
    receiver_component: int = 2
    d_obs: Optional[torch.Tensor] = None   # float64 (nstep, nt, nrx), CPU
    scale: float = 1.0
    window: Optional[torch.Tensor] = None  # float64 (nt,) mask
    options: Dict = field(default_factory=dict)

    @property
    def dims(self):
        return self.eps.dim()


def ricker(nt, dt, freq=4.0e8, peak=None):
    peak = 1.0 / freq if peak is None else peak
    return DeepGPR.wavelet.ricker(freq, nt, dt, peak, dtype=torch.float32).reshape(1, nt, 1)


def acquisition_2d(shape, surface=6, n_shots=2, rec_step=5):
    nx, ny = shape
    margin = 4
    src_y = np.linspace(0.25 * ny, 0.75 * ny, n_shots).round().astype(int)
    rec_y = np.arange(margin, ny - margin + 1, rec_step)
    src = torch.zeros((n_shots, 1, 3), dtype=torch.int32)
    src[:, 0, 0] = surface
    src[:, 0, 1] = torch.as_tensor(src_y, dtype=torch.int32)
    bottom = nx - 1 - surface
    rec = []
    for _ in range(n_shots):
        line = [(surface, int(y), 0) for y in rec_y] + [(bottom, int(y), 0) for y in rec_y]
        rec.append(line)
    return src, torch.tensor(rec, dtype=torch.int32)


def acquisition_3d(shape, surface=3):
    nx, ny, nz = shape
    src = torch.tensor([[[surface, ny // 3, nz // 2]], [[surface, 2 * ny // 3, nz // 2]]], dtype=torch.int32)
    recs = []
    for _ in range(2):
        line = [(surface, y, z) for y in range(3, ny - 2, 4) for z in range(3, nz - 2, 4)]
        line += [(nx - 1 - surface, y, z) for y in range(3, ny - 2, 4) for z in range(3, nz - 2, 4)]
        recs.append(line)
    return src, torch.tensor(recs, dtype=torch.int32)


def make_problem_2d(name="D", shape=(48, 48), nt=400, dt=2.5e-11, dx=0.02, pml=10, order=2,
                    freq=4.0e8, n_shots=2, obs_path="cpu") -> Problem:
    eps, sig = model_2d(name, shape)
    src, rec = acquisition_2d(shape, n_shots=n_shots)
    p = Problem(eps=eps, sig=sig, dx=dx, dt=dt, nt=nt, src_loc=src, rec_loc=rec,
                wavelet=ricker(nt, dt, freq), pml=pml, order=order, mode=2)
    attach_observed(p, *observed_perturbation_2d(shape), path=obs_path)
    return p


def make_problem_3d(name="D", shape=(16, 16, 12), nt=150, dt=2.0e-11, dx=0.02, pml=4, order=2,
                    freq=4.0e8, obs_path="cpu") -> Problem:
    eps, sig = model_3d(name, shape)
    src, rec = acquisition_3d(shape)
    p = Problem(eps=eps, sig=sig, dx=dx, dt=dt, nt=nt, src_loc=src, rec_loc=rec,
                wavelet=ricker(nt, dt, freq), pml=pml, order=order, mode=3)
    attach_observed(p, *observed_perturbation_3d(shape), path=obs_path)
    return p


def attach_observed(problem: Problem, deps, dsig, path="cpu"):
    """d_obs from the perturbed model; the residual at the test model is O(1)."""
    with torch.no_grad():
        d = forward_data(problem, path, problem.eps + deps, problem.sig + dsig)
    problem.d_obs = d.double().cpu()
    problem.scale = float(problem.d_obs.abs().max())
    return problem


# ---------------------------------------------------------------------------
# Forward, misfit and AD gradient
# ---------------------------------------------------------------------------
def compute(problem: Problem, path: str, eps, sig, *, source=None, states=None, history=False,
            **options):
    device = path_device(path)
    kwargs = dict(
        device=device, dx=problem.dx, dt=problem.dt,
        source_amplitudes=(problem.wavelet if source is None else source).to(device),
        source_location=problem.src_loc.to(device), receiver_location=problem.rec_loc.to(device),
        eps_r=eps.to(device=device, dtype=torch.float32) if not eps.requires_grad else eps,
        sigma=sig.to(device=device, dtype=torch.float32) if not sig.requires_grad else sig,
        pmlthick=problem.pml, fdtd_order=problem.order, mode=problem.mode,
        source_direction=problem.source_direction, receiver_component=problem.receiver_component,
        save_wavefield_history=history,
    )
    if states is not None:
        kwargs.update(E=tuple(states[:3]), H=tuple(states[3:6]), PML=tuple(states[6:]))
    merged = dict(problem.options)
    merged.update(options)
    if history:
        merged.setdefault("model_gradient_sampling_interval", 1)
        merged.setdefault("wavefield_storage_dtype", torch.float32)
    kwargs.update(merged)
    with solver_path(path):
        return DeepGPR.compute(**kwargs)


def forward_data(problem: Problem, path: str, eps, sig, **kw):
    return compute(problem, path, eps, sig, **kw)[-1]


def misfit(problem: Problem, data: torch.Tensor) -> torch.Tensor:
    """0.5 * sum(((d - d_obs) * window / scale)^2), accumulated in float64."""
    obs = problem.d_obs.to(data.device)
    residual = (data.double() - obs) / problem.scale
    if problem.window is not None:
        residual = residual * problem.window.to(data.device)[None, :, None]
    return 0.5 * residual.square().sum()


def objective_forward(problem: Problem, path: str, eps, sig, *, source=None, states=None,
                      **options) -> float:
    """J from a forward-only run: no history, no autograd graph, no native backward."""
    with torch.no_grad():
        if states is not None:
            states = [s.detach().clone() for s in states]   # the solver advances states in place
        data = forward_data(problem, path, eps.detach(), sig.detach(),
                            source=None if source is None else source.detach(),
                            states=states, history=False, **options)
        return float(misfit(problem, data))


@dataclass
class ADResult:
    J: float
    grad_eps: Optional[torch.Tensor]
    grad_sig: Optional[torch.Tensor]
    grad_src: Optional[torch.Tensor]
    grad_states: Optional[List[torch.Tensor]]
    tm2d_fast_path: Optional[bool]
    data: torch.Tensor


def objective_ad(problem: Problem, path: str, eps, sig, *, wrt=("eps", "sig"), source=None,
                 states=None, **options) -> ADResult:
    device = path_device(path)
    eps_leaf = eps.detach().to(device=device, dtype=torch.float32).requires_grad_("eps" in wrt)
    sig_leaf = sig.detach().to(device=device, dtype=torch.float32).requires_grad_("sig" in wrt)
    src = (problem.wavelet if source is None else source).detach().to(device).clone()
    src.requires_grad_("src" in wrt)
    state_leaves = None
    if states is not None:
        state_leaves = [s.detach().to(device).clone().requires_grad_("states" in wrt) for s in states]
    need_history = "eps" in wrt or "sig" in wrt or "states" in wrt or "src" in wrt
    out = compute(problem, path, eps_leaf, sig_leaf, source=src, states=state_leaves,
                  history=need_history, **options)
    data = out[-1]
    flag = None
    if data.grad_fn is not None and hasattr(data.grad_fn, "config"):
        flag = bool(getattr(data.grad_fn.config, "tm2d_fast_path", False))
    J = misfit(problem, data)
    J.backward()
    return ADResult(
        J=float(J.detach()),
        grad_eps=eps_leaf.grad.detach().double().cpu() if eps_leaf.grad is not None else None,
        grad_sig=sig_leaf.grad.detach().double().cpu() if sig_leaf.grad is not None else None,
        grad_src=src.grad.detach().double().cpu() if src.grad is not None else None,
        grad_states=[s.grad.detach().double().cpu() if s.grad is not None else None
                     for s in state_leaves] if state_leaves is not None else None,
        tm2d_fast_path=flag,
        data=data.detach().double().cpu(),
    )


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------
def edge_mask(shape, pml) -> torch.Tensor:
    """Zero on the outermost physical layer of every face that carries CPML.

    Those cells are replicated into the CPML and feed the CPML coefficient
    averages, so perturbing them changes the fixed boundary (section 2)."""
    faces = _faces(pml, len(shape))
    mask = torch.ones(shape, dtype=torch.float64)
    for axis in range(len(shape)):
        if faces[2 * axis] > 0:
            mask.narrow(axis, 0, 1).zero_()
        if faces[2 * axis + 1] > 0:
            mask.narrow(axis, shape[axis] - 1, 1).zero_()
    return mask


def _faces(pml, dims):
    if isinstance(pml, int):
        return [pml] * 4 + ([pml] * 2 if dims == 3 else [0, 0])
    pml = list(pml)
    if len(pml) == 4:
        return pml + [0, 0]
    return pml


def normalise(direction: torch.Tensor, scale: float) -> torch.Tensor:
    peak = float(direction.abs().max())
    if peak == 0.0:
        raise ValueError("empty direction")
    return direction / peak * scale


def smooth_noise(shape, generator, passes=3):
    field = torch.randn(shape, generator=generator, dtype=torch.float64)
    for _ in range(passes):
        for axis in range(len(shape)):
            field = 0.25 * (torch.roll(field, 1, axis) + 2.0 * field + torch.roll(field, -1, axis))
    return field


def directions(shape, pml, kind: str, generator, pixel=None) -> torch.Tensor:
    """Unit-peak perturbation directions that respect the fixed-boundary rule."""
    mask = edge_mask(shape, pml)
    if kind == "pixel":
        d = torch.zeros(shape, dtype=torch.float64)
        d[tuple(pixel)] = 1.0
    elif kind == "random":
        d = torch.randn(shape, generator=generator, dtype=torch.float64)
    elif kind == "local":
        center = [0.5 * n for n in shape] if pixel is None else pixel
        d = gaussian(shape, center, 0.06 * shape[0] + 1.0)
        d = d * (1.0 + 0.3 * torch.randn(shape, generator=generator, dtype=torch.float64))
    elif kind == "smooth":
        grids = torch.meshgrid(*[torch.linspace(0, 1, n, dtype=torch.float64) for n in shape], indexing="ij")
        d = torch.ones(shape, dtype=torch.float64)
        for g, k in zip(grids, (1.3, 0.8, 1.1)):
            d = d * torch.cos(math.pi * k * g + 0.3)
        d = d + 0.5 * smooth_noise(shape, generator, passes=6) / 3.0
    else:
        raise ValueError(kind)
    d = d * mask
    return d / float(d.abs().max())


# ---------------------------------------------------------------------------
# Taylor / finite-difference check
# ---------------------------------------------------------------------------
def _noise_from_fit(J0, rows, max_points=6):
    """Float32 noise level of J: residual of a quartic fit of J(h) over the smaller steps.

    The rounding error of the float32 forward is a chaotic function of the
    model; a smooth polynomial cannot follow it, so the fit residual over
    +-h (13 points for the 6 smallest steps) estimates its standard deviation.
    """
    small = sorted(rows, key=lambda r: r["h"])[:max_points]
    if len(small) < 4:
        return float("nan")
    hmax = max(r["h"] for r in small)
    xs = [0.0] + [r["h"] / hmax for r in small] + [-r["h"] / hmax for r in small]
    ys = [J0] + [r["J_plus"] for r in small] + [r["J_minus"] for r in small]
    xs, ys = np.array(xs), np.array(ys) - J0
    basis = np.stack([xs ** k for k in range(5)], 1)
    coef, *_ = np.linalg.lstsq(basis, ys, rcond=None)
    residual = ys - basis @ coef
    dof = max(len(xs) - basis.shape[1], 1)
    return float(np.sqrt(np.sum(residual ** 2) / dof))


def taylor_check(J0: float, g_ad: float, J_at: Callable[[float], float], steps: Sequence[float],
                 *, rel_tol: float = 2.0e-3, noise_factor: float = 10.0,
                 cs_scale: Optional[float] = None) -> dict:
    """Central differences and first-order Taylor remainders along one direction.

    With J(+-h) = J0 +- h g + h^2 c / 2 +- h^3 d / 6 + noise:

    * r(h) = |J(m + h p) - J(m) - h g_AD| must fall like h^2 until the float32
      noise floor; a wrong g_AD (g_AD = g + D) leaves ~h |D| (slope 1).
    * The central difference has error -D + O(h^2) + noise / h.
    * The float32 noise level N of J is the residual of a quartic fit of J(h)
      over the six smallest steps (see _noise_from_fit), floored by two tiny
      probe steps.

    Verdict
      PASS          Taylor slope >= 1.7 over the three smallest steps with
                    r(h) > 10 N (and >= 1.5 between the two smallest) and a central
                    difference within rel_tol of g_AD at two adjacent steps
                    (or for the extrapolation h -> 0);
      PASS (noise-limited)
                    as above but the agreement is limited by float32 noise
                    because the directional derivative is small compared with
                    |grad J| |p| (cancellation): the FD-AD difference is within
                    the measured FD uncertainty (adjacent-step spread + 3 N / h)
                    and that uncertainty is < 1e-4 of |grad J| |p|;
      INCONCLUSIVE  the Taylor slope cannot be evaluated (signal at the noise
                    floor) or the FD uncertainty is too large to decide;
      FAIL          slope < 1.7, or an FD-AD difference larger than the FD
                    uncertainty.
    """
    rows = []
    for h in steps:
        jp, jm = J_at(+h), J_at(-h)
        fd = (jp - jm) / (2.0 * h)
        rows.append(dict(h=h, J_plus=jp, J_minus=jm, fd=fd,
                         r1=abs(jp - J0 - h * g_ad), r1_minus=abs(jm - J0 + h * g_ad),
                         sym=abs(0.5 * (jp + jm) - J0),
                         fd_abs_err=abs(fd - g_ad),
                         fd_rel_err=abs(fd - g_ad) / max(abs(g_ad), abs(fd), 1e-300)))
    rows.sort(key=lambda r: -r["h"])
    h_n = min(steps) / 100.0
    probe = max(abs(J_at(+h_n) - J0 - h_n * g_ad), abs(J_at(-h_n) - J0 + h_n * g_ad))
    noise_fit = _noise_from_fit(J0, rows)
    noise = max(noise_fit if math.isfinite(noise_fit) else 0.0, probe, 1e-15 * max(abs(J0), 1e-300))
    valid = [r for r in rows if min(r["r1"], r["r1_minus"]) > noise_factor * noise]
    slopes = [math.log(a["r1"] / b["r1"]) / math.log(a["h"] / b["h"]) for a, b in zip(valid[:-1], valid[1:])]
    # The asymptotic order is read from the three smallest steps above the noise floor
    # (large steps may be outside the Taylor regime of strongly nonlinear maps).
    tail = valid[-3:]
    if len(tail) >= 2:
        fit_slope = float(np.polyfit(np.log([r["h"] for r in tail]), np.log([r["r1"] for r in tail]), 1)[0])
    else:
        fit_slope = float("nan")
    if len(valid) >= 2:
        fit_slope_all = float(np.polyfit(np.log([r["h"] for r in valid]), np.log([r["r1"] for r in valid]), 1)[0])
    else:
        fit_slope_all = float("nan")
    best = min(rows, key=lambda r: r["fd_rel_err"])
    pair = min((max(a["fd_rel_err"], b["fd_rel_err"]), a["h"]) for a, b in zip(rows[:-1], rows[1:]))
    usable = [r for r in rows if r["sym"] > noise_factor * noise] or rows
    if len(usable) >= 3:
        hs = np.array([r["h"] for r in usable])
        fds = np.array([r["fd"] for r in usable])
        basis = np.stack([np.ones_like(hs), hs ** 2, hs ** 4], 1)[:, : min(3, len(usable) - 1)]
        g_extrap = float(np.linalg.lstsq(basis, fds, rcond=None)[0][0])
        rel_extrap = abs(g_extrap - g_ad) / max(abs(g_ad), 1e-300)
    else:
        g_extrap, rel_extrap = float("nan"), float("nan")
    agreement = min(pair[0], rel_extrap if math.isfinite(rel_extrap) else 1.0)
    # FD uncertainty: spread between adjacent steps (truncation) + 3 N / h (noise).
    z_min, uncertainty = float("inf"), float("inf")
    for a, b in zip(rows[:-1], rows[1:]):
        u = abs(a["fd"] - b["fd"]) + 3.0 * noise / b["h"]
        z = abs(b["fd"] - g_ad) / max(u, 1e-300)
        if z < z_min or (z <= 1.0 and u < uncertainty):
            z_min = min(z_min, z)
            if z <= 1.0:
                uncertainty = min(uncertainty, u)
    rel_cs = (min(r["fd_abs_err"] for r in rows) / cs_scale) if cs_scale else float("nan")
    signal = abs(g_ad) * max(steps)
    slope_ok = len(valid) >= 3 and fit_slope >= 1.7 and slopes[-1] >= 1.5
    verdict, reason = "PASS", ""
    if len(valid) >= 3 and not slope_ok:
        verdict, reason = "FAIL", f"Taylor slope {fit_slope:.2f} (smallest pair {slopes[-1]:.2f})"
    elif agreement <= rel_tol:
        if signal < 1e2 * noise or len(valid) < 3:
            verdict, reason = "INCONCLUSIVE", f"FD agrees but only {len(valid)} steps above {noise_factor:g} x noise"
    elif z_min <= 1.0 and cs_scale and uncertainty < 1e-4 * cs_scale and slope_ok:
        verdict = "PASS (noise-limited)"
        reason = (f"|g|/(|grad||p|) = {abs(g_ad) / cs_scale:.1e}; FD-AD difference within the FD "
                  f"uncertainty {uncertainty:.1e} = {uncertainty / cs_scale:.1e} |grad||p|")
    elif z_min <= 1.0:
        verdict, reason = "INCONCLUSIVE", f"FD uncertainty too large (relative error {agreement:.1e})"
    else:
        verdict, reason = "FAIL", f"central-difference relative error {agreement:.2e} beyond FD uncertainty (z={z_min:.1f})"
    return dict(J0=J0, g_ad=g_ad, g_fd=best["fd"], best_h=best["h"], rel_err=best["fd_rel_err"],
                pair_rel_err=pair[0], g_fd_extrap=g_extrap, rel_err_extrap=rel_extrap,
                agreement=agreement, abs_err=best["fd_abs_err"], fit_slope=fit_slope, fit_slope_all=fit_slope_all,
                local_slopes=slopes, n_valid=len(valid), noise=noise, noise_fit=noise_fit,
                noise_probe=probe, noise_step=h_n, z_min=z_min, fd_uncertainty=uncertainty,
                cs_scale=cs_scale, rel_err_cs=rel_cs, rows=rows, verdict=verdict, reason=reason)


def physical_check(problem: Problem, path: str, kind: str, d_eps, d_sig, ad: ADResult, steps=STEPS,
                   eps0=None, sig0=None, **options) -> dict:
    """Directional derivative check for eps / sigma / joint directions (scaled)."""
    eps0 = problem.eps if eps0 is None else eps0
    sig0 = problem.sig if sig0 is None else sig0
    s_eps, s_sig = float(eps0.mean()), float(sig0.mean())
    pe = d_eps * s_eps if d_eps is not None else torch.zeros_like(eps0)
    ps = d_sig * s_sig if d_sig is not None else torch.zeros_like(sig0)
    g = 0.0
    cs2 = 0.0
    if d_eps is not None:
        g += float((ad.grad_eps * pe).sum())
        cs2 += float(ad.grad_eps.norm() * pe.norm()) ** 2
    if d_sig is not None:
        g += float((ad.grad_sig * ps).sum())
        cs2 += float(ad.grad_sig.norm() * ps.norm()) ** 2
    # Keep eps >= 1 and sigma >= 0 (and PEC far away) at every step.
    allowed = [h for h in steps
               if float((eps0 - h * pe.abs()).min()) >= 1.0 and float((sig0 - h * ps.abs()).min()) >= 0.0
               and float((sig0 + h * ps.abs()).max()) < 1.0]
    if len(allowed) < 5:
        raise ValueError("too few steps inside the physical bounds")
    J_at = lambda h: objective_forward(problem, path, eps0 + h * pe, sig0 + h * ps, **options)
    result = taylor_check(ad.J, g, J_at, allowed, cs_scale=math.sqrt(cs2))
    result.update(path=path, kind=kind, scale_eps=s_eps, scale_sig=s_sig)
    return result


def summarise(result: dict) -> str:
    return (f"{result.get('path','')} {result.get('label','')} {result.get('kind','')}: "
            f"AD={result['g_ad']:.6e} FD={result['g_fd']:.6e} rel={result['rel_err']:.2e} "
            f"pair={result.get('pair_rel_err', float('nan')):.2e} "
            f"extrap={result.get('rel_err_extrap', float('nan')):.2e} n={result.get('n_valid')} "
            f"slope={result['fit_slope']:.2f} local={['%.2f' % s for s in result['local_slopes']]} "
            f"-> {result['verdict']} {result['reason']}")


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().cpu()
    b = b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().cpu().flatten()
    b = b.double().cpu().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


# ---------------------------------------------------------------------------
# Frozen external PML (tests/test_external_pml.py::raw_compute)
# ---------------------------------------------------------------------------
def pad_replicate(model: torch.Tensor, pml) -> torch.Tensor:
    """The extended material DeepGPR builds (edge replication, no Yee halo)."""
    faces = _faces(pml, model.dim())
    m = model if model.dim() == 3 else model[..., None]
    padded = torch.nn.functional.pad(m[None, None], (faces[4], faces[5], faces[2], faces[3], faces[0], faces[1]),
                                     mode="replicate")[0, 0]
    return padded if model.dim() == 3 else padded[..., 0]


def embed_physical(direction: torch.Tensor, pml) -> torch.Tensor:
    """Zero-pad a physical-model perturbation onto the extended grid."""
    faces = _faces(pml, direction.dim())
    d = direction if direction.dim() == 3 else direction[..., None]
    padded = torch.nn.functional.pad(d[None, None], (faces[4], faces[5], faces[2], faces[3], faces[0], faces[1]))[0, 0]
    return padded if direction.dim() == 3 else padded[..., 0]


def objective_frozen_pml(problem: Problem, path: str, eps_full, sig_full) -> float:
    """J for explicitly supplied extended materials with CPML coefficients of the
    unperturbed model: the extended (replicated) materials and the coefficient
    averages stay fixed, as in the documented FWI gradient."""
    faces = _faces(problem.pml, problem.eps.dim())
    pml_tensor = torch.tensor(faces, dtype=torch.int32)
    fixed_eps = pad_replicate(problem.eps, problem.pml).float()
    orig_init, orig_shift, orig_coeffs = (_modeling.initialization, _modeling._shift_locations,
                                          _modeling.build_pml_coeffs)

    def prepared(*values):
        values = list(values)
        values[-2] = 0          # validate the full grid without extending again
        result = list(orig_init(*values))
        result[14] = pml_tensor.clone()
        return tuple(result)

    def fixed_coefficients(eps_r, *values):
        e = fixed_eps.to(device=eps_r.device, dtype=eps_r.dtype)
        return orig_coeffs(e if e.dim() == 3 else e[..., None], *values)

    offset = torch.tensor(faces[::2], dtype=torch.int32)
    shifted = Problem(**{**problem.__dict__})
    shifted.src_loc = problem.src_loc + offset
    shifted.rec_loc = problem.rec_loc + offset
    _modeling.initialization = prepared
    _modeling._shift_locations = lambda loc, p, device: loc.to(device=device, dtype=torch.int32).contiguous()
    _modeling.build_pml_coeffs = fixed_coefficients
    try:
        with torch.no_grad():
            data = forward_data(shifted, path, eps_full, sig_full)
    finally:
        _modeling.initialization = orig_init
        _modeling._shift_locations = orig_shift
        _modeling.build_pml_coeffs = orig_coeffs
    return float(misfit(problem, data)), data


def dense_extrapolation(J_at: Callable[[float], float], g_ad: float, h_max: float, h_min: float,
                        n: int = 13, noise: Optional[float] = None) -> dict:
    """Central differences on a dense geometric grid of steps and a weighted fit
    fd(h) = g0 + a h^2 + b h^4 (weights from the noise term N / h). Returns g0,
    its standard error and the normalised discrepancy |g0 - g_AD| / se."""
    hs = np.geomspace(h_max, h_min, n)
    J0 = J_at(0.0)
    fds, Js = [], []
    for h in hs:
        jp, jm = J_at(+h), J_at(-h)
        Js.append((jp, jm))
        fds.append((jp - jm) / (2 * h))
    fds = np.array(fds)
    basis = np.stack([np.ones_like(hs), hs ** 2, hs ** 4], 1)
    # First pass: unweighted fit, noise level from its residual at the small steps.
    coef = np.linalg.lstsq(basis, fds, rcond=None)[0]
    resid = fds - basis @ coef
    if noise is None:
        noise = float(np.median(np.abs(resid[-5:]) * hs[-5:]))
    sigma = np.maximum(noise / hs, 1e-30)
    w = 1.0 / sigma
    coef, *_ = np.linalg.lstsq(basis * w[:, None], fds * w, rcond=None)
    resid = (fds - basis @ coef) * w
    dof = max(len(hs) - 3, 1)
    scale = max(float(np.sqrt(np.sum(resid ** 2) / dof)), 1.0)
    cov = np.linalg.inv((basis * w[:, None]).T @ (basis * w[:, None])) * scale ** 2
    g0, se = float(coef[0]), float(np.sqrt(cov[0, 0]))
    return dict(hs=hs.tolist(), fds=fds.tolist(), J0=J0, g0=g0, se=se, g_ad=g_ad,
                rel_err=abs(g0 - g_ad) / max(abs(g_ad), 1e-300), rel_se=se / max(abs(g_ad), 1e-300),
                z=abs(g0 - g_ad) / max(se, 1e-300), noise=noise, chi2_scale=scale)
