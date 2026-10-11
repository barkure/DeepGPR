"""Independent float64 reference of DeepGPR's 2D Ez-TM order-2 scheme (test oracle).

A line-by-line re-implementation of the discrete operator executed by the
native CPU backend (deepgpr_cpu.c: update_h_cpu, cpml_h_cpu, update_e_cpu,
cpml_e_cpu, inject_sources_cpu, sample_receivers_cpu) for 2D models, Ez
source and receivers, fdtd_order = 2, mu_r = 1, written in PyTorch float64
without any DeepGPR solver code. Only the CPML profile coefficients (a fixed,
detached function of the boundary material) are taken from
DeepGPR.solver.pml.build_pml_coeffs, as float32 values promoted to float64.

Because it runs in float64 with autograd, it provides
* forward data to compare with DeepGPR's float32 forward,
* the exact float64 gradient of the same discrete problem (autograd through
  the time loop, an independent adjoint) to compare cell by cell, and
* finite differences with tiny steps (1e-7) far below the float32 noise.

The physical model is extended into the CPML by edge replication exactly like
DeepGPR (preprocessing.model_setup._ExtendModel), and the gradient of the
extended model is cropped to the physical cells (no CPML sensitivities summed
into the edge), matching DeepGPR's fixed-boundary definition.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import gv_common as gv
from DeepGPR.solver.pml import build_pml_coeffs

F32 = lambda x: float(np.float32(x))   # noqa: E731  (constants as the native code sees them)
E0 = F32(8.8541878128e-12)
M0 = F32(1.25663706212e-06)


class _CropExtend(torch.autograd.Function):
    """Replicate into the CPML; backward returns only the physical slice (as DeepGPR)."""

    @staticmethod
    def forward(ctx, model, pml):
        ctx.sl = (slice(pml[0], pml[0] + model.shape[0]), slice(pml[2], pml[2] + model.shape[1]))
        return F.pad(model[None, None, :, :, None], (0, 0, pml[2], pml[3], pml[0], pml[1]),
                     mode="replicate")[0, 0, :, :, 0]

    @staticmethod
    def backward(ctx, g):
        return g[ctx.sl].contiguous(), None


def _coefficients(eps_ext, pml, dt, spacing):
    nx, ny = eps_ext.shape
    eps3 = eps_ext.detach().float()[..., None]
    mu_pad = torch.ones((nx + 1, ny + 1, 2), dtype=torch.float32)
    arrays = build_pml_coeffs(eps3, mu_pad, dt, spacing, nx, ny, 1,
                              torch.tensor(pml + [0, 0], dtype=torch.int32), torch.device("cpu"), torch.float32)
    coeffs = arrays[6:]
    names = ("x0", "xm", "y0", "ym")
    out = {}
    for face, name in enumerate(names):
        e, h = coeffs[2 * face], coeffs[2 * face + 1]
        out[name] = (e.double().reshape(4, -1) if e.numel() else None,
                     h.double().reshape(4, -1) if h.numel() else None)
    return out


def _coefficients_differentiable(eps_ext, pml, dt, spacing, dtype):
    """CPML coefficients as a differentiable function of the boundary-plane average
    (port of solver/pml.py: quartic sigma profile, alpha = 0, kappa = 1)."""
    from DeepGPR.config.constants import VACUUM_PERMEABILITY as m0, VACUUM_PERMITTIVITY as e0
    nx, ny = eps_ext.shape
    planes = {"x0": eps_ext[0, :], "xm": eps_ext[nx - pml[1], :] if pml[1] else None,
              "y0": eps_ext[:, 0], "ym": eps_ext[:, ny - pml[3]] if pml[3] else None}
    out = {}
    for face, name in enumerate(("x0", "xm", "y0", "ym")):
        L = pml[face]
        if L == 0:
            out[name] = (None, None)
            continue
        d = spacing[face // 2]
        sigma_max = (0.8 * 5) / ((m0 / e0) ** 0.5 * d * torch.sqrt(planes[name].mean() * 1.0))
        tmp = (torch.linspace(0, L + 0.5, steps=2 * (L + 1), dtype=dtype) / L) ** 4
        prof_e, prof_h = tmp[0:-1:2][:-1], tmp[1::2][:-1]
        res = []
        for prof in (prof_e, prof_h):
            sig_p = prof * sigma_max
            den = 2 * e0 + dt * sig_p
            RA = (2 * e0) / den
            RB = (2 * e0) / den
            RE = (2 * e0 - dt * sig_p) / den
            RF = (2 * sig_p * dt) / den
            res.append(torch.stack([RA, RB, RE, RF]))
        out[name] = tuple(res)
    return out


def simulate(problem: "gv.Problem", eps_phys: torch.Tensor, sig_phys: torch.Tensor,
             source: torch.Tensor | None = None, dtype=torch.float64, frozen=None,
             total: bool = False) -> torch.Tensor:
    """Receiver data (nstep, nt, nrx) for a 2D order-2 problem (float64 by default;
    dtype=torch.float32 runs the same reference in single precision to measure
    the float32 rounding level of this computation).

    Boundary treatment:
      default  DeepGPR's definition: the CPML materials replicate the current
               edge, the CPML coefficients are detached and the gradient is the
               physical slice (fixed-boundary gradient);
      frozen=(eps0, sig0)
               the CPML materials and coefficients come from eps0/sig0 and stay
               fixed: the forward whose exact derivative is the fixed-boundary
               gradient (used for float64 finite differences);
      total=True
               standard replicate padding (CPML sensitivities summed into the
               edge) and differentiable CPML coefficients: the total derivative
               of the forward that rebuilds the CPML from the current edge.
    """
    assert problem.eps.dim() == 2 and problem.order == 2
    pml = gv._faces(problem.pml, 2)[:4]
    spacing = problem.dx if isinstance(problem.dx, (tuple, list)) else (problem.dx,) * 3
    dx, dy, dz = (F32(v) for v in spacing)
    dt = F32(problem.dt)
    pad = lambda m, mode: F.pad(m[None, None, :, :, None], (0, 0, pml[2], pml[3], pml[0], pml[1]),
                                mode=mode)[0, 0, :, :, 0]
    if frozen is not None:
        eps0, sig0 = (t.to(dtype) for t in frozen)
        eps = pad(eps0, "replicate") + pad(eps_phys.to(dtype) - eps0, "constant")
        sig = pad(sig0, "replicate") + pad(sig_phys.to(dtype) - sig0, "constant")
        coeff_source = pad(eps0, "replicate")
    elif total:
        eps = pad(eps_phys.to(dtype), "replicate")
        sig = pad(sig_phys.to(dtype), "replicate")
        coeff_source = None
    else:
        eps = _CropExtend.apply(eps_phys.to(dtype), pml)
        sig = _CropExtend.apply(sig_phys.to(dtype), pml)
        coeff_source = eps
    Nx, Ny = eps.shape
    NXF, NYF = Nx + 1, Ny + 1
    # Update coefficients on the (NXF, NYF) field grid; the halo row/column stays 0.
    e_term = E0 * eps / dt
    s_term = 0.5 * sig
    EA = e_term + s_term
    ce_hist = F.pad((e_term - s_term) / EA, (0, 1, 0, 1))
    ce_curl = F.pad((1.0 / dx) / EA, (0, 1, 0, 1))
    ce_rhs = F.pad(1.0 / EA, (0, 1, 0, 1))
    HA = M0 * 1.0 / dt
    ch_curl = F.pad(torch.full((Nx, Ny), (1.0 / dx) / HA, dtype=dtype), (0, 1, 0, 1))
    ch_rhs = F.pad(torch.full((Nx, Ny), 1.0 / HA, dtype=dtype), (0, 1, 0, 1))
    ratio_y = dx / dy
    if total:
        coeffs = _coefficients_differentiable(eps, pml, problem.dt, spacing, dtype)
    else:
        coeffs = _coefficients(coeff_source, pml, problem.dt, spacing)

    I = torch.arange(NXF)[:, None].expand(NXF, NYF)
    J = torch.arange(NYF)[None, :].expand(NXF, NYF)
    zero = torch.zeros((NXF, NYF), dtype=dtype)

    def face_maps(name, electric):
        """(mask, RA-1, RB, RE, RF) maps of one CPML face on the field grid."""
        p = {"x0": pml[0], "xm": pml[1], "y0": pml[2], "ym": pml[3]}[name]
        if p == 0:
            return None
        R = coeffs[name][0 if electric else 1]
        if electric:
            if name == "x0":
                mask = (I >= 1) & (I <= p)
                idx = (p - I).clamp(0, p - 1)
            elif name == "xm":
                mask = (I >= NXF - 1 - p) & (I < NXF - 1) & (I > 0)
                idx = (I - (NXF - 1 - p)).clamp(0, p - 1)
            elif name == "y0":
                mask = (J >= 1) & (J <= p)
                idx = (p - J).clamp(0, p - 1)
            else:
                mask = (J >= NYF - 1 - p) & (J < NYF - 1) & (J > 0)
                idx = (J - (NYF - 1 - p)).clamp(0, p - 1)
        else:
            if name == "x0":
                mask = I < p
                idx = (p - 1 - I).clamp(0, p - 1)
            elif name == "xm":
                mask = (I >= NXF - 1 - p) & (I < NXF - 1)
                idx = (I - (NXF - 1 - p)).clamp(0, p - 1)
            elif name == "y0":
                mask = J < p
                idx = (p - 1 - J).clamp(0, p - 1)
            else:
                mask = (J >= NYF - 1 - p) & (J < NYF - 1)
                idx = (J - (NYF - 1 - p)).clamp(0, p - 1)
        m = mask.to(dtype)
        R = R.to(dtype)
        return m, (R[0][idx] - 1.0) * m, R[1][idx] * m, R[2][idx] * m, R[3][idx] * m

    e_faces = {n: face_maps(n, True) for n in ("x0", "xm", "y0", "ym")}
    h_faces = {n: face_maps(n, False) for n in ("x0", "xm", "y0", "ym")}
    do_ez = ((I > 0) & (I < NXF - 1) & (J > 0) & (J < NYF - 1)).to(dtype)
    do_hx = ((I > 0) & (I < NXF - 1) & (J < NYF - 1)).to(dtype)
    do_hy = ((I < NXF - 1) & (J > 0) & (J < NYF - 1)).to(dtype)

    def bwd_x(f):   # f[i] - f[i-1]
        return f - F.pad(f, (0, 0, 1, 0))[:-1]

    def bwd_y(f):
        return f - F.pad(f, (1, 0))[:, :-1]

    def fwd_x(f):   # f[i+1] - f[i]
        return F.pad(f, (0, 0, 0, 1))[1:] - f

    def fwd_y(f):
        return F.pad(f, (0, 1))[:, 1:] - f

    wave = (problem.wavelet if source is None else source).to(dtype)
    nsteps = problem.src_loc.shape[0]
    offset = torch.tensor([pml[0], pml[2]])
    out = []
    for shot in range(nsteps):
        src = [tuple((problem.src_loc[shot, s, :2] + offset).tolist()) for s in range(problem.src_loc.shape[1])]
        rec = (problem.rec_loc[shot, :, :2] + offset).long()
        Ez = zero.clone()
        Hx = zero.clone()
        Hy = zero.clone()
        phi_e = {n: zero.clone() for n in e_faces}
        phi_h = {n: zero.clone() for n in h_faces}
        trace = []
        for it in range(problem.nt):
            # H base update (Ey = Ex = 0 in the 2D TM subsystem).
            Hx = Hx + do_hx * ((1.0 - 1.0) * Hx - ch_curl * ratio_y * fwd_y(Ez))
            Hy = Hy + do_hy * (ch_curl * fwd_x(Ez))
            # H CPML (x faces act on Hy, y faces on Hx).
            for n, maps in h_faces.items():
                if maps is None:
                    continue
                m, RA1, RB, RE, RF = maps
                if n in ("x0", "xm"):
                    dE = fwd_x(Ez) / dx
                    Hy = Hy + m * ch_rhs * (RA1 * dE + RB * phi_h[n])
                else:
                    dE = fwd_y(Ez) / dy
                    Hx = Hx - m * ch_rhs * (RA1 * dE + RB * phi_h[n])
                phi_h[n] = RE * phi_h[n] - RF * dE
            # E base update.
            Ez = Ez + do_ez * ((ce_hist - 1.0) * Ez + ce_curl * bwd_x(Hy) - ce_curl * ratio_y * bwd_y(Hx))
            for n, maps in e_faces.items():
                if maps is None:
                    continue
                m, RA1, RB, RE, RF = maps
                if n in ("x0", "xm"):
                    dH = bwd_x(Hy) / dx
                    Ez = Ez + m * ce_rhs * (RA1 * dH + RB * phi_e[n])
                else:
                    dH = bwd_y(Hx) / dy
                    Ez = Ez - m * ce_rhs * (RA1 * dH + RB * phi_e[n])
                phi_e[n] = RE * phi_e[n] - RF * dH
            # Source injection (Hertzian dipole along z, length dz) and receivers.
            for s, (i0, j0) in enumerate(src):
                w = wave[s if wave.shape[0] > 1 else 0, it, 0]
                onehot = torch.zeros((NXF, NYF), dtype=dtype)
                onehot[i0, j0] = 1.0
                Ez = Ez - onehot * ce_rhs[i0, j0] * w * dz / (dx * dy * dz)
            trace.append(Ez[rec[:, 0], rec[:, 1]])
        out.append(torch.stack(trace))
    return torch.stack(out)


def misfit(problem, data):
    return gv.misfit(problem, data)
