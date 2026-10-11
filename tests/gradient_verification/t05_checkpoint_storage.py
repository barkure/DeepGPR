"""Section 7: time checkpointing (segments) and wavefield-history options.

Part A, segments. DeepGPR has no internal time checkpointing; callers split the
source into segments, pass the 30 E/H/CPML states between calls and wrap each
call in torch.utils.checkpoint (cloning the states first, the solver advances
them in place), as tests/checkpoint_example.ipynb. For K = 1 ... 40 segments
(uneven lengths included), reentrant and non-reentrant checkpointing, and
without checkpointing (plain graph): receiver data and final states must be
bitwise equal to the unsegmented run; gradients equal up to float32 summation
order; the segmented AD is also checked against FD of the forward.

Part B, history options against the exact AD (S = 1, float32, stored R,
extended history, no offload): E-only ("auto"/"reconstructed"), physical
region, S = 2, 3, "auto", above the bound, fp16 / bf16 (E+R and E-only),
INT8 (E+R and E-only), async offload, CUDA fast vs general path. Exact
configurations are also FD-checked; approximate ones are compared with the
exact AD and with the documented error magnitudes. CUDA peak memory (PyTorch
allocator only) and wall time are recorded.

Usage: python t05_checkpoint_storage.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402
import DeepGPR  # noqa: E402

# Documented magnitudes (relative L2 vs float32): tests/profiling_results/history_adjoint_gpu_report.md
# section 0 (2D benchmark) and docs/API.md section 4.4.
DOCUMENTED = {"fp16": 2.4e-5, "bf16": 1.6e-4, "int8": 4.5e-4}            # 2D benchmark, worst of eps/sigma
DOCUMENTED_3D = {"fp16": 4.4e-4, "bf16": 3.9e-3, "int8": 3.2e-2}         # 3D benchmark, worst of eps/sigma


def split_sizes(nt, k):
    base, extra = divmod(nt, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def segmented(problem, path, eps, sig, k, mode):
    """Receiver data and final states of a K-segment run (mode: reentrant / nonreentrant / plain)."""
    device = gv.path_device(path)
    E, H, PML = DeepGPR.checkpoint_initial_field(
        device=device, dx=problem.dx, dt=problem.dt, source_amplitudes=problem.wavelet.to(device),
        source_location=problem.src_loc.to(device), receiver_location=problem.rec_loc.to(device),
        er=eps.detach(), se=sig.detach(), pmlthick=problem.pml, fdtd_order=problem.order)
    state = (*E, *H, *PML)

    def segment(e, s, wave, *boundary):
        work = [t.clone() for t in boundary]
        out = gv.compute(problem, path, e, s, source=wave, states=work, history=e.requires_grad or s.requires_grad)
        return (*out[1], *out[2], *out[3], out[-1])

    traces, start = [], 0
    for size in split_sizes(problem.nt, k):
        wave = problem.wavelet[:, start:start + size].to(device)
        start += size
        if mode == "plain":
            outs = segment(eps, sig, wave, *state)
        else:
            outs = checkpoint(segment, eps, sig, wave, *state, use_reentrant=(mode == "reentrant"))
        state, data = outs[:-1], outs[-1]
        traces.append(data)
    return torch.cat(traces, dim=1), state


def part_a(paths, results):
    problem = gv.make_problem_2d("D", nt=400)
    rng = torch.Generator().manual_seed(42)
    d_eps = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    d_sig = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    s_eps, s_sig = float(problem.eps.mean()), float(problem.sig.mean())
    for path in paths:
        device = gv.path_device(path)
        full = gv.objective_ad(problem, path, problem.eps, problem.sig)
        ref_states = gv.compute(problem, path, problem.eps, problem.sig)
        ref_states = [t.detach().cpu() for t in (*ref_states[1], *ref_states[2], *ref_states[3])]
        for k in (1, 2, 3, 4, 5, 7, 13, 40):
            for mode in (("reentrant", "nonreentrant", "plain") if k in (1, 4, 7) else ("reentrant",)):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                eps = problem.eps.to(device=device, dtype=torch.float32).requires_grad_(True)
                sig = problem.sig.to(device=device, dtype=torch.float32).requires_grad_(True)
                data, state = segmented(problem, path, eps, sig, k, mode)
                J = gv.misfit(problem, data)
                J.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None
                ge, gs = eps.grad.double().cpu(), sig.grad.double().cpu()
                row = dict(part="segments", path=path, k=k, mode=mode, sizes=split_sizes(problem.nt, k),
                           data_bitwise=bool(torch.equal(data.detach().double().cpu(), full.data)),
                           states_bitwise=all(torch.equal(a.detach().cpu(), b) for a, b in zip(state, ref_states)),
                           eps_rel_to_unsegmented=gv.relative_l2(ge, full.grad_eps),
                           sig_rel_to_unsegmented=gv.relative_l2(gs, full.grad_sig),
                           J=float(J), J_full=full.J, peak_mib=peak, seconds=elapsed)
                # FD of the (identical) forward against the segmented AD.
                ad = gv.ADResult(J=float(J), grad_eps=ge, grad_sig=gs, grad_src=None, grad_states=None,
                                 tm2d_fast_path=None, data=data.detach().double().cpu())
                chk = gv.physical_check(problem, path, "random", d_eps, d_sig, ad, steps=gv.STEPS)
                row.update(fd_verdict=chk["verdict"], fd_rel_err=chk["rel_err"], fd_slope=chk["fit_slope"],
                           g_ad=chk["g_ad"], g_fd=chk["g_fd"])
                ok = row["data_bitwise"] and row["states_bitwise"] and max(
                    row["eps_rel_to_unsegmented"], row["sig_rel_to_unsegmented"]) < 1e-5 and chk["verdict"].startswith("PASS")
                row["verdict"] = "PASS" if ok else "FAIL"
                results.append(row)
                print(f"segments {path} K={k} {mode}: data bitwise {row['data_bitwise']} states bitwise "
                      f"{row['states_bitwise']} grad rel {row['eps_rel_to_unsegmented']:.1e}/"
                      f"{row['sig_rel_to_unsegmented']:.1e} FD {chk['verdict']} ({chk['rel_err']:.1e}) "
                      f"peak {peak} MiB {elapsed:.2f}s -> {row['verdict']}", flush=True)


def history_configs(path, bound):
    base = dict(model_gradient_sampling_interval=1, wavefield_storage_dtype=torch.float32)
    cfgs = [
        ("exact (stored, fp32, S=1)", dict(base, wavefield_rhs_history="stored"), "exact"),
        ("E-only auto (fp32)", dict(base, wavefield_rhs_history="auto"), "exact"),
        ("E-only reconstructed (fp32)", dict(base, wavefield_rhs_history="reconstructed"), "exact"),
        ("physical, E-only (fp32)", dict(base, wavefield_history_region="physical"), "exact"),
        ("physical, stored (fp32)", dict(base, wavefield_history_region="physical", wavefield_rhs_history="stored"), "exact"),
        ("S=2", dict(base, model_gradient_sampling_interval=2), "sampling"),
        ("S=3", dict(base, model_gradient_sampling_interval=3), "sampling"),
        ("S=auto", dict(base, model_gradient_sampling_interval="auto"), "sampling"),
        (f"S={2 * bound + 1} (above bound {bound})", dict(base, model_gradient_sampling_interval=2 * bound + 1), "sampling"),
        ("fp16 (E+R)", dict(base, wavefield_storage_dtype=torch.float16), "fp16"),
        ("bf16 (E+R)", dict(base, wavefield_storage_dtype=torch.bfloat16), "bf16"),
        ("fp16 E-only", dict(base, wavefield_storage_dtype=torch.float16, wavefield_rhs_history="reconstructed"), "fp16"),
        ("bf16 E-only", dict(base, wavefield_storage_dtype=torch.bfloat16, wavefield_rhs_history="reconstructed"), "bf16"),
    ]
    if path != "cpu":
        cfgs += [
            ("int8 (E+R)", dict(base, wavefield_compression="int8"), "int8"),
            ("int8 E-only", dict(base, wavefield_compression="int8", wavefield_rhs_history="reconstructed"), "int8"),
            ("int8 physical", dict(base, wavefield_compression="int8", wavefield_history_region="physical"), "int8"),
            ("async offload (fp32 E-only)", dict(base, use_async_offload=True), "exact"),
            ("async offload (fp32 stored)", dict(base, use_async_offload=True, wavefield_rhs_history="stored"), "exact"),
            ("async offload (fp16)", dict(base, use_async_offload=True, wavefield_storage_dtype=torch.float16), "fp16"),
            ("async + physical (fp32)", dict(base, use_async_offload=True, wavefield_history_region="physical"), "exact"),
        ]
    return cfgs


def part_b(paths, results, dims=2):
    problem = gv.make_problem_2d("D", nt=400) if dims == 2 else gv.make_problem_3d("D")
    bound = DeepGPR.recommended_sampling_interval(problem.wavelet, problem.dt)
    rng = torch.Generator().manual_seed(77)
    d_eps = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    d_sig = gv.directions(problem.eps.shape, problem.pml, "random", rng)
    s_eps, s_sig = float(problem.eps.mean()), float(problem.sig.mean())
    for path in paths:
        device = gv.path_device(path)
        exact = None
        for name, options, kind in history_configs(path, bound):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            ad = gv.objective_ad(problem, path, problem.eps, problem.sig, **options)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None
            if exact is None:
                exact = ad
            row = dict(part="history", dims=dims, path=path, config=name, kind=kind, options={k: str(v) for k, v in options.items()},
                       tm2d_fast_path=ad.tm2d_fast_path, data_bitwise_vs_exact=bool(torch.equal(ad.data, exact.data)),
                       eps_rel_l2=gv.relative_l2(ad.grad_eps, exact.grad_eps),
                       sig_rel_l2=gv.relative_l2(ad.grad_sig, exact.grad_sig),
                       eps_cos=gv.cosine(ad.grad_eps, exact.grad_eps), peak_mib=peak, seconds=elapsed,
                       sampling_bound=bound)
            g = float((ad.grad_eps * d_eps * s_eps).sum() + (ad.grad_sig * d_sig * s_sig).sum())
            g0 = float((exact.grad_eps * d_eps * s_eps).sum() + (exact.grad_sig * d_sig * s_sig).sum())
            row["directional_rel_err_vs_exact"] = abs(g - g0) / abs(g0)
            if kind == "exact":
                # Forward-only runs cannot carry history options (e.g. async offload); the
                # forward is bitwise identical across them (data_bitwise_vs_exact).
                chk = gv.physical_check(problem, path, "random", d_eps, d_sig, ad, steps=gv.STEPS)
                row.update(fd_verdict=chk["verdict"], fd_rel_err=chk["rel_err"], fd_slope=chk["fit_slope"])
                limit = 1e-5 if path != "cpu" else 1e-6
                ok = row["data_bitwise_vs_exact"] and max(row["eps_rel_l2"], row["sig_rel_l2"]) <= limit and chk["verdict"].startswith("PASS")
                row["verdict"] = "PASS" if ok else "FAIL"
            elif kind in DOCUMENTED:
                doc = (DOCUMENTED if dims == 2 else DOCUMENTED_3D)[kind] * (3.5 if "E-only" in name else 1.0)
                worst = max(row["eps_rel_l2"], row["sig_rel_l2"])
                row["documented"] = doc
                row["verdict"] = ("PASS (approximation within documented magnitude)" if worst <= 10 * doc
                                  else "OUT OF DOCUMENTED MAGNITUDE")
                row["verdict"] = row["verdict"] if row["data_bitwise_vs_exact"] else "FAIL (forward changed)"
            else:
                row["verdict"] = "MEASURED (approximation)" if row["data_bitwise_vs_exact"] else "FAIL (forward changed)"
            results.append(row)
            print(f"history {dims}D {path:12s} {name:34s} data bitwise {row['data_bitwise_vs_exact']} "
                  f"eps {row['eps_rel_l2']:.2e} sig {row['sig_rel_l2']:.2e} dir {row['directional_rel_err_vs_exact']:.1e} "
                  f"peak {peak if peak is None else round(peak, 1)} MiB {elapsed:.2f}s "
                  f"{row.get('fd_verdict', '')} -> {row['verdict']}", flush=True)


def main():
    gv.seed_all()
    paths = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    results = []
    part_a(paths, results)
    part_b(paths, results, dims=2)
    part_b(["cpu"] + (["cuda_general"] if gv.cuda_available() else []), results, dims=3)
    gv.save_json("t05_checkpoint_storage.json", dict(results=results, documented=DOCUMENTED,
                                                     documented_3d=DOCUMENTED_3D))
    print("verdicts", {v: sum(r["verdict"] == v for r in results) for v in sorted({r["verdict"] for r in results})})


if __name__ == "__main__":
    main()
