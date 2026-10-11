"""Fresh CPU-only checkpoint audit; uses synthetic inputs and no saved t05 data.

Run from the repository root with OMP_NUM_THREADS=1 and CUDA_VISIBLE_DEVICES=''.
Outputs are confined to this audit's data/checkpoint_results.json.
"""
from __future__ import annotations

import hashlib
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gv_common as gv
import torch
import DeepGPR
from DeepGPR.solver.autograd import STATE_NAMES
from DeepGPR.native.loader import get_deepgpr_lib
from torch.utils.checkpoint import checkpoint

SEED = 2026101019
OPTIONS = dict(model_gradient_sampling_interval=1,
               wavefield_storage_dtype=torch.float32,
               wavefield_rhs_history="stored")


def rel(a, b):
    a, b = a.double(), b.double()
    denom = float(torch.linalg.vector_norm(b))
    diff = float(torch.linalg.vector_norm(a - b))
    return diff / denom if denom else (0.0 if not diff else math.inf)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split(nt, k):
    q, r = divmod(nt, k)
    return [q + (i < r) for i in range(k)]


def initialize(problem, seeded=False):
    e, h, p = DeepGPR.checkpoint_initial_field(
        device="cpu", dx=problem.dx, dt=problem.dt,
        source_amplitudes=problem.wavelet,
        source_location=problem.src_loc, receiver_location=problem.rec_loc,
        er=problem.eps, se=problem.sig, pmlthick=problem.pml,
        fdtd_order=problem.order)
    states = list((*e, *h, *p))
    if seeded:
        rng = torch.Generator().manual_seed(SEED + 1)
        for i, state in enumerate(states):
            # Small nonzero all-component initial state; every 3D CPML face is active.
            scale = 1e-4 if i < 3 else (1e-7 if i < 6 else 1e-6)
            state.copy_(torch.randn(state.shape, generator=rng) * scale)
    return states


def solve(problem, eps, sig, source, initial, k=None, mode="plain",
          history=False, clone_boundaries=True, options=None, call_record=None):
    state = tuple(initial)
    options = OPTIONS if options is None else options

    def segment(e, s, wave, *boundary):
        if call_record is not None:
            call_record.append({"nt": wave.shape[1], "grad_enabled": torch.is_grad_enabled()})
        work = [t.clone() for t in boundary] if clone_boundaries else list(boundary)
        out = gv.compute(problem, "cpu", e, s, source=wave, states=work,
                         history=history, **(options if history else {}))
        return (*out[1], *out[2], *out[3], out[-1])

    if k is None:
        out = segment(eps, sig, source, *state)
        return out[-1], out[:-1]
    traces, start = [], 0
    for size in split(problem.nt, k):
        wave = source[:, start:start + size]
        start += size
        if mode == "plain":
            out = segment(eps, sig, wave, *state)
        else:
            out = checkpoint(segment, eps, sig, wave, *state,
                             use_reentrant=(mode == "reentrant"))
        state, trace = out[:-1], out[-1]
        traces.append(trace)
    return torch.cat(traces, 1), state


def terminal_weights(states):
    rng = torch.Generator().manual_seed(SEED + 2)
    weights = []
    for state in states:
        if state.numel():
            rms = max(float(state.double().square().mean().sqrt()), 1e-12)
            weights.append(torch.randn(state.shape, generator=rng).double()
                           * (3e-3 / rms / math.sqrt(state.numel())))
        else:
            weights.append(torch.empty_like(state, dtype=torch.float64))
    return weights


def objective(problem, data, states, weights):
    # Nonzero terminal cotangents exercise all 30 state outputs, including CPML.
    return gv.misfit(problem, data) + sum((s.double() * w).sum()
                                        for s, w in zip(states, weights))


def ad_run(problem, initial, weights, k=None, mode="plain", options=None,
           clone_boundaries=True):
    eps = problem.eps.float().detach().clone().requires_grad_()
    sig = problem.sig.float().detach().clone().requires_grad_()
    wave = problem.wavelet.detach().clone().requires_grad_()
    initial_leaves = [s.detach().clone().requires_grad_() for s in initial]
    frozen = [s.detach().clone() for s in initial_leaves]
    calls = []
    t0 = time.perf_counter()
    data, state = solve(problem, eps, sig, wave, initial_leaves, k, mode, True,
                        clone_boundaries, options, calls)
    frozen_final = [s.detach().clone() for s in state]
    loss = objective(problem, data, state, weights)
    loss.backward()
    return dict(J=float(loss), data=data.detach().clone(),
                state=frozen_final, eps=eps.grad.detach().clone(),
                sig=sig.grad.detach().clone(), source=wave.grad.detach().clone(),
                initial_grads=[s.grad.detach().clone() if s.grad is not None else None
                               for s in initial_leaves],
                inputs_unchanged=all(torch.equal(a, b) for a, b in zip(initial_leaves, frozen)),
                final_unchanged_by_backward=all(torch.equal(a, b) for a, b in zip(state, frozen_final)),
                calls=calls, seconds=time.perf_counter() - t0)


def compare(row, reference):
    states = [dict(name=name, shape=list(a.shape), bitwise=torch.equal(a, b),
                   relative_l2=rel(a, b), max_abs=float((a - b).abs().max()) if a.numel() else 0.)
              for name, a, b in zip(STATE_NAMES, row["state"], reference["state"])]
    gstates = []
    for name, a, b in zip(STATE_NAMES, row["initial_grads"], reference["initial_grads"]):
        gstates.append(dict(name=name, present=a is not None and b is not None,
                            relative_l2=rel(a, b) if a is not None and b is not None else None))
    return dict(data_bitwise=torch.equal(row["data"], reference["data"]),
                data_relative_l2=rel(row["data"], reference["data"]),
                all_30_states_bitwise=all(s["bitwise"] for s in states), states=states,
                eps_relative_l2=rel(row["eps"], reference["eps"]),
                sig_relative_l2=rel(row["sig"], reference["sig"]),
                source_gradient_relative_l2=rel(row["source"], reference["source"]),
                initial_state_gradients=gstates, J=row["J"],
                reference_J=reference["J"], inputs_unchanged=row["inputs_unchanged"],
                final_unchanged_by_backward=row["final_unchanged_by_backward"],
                calls=row["calls"], seconds=row["seconds"])


def fd_suite(problem, initial, weights, reference):
    """Central differences run forward only, with history disabled and no backward.

    All physical cells are eligible. A separate direction covers every outer
    physical-model face; no edge_mask or sensitivity-derived direction is used.
    """
    rng = torch.Generator().manual_seed(SEED + 3)
    shape = problem.eps.shape
    ring = torch.zeros(shape, dtype=torch.bool)
    for dim in range(len(shape)):
        sl = [slice(None)] * len(shape)
        sl[dim] = 0
        ring[tuple(sl)] = True
        sl[dim] = -1
        ring[tuple(sl)] = True
    rows = []
    for parameter in ("eps", "sig"):
        for region in ("entire_physical_model", "outer_physical_faces"):
            direction = torch.randn(shape, generator=rng, dtype=torch.float64)
            if region == "outer_physical_faces":
                direction = direction * ring
            direction /= direction.abs().max()
            scale = 4.0 if parameter == "eps" else 0.01
            direction *= scale
            gad = float((reference[parameter].double() * direction).sum())
            sweep = []
            for h in (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003):
                losses = []
                for sign in (-1, 1):
                    e, s = problem.eps.clone(), problem.sig.clone()
                    if parameter == "eps":
                        e += sign * h * direction
                    else:
                        s += sign * h * direction
                    with torch.no_grad():
                        data, states = solve(problem, e, s, problem.wavelet,
                                             [x.detach().clone() for x in initial],
                                             history=False)
                        losses.append(float(objective(problem, data, states, weights)))
                gfd = (losses[1] - losses[0]) / (2 * h)
                err = abs(gfd - gad) / max(abs(gfd), abs(gad), 1e-30)
                sweep.append(dict(relative_step=h, Jminus=losses[0], Jplus=losses[1],
                                  g_fd=gfd, relative_error=err))
            acceptable = sum(x["relative_error"] < 0.01 for x in sweep)
            best = min(x["relative_error"] for x in sweep)
            rows.append(dict(parameter=parameter, region=region,
                             active_cells=int(torch.count_nonzero(direction)),
                             total_physical_cells=direction.numel(),
                             g_ad=gad, sweep=sweep, best_relative_error=best,
                             verdict="PASS" if best < 0.005 and acceptable >= 2 else "FAIL",
                             criterion="best <0.5% and >=2 steps <1%; float32 native forward"))
            print("FD", parameter, region, "AD", gad, "best", best,
                  rows[-1]["verdict"], flush=True)
    return rows


def state_dot_test(problem, nt):
    """Independent linear-state transpose identity, using no source at all.

    LHS uses a forward-only solve. The material values and CPML coefficients
    are fixed in both calls; no material gradient enters this identity.
    """
    p = replace(problem, nt=nt, wavelet=torch.zeros(1, nt, 1))
    x = initialize(p, seeded=True)
    with torch.no_grad():
        ax_data, ax_state = solve(p, p.eps, p.sig, p.wavelet, x, history=False)
    rng = torch.Generator().manual_seed(SEED + 4 + nt)
    y_state = [torch.randn(s.shape, generator=rng).double() for s in ax_state]
    y_data = torch.randn(ax_data.shape, generator=rng).double()
    lhs = sum(float((a.double() * y).sum()) for a, y in zip(ax_state, y_state))
    lhs += float((ax_data.double() * y_data).sum())
    leaves = [s.detach().clone().requires_grad_() for s in x]
    data, states = solve(p, p.eps, p.sig, p.wavelet, leaves, history=True)
    functional = sum((a.double() * y).sum() for a, y in zip(states, y_state))
    functional += (data.double() * y_data).sum()
    functional.backward()
    rhs = sum(float((a.double() * leaf.grad.double()).sum()) for a, leaf in zip(x, leaves))
    denominator = max(abs(lhs), abs(rhs), 1e-30)
    error = abs(lhs - rhs) / denominator
    row = dict(nt=nt, order=p.order, model_shape=list(p.eps.shape), source_all_zero=True,
               all_30_initial_states_nonzero=all(bool(torch.count_nonzero(s)) for s in x),
               includes_all_30_terminal_states_and_receivers=True,
               lhs_forward_only=lhs, rhs_from_native_state_adjoint=rhs,
               relative_error=error, threshold=2e-5,
               all_30_initial_cotangents_present=all(s.grad is not None for s in leaves),
               state_inputs_preserved=all(torch.equal(a,b) for a,b in zip(x,leaves)),
               verdict="PASS" if error < 2e-5 else "FAIL")
    print("state_dot", nt, "lhs/rhs", lhs, rhs, "error", error, row["verdict"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="Save an independent checkpoint_results_TAG.json")
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[a-z0-9_]+", args.tag):
        raise ValueError("tag must contain only lowercase ASCII letters, digits and underscores")
    torch.set_num_threads(1)
    gv.seed_all(SEED)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("This audit requires OMP_NUM_THREADS=1")
    result = dict(environment=gv.environment(), command=" ".join(sys.argv),
                  executable=sys.executable, seed=SEED, exact_options={k: str(v) for k, v in OPTIONS.items()},
                  native_cpu_sha256=sha(gv.REPO / "src/DeepGPR/lib/deepgpr_cpu.so"),
                  cpu_only=True, cpu_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  tests=[])
    result["invocation"] = (
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
        + sys.executable + " " + str(Path(__file__).relative_to(gv.REPO))
        + (" --tag " + args.tag if args.tag else ""))
    result["environment"]["mkl_num_threads"] = os.environ.get("MKL_NUM_THREADS")
    result["environment"]["openblas_num_threads"] = os.environ.get("OPENBLAS_NUM_THREADS")
    result["environment"]["seed"] = SEED
    problem = gv.make_problem_3d("D", shape=(10, 11, 8), nt=151, pml=3)
    problem.mode = 3
    initial = initialize(problem, seeded=True)
    # Observations are freshly simulated from a perturbed model with the same
    # synthetic initial state, avoiding an irrelevant zero-state residual.
    de, ds = gv.observed_perturbation_3d(problem.eps.shape)
    with torch.no_grad():
        obs, _ = solve(problem, problem.eps + de, problem.sig + ds,
                       problem.wavelet, initial, history=False)
        problem.d_obs = obs.double()
        problem.scale = float(obs.abs().max())
        _, base_state = solve(problem, problem.eps, problem.sig,
                              problem.wavelet, initial, history=False)
    weights = terminal_weights(base_state)
    reference = ad_run(problem, initial, weights)
    result["synthetic_problem"] = dict(shape=list(problem.eps.shape), nt=problem.nt,
                                       pml=problem.pml, dx=problem.dx, dt=problem.dt,
                                       order=problem.order, mode=problem.mode,
                                       source_locations=problem.src_loc.tolist(),
                                       receiver_locations=problem.rec_loc.tolist(),
                                       initial_state_all_30_nonzero=all(bool(torch.count_nonzero(s)) for s in initial),
                                       objective="float64 receiver misfit plus random linear terminal functional on all 30 states",
                                       observation_scale=problem.scale)
    result["unsegmented_reference"] = dict(J=reference["J"],
                                          eps_norm=float(reference["eps"].double().norm()),
                                          sig_norm=float(reference["sig"].double().norm()),
                                          seconds=reference["seconds"],
                                          inputs_unchanged=reference["inputs_unchanged"])
    for k in (1, 2, 3, 5):
        for mode in ("plain", "reentrant", "nonreentrant"):
            actual = ad_run(problem, initial, weights, k=k, mode=mode)
            row = dict(kind="checkpoint", k=k, mode=mode, sizes=split(problem.nt, k),
                       **compare(actual, reference))
            worst_state_grad = max(x["relative_l2"] for x in row["initial_state_gradients"])
            row["verdict"] = "PASS" if (row["data_bitwise"] and row["all_30_states_bitwise"]
                and max(row["eps_relative_l2"], row["sig_relative_l2"],
                        row["source_gradient_relative_l2"], worst_state_grad) < 1e-5
                and row["inputs_unchanged"] and row["final_unchanged_by_backward"]) else "FAIL"
            result["tests"].append(row)
            print("checkpoint", k, mode, row["verdict"], "eps/sig",
                  row["eps_relative_l2"], row["sig_relative_l2"], flush=True)
    result["finite_differences"] = fd_suite(problem, initial, weights, reference)
    result["finite_difference_semantics"] = (
        "These finite differences differentiate the full public forward, including "
        "regenerated replicated PML materials and CPML coefficients. Production backward "
        "holds that numerical boundary fixed: docs/API.md line 131. A FAIL is retained "
        "as a failure of the full public-forward derivative, not converted to PASS "
        "by the fixed-PML convention. It is separate from checkpoint equivalence.")
    result["linear_state_dot_tests"] = [state_dot_test(problem, nt) for nt in (1, 5)]
    # Additional diagnostic only: removing external CPML makes the public
    # forward and fixed-boundary backward conventions coincide. This never
    # replaces or masks the PML finite-difference failures above.
    no_pml = gv.make_problem_3d("D", shape=(10, 11, 8), nt=151, pml=0)
    no_pml.mode = 3
    no_pml_initial = initialize(no_pml, seeded=True)
    de, ds = gv.observed_perturbation_3d(no_pml.eps.shape)
    with torch.no_grad():
        no_pml_obs, _ = solve(no_pml, no_pml.eps + de, no_pml.sig + ds,
                             no_pml.wavelet, no_pml_initial, history=False)
        no_pml.d_obs = no_pml_obs.double()
        no_pml.scale = float(no_pml_obs.abs().max())
        _, no_pml_terminal = solve(no_pml, no_pml.eps, no_pml.sig,
                                   no_pml.wavelet, no_pml_initial, history=False)
    no_pml_weights = terminal_weights(no_pml_terminal)
    no_pml_reference = ad_run(no_pml, no_pml_initial, no_pml_weights)
    result["no_external_pml_additional_diagnostic"] = dict(
        pml=0, does_not_replace_pml_failures=True,
        finite_differences=fd_suite(no_pml, no_pml_initial, no_pml_weights, no_pml_reference))
    # Negative control: omitting clones is deliberately unsafe. Record behavior,
    # without treating an expected corruption as production failure.
    try:
        unsafe = ad_run(problem, initial, weights, k=3, mode="reentrant", clone_boundaries=False)
        result["negative_control_without_clones"] = compare(unsafe, reference)
        result["negative_control_without_clones"]["expected_unsafe_behavior_observed"] = (
            not unsafe["inputs_unchanged"] or rel(unsafe["eps"], reference["eps"]) > 1e-5)
    except Exception as exc:
        result["negative_control_without_clones"] = dict(error=repr(exc), expected_unsafe_behavior_observed=True)
    history_configs = [
        ("fp32_auto", dict(wavefield_rhs_history="auto"), "exact"),
        ("fp32_reconstructed", dict(wavefield_rhs_history="reconstructed"), "exact"),
        ("fp32_physical_stored", dict(wavefield_history_region="physical"), "exact"),
        ("fp32_physical_reconstructed", dict(wavefield_history_region="physical", wavefield_rhs_history="reconstructed"), "exact"),
        ("fp16_stored", dict(wavefield_storage_dtype=torch.float16), "approximate"),
        ("bf16_stored", dict(wavefield_storage_dtype=torch.bfloat16), "approximate"),
        ("fp16_reconstructed", dict(wavefield_storage_dtype=torch.float16, wavefield_rhs_history="reconstructed"), "approximate"),
        ("bf16_reconstructed", dict(wavefield_storage_dtype=torch.bfloat16, wavefield_rhs_history="reconstructed"), "approximate"),
        ("sampling_2", dict(model_gradient_sampling_interval=2), "approximate"),
        ("sampling_3", dict(model_gradient_sampling_interval=3), "approximate"),
        ("async_requested_cpu", dict(use_async_offload=True), "cpu_noop"),
    ]
    for name, options, kind in history_configs:
        configured = dict(OPTIONS, **options)
        actual = ad_run(problem, initial, weights, options=configured)
        row = dict(kind="history", name=name, approximation=kind,
                   options={k: str(v) for k, v in configured.items()},
                   **compare(actual, reference))
        row["verdict"] = ("PASS" if row["data_bitwise"] and row["all_30_states_bitwise"]
                          and max(row["eps_relative_l2"],row["sig_relative_l2"]) < 1e-5
                          else "FAIL") if kind in ("exact", "cpu_noop") else "MEASURED_APPROXIMATION"
        result["tests"].append(row)
        print("history", name, row["verdict"], "eps/sig", row["eps_relative_l2"], row["sig_relative_l2"], flush=True)
    result["cpu_int8_validation"] = None
    try:
        ad_run(problem, initial, weights, options=dict(OPTIONS, wavefield_compression="int8"))
        result["cpu_int8_validation"] = dict(expected_rejection=False)
    except ValueError as exc:
        result["cpu_int8_validation"] = dict(expected_rejection=True, message=str(exc))
    library = get_deepgpr_lib("cpu")
    symbols = ("deepgpr_supports_external_pml", "deepgpr_supports_rhs_reconstruction",
               "deepgpr_supports_physical_history", "deepgpr_supports_int8_wavefield")
    result["cpu_capabilities"] = {s: int(getattr(library,s)()) if hasattr(library,s) else None for s in symbols}
    result["run_tag"] = args.tag
    result["script_sha256"] = sha(Path(__file__))
    result["failure_counts"] = dict(
        checkpoint=sum(r["verdict"]=="FAIL" for r in result["tests"] if r["kind"]=="checkpoint"),
        exact_storage=sum(r["verdict"]=="FAIL" for r in result["tests"] if r["kind"]=="history"),
        public_material_fd=sum(r["verdict"]=="FAIL" for r in result["finite_differences"]),
        state_dot=sum(r["verdict"]=="FAIL" for r in result["linear_state_dot_tests"]),
        no_external_pml_diagnostic_fd=sum(r["verdict"]=="FAIL" for r in result["no_external_pml_additional_diagnostic"]["finite_differences"]))
    filename = "checkpoint_results" + ("_" + args.tag if args.tag else "") + ".json"
    path = HERE / "data" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gv.to_jsonable(result), indent=2) + "\n")
    print("wrote", path, flush=True)
    failures = [r for r in result["tests"] if r["verdict"] == "FAIL"]
    failures += [r for r in result["finite_differences"] if r["verdict"] == "FAIL"]
    failures += [r for r in result["linear_state_dot_tests"] if r["verdict"] == "FAIL"]
    print("independent audit failures", len(failures), flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
