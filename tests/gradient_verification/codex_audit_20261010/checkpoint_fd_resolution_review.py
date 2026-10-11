"""CPU-only resolution review of the retained weak no-PML sigma-face FD FAIL.

Never edits the original checkpoint result file or changes its raw verdict.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import checkpoint_verification as cp
gv, torch = cp.gv, cp.torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(review):
    """Preserve raw FAIL; report resolution evidence separately from correctness.

    The endpoint ULP residual envelope is an empirical scale, not a certified
    upper bound or a new pass threshold. Neighbor sweeps only diagnose whether
    an error plateau exists; their favorable h values never replace the raw test.
    """
    budgets = []
    for row in review["h_repeats"]:
        h = row["h"]
        endpoint_rows = [x for x in review["endpoint_ulp_probes"] if x["h"] == h]
        envelope = sum(x["max_abs_residual"] for x in endpoint_rows)
        first = row["repeats"][0]
        budgets.append(dict(
            h=h, absolute_derivative_error=first["absolute_derivative_error"],
            objective_numerator_discrepancy=first["objective_numerator_discrepancy"],
            endpoint_one_ulp_empirical_residual_sum=envelope,
            empirical_derivative_resolution=envelope/(2*h),
            discrepancy_to_empirical_resolution_ratio=(
                first["objective_numerator_discrepancy"]/envelope if envelope else None),
            original_values_reproduced=row["original_values_reproduced"],
            repeat_loss_range=row["repeat_loss_range"]))
    neighbors = []
    for row in review["h_repeats"]:
        h = row["h"]
        errors = [x["signed_derivative_error"] for x in review["h_neighbors"]
                  if x["center_h"] == h]
        errors.append(row["repeats"][0]["g_fd"]-review["fresh_g_ad"])
        neighbors.append(dict(center_h=h, signed_error_min=min(errors),
                              signed_error_max=max(errors),
                              signed_error_changes_sign=min(errors)<0<max(errors)))
    reproduced = review["fresh_gradient_identical"] and all(
        x["original_values_reproduced"] for x in budgets)
    unresolved_by_resolution = all(
        x["objective_numerator_discrepancy"] <=
        x["endpoint_one_ulp_empirical_residual_sum"] for x in budgets)
    no_stable_sign = any(x["signed_error_changes_sign"] for x in neighbors)
    review["classification"] = ("INCONCLUSIVE_FLOAT32_RESOLUTION"
        if reproduced and unresolved_by_resolution and no_stable_sign
        else "FAIL_UNRESOLVED")
    review["resolution_review"] = dict(
        raw_verdict_preserved=review["original_raw_verdict"],
        raw_criterion_preserved=review["original_criterion"],
        does_not_award_pass=True,
        h_absolute_error_and_resolution=budgets,
        neighboring_h_signed_errors=neighbors,
        interpretation=(
            "The unchanged weak directional derivative is not resolved reliably "
            "enough by this float32 forward to confirm or refute a defect at the "
            "original relative tolerance. Exact repeated-input determinism gives "
            "zero run-to-run noise, but one-ULP material probes reveal comparable "
            "material-dependent forward quantization, and neighboring h changes "
            "the error sign. Original raw FAIL is retained; no PASS is inferred."),
        limitations=(
            "The sum of observed endpoint one-ULP residual maxima is an empirical "
            "resolution indicator, not a rigorous roundoff bound or statistical "
            "confidence interval. Linear material effects use baseline AD and are "
            "tiny compared with observed jumps, especially face-only probes. "
            "All-plus/random probes include interior cells and measure the local "
            "objective resolution rather than changing the FD direction. No "
            "claim that the production gradient is correct follows; higher "
            "forward precision or an independent analytical oracle is required "
            "to settle this weak derivative. Existing PML FAILs are unaffected."))


def main():
    torch.set_num_threads(1)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("requires OMP_NUM_THREADS=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This review is CPU-only; set CUDA_VISIBLE_DEVICES='' explicitly")
    gv.seed_all(cp.SEED)
    original_path = HERE / "data/checkpoint_results_after_native_fix.json"
    original = json.loads(original_path.read_text())
    original_row = next(r for r in original["no_external_pml_additional_diagnostic"]["finite_differences"]
                        if r["parameter"] == "sig" and r["region"] == "outer_physical_faces")
    p = gv.make_problem_3d("D", shape=(10, 11, 8), nt=151, pml=0)
    p.mode = 3
    initial = cp.initialize(p, seeded=True)
    de, ds = gv.observed_perturbation_3d(p.eps.shape)
    with torch.no_grad():
        obs, _ = cp.solve(p, p.eps + de, p.sig + ds, p.wavelet, initial, history=False)
        p.d_obs = obs.double()
        p.scale = float(obs.abs().max())
        _, base_state = cp.solve(p, p.eps, p.sig, p.wavelet, initial, history=False)
    weights = cp.terminal_weights(base_state)
    ad = cp.ad_run(p, initial, weights)
    # Reproduce the exact RNG consumption of cp.fd_suite, rather than choosing
    # a direction with a larger derivative after seeing the retained failure.
    rng = torch.Generator().manual_seed(cp.SEED + 3)
    ring = torch.zeros(p.eps.shape, dtype=torch.bool)
    for dim in range(p.eps.dim()):
        sl = [slice(None)] * p.eps.dim()
        sl[dim] = 0
        ring[tuple(sl)] = True
        sl[dim] = -1
        ring[tuple(sl)] = True
    direction = None
    for parameter in ("eps", "sig"):
        for region in ("all", "faces"):
            d = torch.randn(p.eps.shape, generator=rng, dtype=torch.float64)
            if region == "faces":
                d *= ring
            d /= d.abs().max()
            d *= 4. if parameter == "eps" else .01
            if parameter == "sig" and region == "faces":
                direction = d
    gad = float((ad["sig"].double() * direction).sum())

    def forward_loss(e, s):
        # Same fixed state and unchanged J; only native forward is used here.
        with torch.no_grad():
            data, states = cp.solve(p, e, s, p.wavelet,
                                    [x.detach().clone() for x in initial], history=False)
            return float(cp.objective(p, data, states, weights))

    ebase, sbase = p.eps.float(), p.sig.float()
    jbase = forward_loss(ebase, sbase)
    environment = gv.environment()
    environment["seed"] = cp.SEED
    review = dict(environment=environment, executable=sys.executable,
                  invocation="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
                  + sys.executable + " " + str(Path(__file__).relative_to(gv.REPO)),
                  input_path=str(original_path), input_sha256=sha(original_path),
                  script_sha256=sha(__file__), native_cpu_sha256=sha(gv.REPO / "src/DeepGPR/lib/deepgpr_cpu.so"),
                  original_native_cpu_sha256=original["native_cpu_sha256"],
                  original_raw_verdict=original_row["verdict"], original_criterion=original_row["criterion"],
                  original_g_ad=original_row["g_ad"], fresh_g_ad=gad,
                  fresh_gradient_identical=(gad == original_row["g_ad"]),
                  baseline_J=jbase, direction=direction.tolist(),
                  unchanged_objective="same receiver misfit + fixed linear terminal functional over all 30 states",
                  h_repeats=[], h_neighbors=[], material_ulp_probes=[], endpoint_ulp_probes=[])
    for h in (.1, .03, .01, .003):
        repeated=[]
        for _ in range(3):
            minus=forward_loss(p.eps, p.sig - h * direction)
            plus=forward_loss(p.eps, p.sig + h * direction)
            gfd=(plus-minus)/(2*h)
            absolute_error=abs(gfd-gad)
            repeated.append(dict(Jminus=minus,Jplus=plus,g_fd=gfd,
                                 absolute_derivative_error=absolute_error,
                                 relative_derivative_error=absolute_error/max(abs(gfd),abs(gad),1e-30),
                                 objective_numerator_discrepancy=abs((plus-minus)-2*h*gad)))
        old = next(x for x in original_row["sweep"] if x["relative_step"]==h)
        row=dict(h=h,repeats=repeated,
                 original_values_reproduced=(repeated[0]["Jminus"]==old["Jminus"]
                                             and repeated[0]["Jplus"]==old["Jplus"]),
                 repeat_loss_range=max(x["Jminus"] for x in repeated)-min(x["Jminus"] for x in repeated)
                                   +max(x["Jplus"] for x in repeated)-min(x["Jplus"] for x in repeated))
        review["h_repeats"].append(row)
        print("repeat",h,"gfd",repeated[0]["g_fd"],"abs_error",repeated[0]["absolute_derivative_error"],
              "relative",repeated[0]["relative_derivative_error"],"reproduced",row["original_values_reproduced"],flush=True)
        # Symmetric +/-20% neighboring h, no changed direction/objective.
        for factor in (.8,.9,1.1,1.2):
            hn=h*factor
            jm=forward_loss(p.eps,p.sig-hn*direction)
            jp=forward_loss(p.eps,p.sig+hn*direction)
            gfd=(jp-jm)/(2*hn)
            review["h_neighbors"].append(dict(h=hn,center_h=h,Jminus=jm,Jplus=jp,g_fd=gfd,
                 absolute_derivative_error=abs(gfd-gad),signed_derivative_error=gfd-gad,
                 relative_derivative_error=abs(gfd-gad)/max(abs(gfd),abs(gad),1e-30)))

    probe_rng = torch.Generator().manual_seed(2026101027)
    patterns = {"all_plus":torch.ones_like(sbase),
                "random_sign":torch.where(torch.rand(sbase.shape,generator=probe_rng)>.5,
                                          torch.ones_like(sbase),-torch.ones_like(sbase)),
                "same_sigma_direction_sign_faces":direction.sign().float()}

    def ulp_shift(value,pattern,steps,sign):
        work=value.clone()
        toward=torch.where(pattern*sign>0,torch.full_like(value,float("inf")),
                           torch.full_like(value,-float("inf")))
        active=pattern != 0
        for _ in range(steps):
            work=torch.where(active,torch.nextafter(work,toward),work)
        return work

    def probe(e,s,j0,parameter,pattern_name,pattern,steps,sign):
        actual=ulp_shift(e if parameter=="eps" else s,pattern,steps,sign)
        ep,sp=(actual,s) if parameter=="eps" else (e,actual)
        value=forward_loss(ep,sp)
        delta=(actual-(e if parameter=="eps" else s)).double()
        predicted=float((ad[parameter].double()*delta).sum())
        return dict(parameter=parameter,pattern=pattern_name,ulp_steps=steps,sign=sign,
                    J=value,objective_jump=value-j0,material_delta_max=float(delta.abs().max()),
                    ad_linear_material_effect=predicted,
                    residual_after_linear_prediction=(value-j0)-predicted)

    # Baseline +/- one native float32 material ULP, with both material types.
    # More ULPs are diagnostic only: sigma ULPs can leave float32 update
    # coefficients unchanged until a quantization threshold is crossed.
    for parameter in ("eps","sig"):
        for name,pattern in patterns.items():
            for steps in (1,2,4,8,16,32,64,128):
                for sign in (-1,1):
                    review["material_ulp_probes"].append(probe(ebase,sbase,jbase,parameter,
                                                              name,pattern,steps,sign))
    # Objective resolution is assessed at both actual +/-h endpoints. Only
    # one-ULP sigma probes enter the primary finite-difference error budget;
    # epsilon probes and larger sigma ULP scans remain separate diagnostics.
    for h in (.1,.03,.01,.003):
        for endpoint_sign in (-1,1):
            s=(p.sig+endpoint_sign*h*direction).float()
            j0=forward_loss(ebase,s)
            probes=[]
            for name,pattern in patterns.items():
                for sign in (-1,1):
                    probes.append(probe(ebase,s,j0,"sig",name,pattern,1,sign))
            review["endpoint_ulp_probes"].append(dict(h=h,endpoint_sign=endpoint_sign,Jbaseline=j0,probes=probes,
                max_abs_objective_jump=max(abs(x["objective_jump"]) for x in probes),
                max_abs_residual=max(abs(x["residual_after_linear_prediction"]) for x in probes)))

    # No relative tolerance is altered; raw FAIL and derived review coexist.
    classify(review)
    out=HERE/"data/checkpoint_fd_resolution_review.json"
    out.write_text(json.dumps(gv.to_jsonable(review),indent=2)+"\n")
    for row in review["endpoint_ulp_probes"]:
        print("ULP endpoint",row["h"],row["endpoint_sign"],"jump",row["max_abs_objective_jump"],
              "residual",row["max_abs_residual"],flush=True)
    print("wrote",out,flush=True)


if __name__ == "__main__":
    main()
