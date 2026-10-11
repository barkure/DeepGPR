"""Independent active/frozen physical-grid FWI verification; CPU only.

Reuse this audit directory's actual loss/propagation/strict FD helpers without
executing its 36 network cases. Inactive physical tensors are requires_grad=False,
omitted from the optimizer/AD parameter list and held fixed in every FD call.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode=True
import torch
from torch import nn
import numpy as np
import network_verification as audit

AUDIT=Path(__file__).resolve().parent
REPO=audit.REPO
COMMAND=("OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
         "/home/ll/anaconda3/envs/deepgpr/bin/python "
         "tests/gradient_verification/codex_audit_20261010/grid_fwi_freeze_verification.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()


def clone_only_active_step(physical,parameters,gradients):
    """Actually update active clones only; include frozen clones in the evidence."""
    active_names=[name for name,_ in parameters]
    before={name:value.detach().clone() for name,value in physical.items()}
    clones={name:nn.Parameter(value.clone()) if name in active_names else value.clone()
            for name,value in before.items()}
    largest_gradient=max(float(g.abs().max()) for g in gradients)
    scale=max(1.,max(float(before[name].abs().max()) for name in active_names))
    target=1e-10*scale
    lr=min(1e-8,target/max(largest_gradient,1e-30))
    for (name,_),gradient in zip(parameters,gradients):clones[name].grad=gradient.clone()
    optimizer=torch.optim.SGD([clones[name] for name in active_names],lr=lr)
    optimizer.step()
    by_name=dict(zip(active_names,gradients))
    records=[]
    for name,clone in clones.items():
        active=name in active_names
        actual=clone.detach();base=before[name];delta=actual-base
        gradient=by_name.get(name)
        expected=base-lr*gradient if active else base
        tolerance=2*torch.finfo(base.dtype).eps*max(1.,float(base.abs().max())) if active else 0.
        difference=float((actual-expected).abs().max())
        record={"name":name,"active":active,"included_in_optimizer":active,
                "clone_requires_grad":clone.requires_grad,"clone_grad_none":clone.grad is None,
                "original_requires_grad":physical[name].requires_grad,
                "original_unchanged":torch.equal(physical[name].detach(),base),
                "changed_elements":int(delta.count_nonzero()),"numel":base.numel(),
                "finite":bool(torch.isfinite(actual).all()),
                "max_abs_update":float(delta.abs().max()),"update_norm":float(delta.norm()),
                "update_dot_gradient":float((delta*gradient).sum()) if active else None,
                "max_error_vs_expected":difference,"rounding_tolerance":tolerance,
                "matches_expected":difference<=tolerance,
                "before_sha256":tensor_digest(base),"after_sha256":tensor_digest(actual)}
        record["frozen_bitwise_unchanged"]=torch.equal(actual,base) if not active else None
        records.append(record)
    active_ok=all(r["matches_expected"] and r["changed_elements"]>0 and
                  r["update_dot_gradient"]<=0 for r in records if r["active"])
    frozen_ok=all(r["frozen_bitwise_unchanged"] and not r["clone_requires_grad"] and
                  r["clone_grad_none"] and not r["included_in_optimizer"]
                  for r in records if not r["active"])
    return {"optimizer":"torch.optim.SGD.step()","clones_only":True,
            "gradient_source":"actual fresh combined total_loss.backward()",
            "learning_rate":lr,"target_max_abs_update":target,"optimizer_parameter_names":active_names,
            "loss_descent_evaluated":False,"parameters":records,
            "only_active_clones_updated":active_ok and frozen_ok,
            "all_originals_unchanged":all(r["original_unchanged"] for r in records)}


def run_case(coords,args,pml,obs,mode,kind):
    x=coords.detach().reshape(*audit.SHAPE,2)
    initial={"eps":4.+.07*x[...,0]+.035*x[...,1]+.008*torch.sin(x[...,0]*2),
             "sigma":.011+.0002*x[...,0]+.0001*x[...,1]+.00003*torch.sin(x[...,1]*2)}
    active_names=(["eps"] if mode=="epsilon_only" else ["sigma"] if mode=="sigma_only"
                  else ["eps","sigma"])
    physical={name:value.clone().requires_grad_(name in active_names) for name,value in initial.items()}
    parameters=[(name,physical[name]) for name in active_names]
    frozen={name:value.detach().clone() for name,value in physical.items() if name not in active_names}

    def fields(active_parameters=None):
        # A caller may supply a replacement active vector. Frozen tensors are
        # captured constants and are never made trainable or perturbed by FD.
        current=dict(parameters if active_parameters is None else active_parameters)
        current.update(frozen)
        return current["eps"],current["sigma"]

    fixed=tuple(value.detach().clone() for value in fields())
    J0,gradients,records,comparison=audit.ad_terms(parameters,fields,fixed,mode,kind,pml,obs,args)
    step=clone_only_active_step(physical,parameters,gradients[2])
    resolution_seed=audit.SEED+int(hashlib.sha256(f"Grid-FWI-freeze,{pml},{mode},{kind},resolution".encode()).hexdigest()[:7],16)
    resolution=audit.measure_forward_resolution(fixed,mode,kind,pml,obs,args,J0,resolution_seed)
    base=[p.detach().clone() for _,p in parameters]
    groups=("epsilon",) if mode=="epsilon_only" else ("sigma",) if mode=="sigma_only" else ("epsilon","sigma","joint")
    directions=[]
    for group in groups:
        seed=audit.SEED+int(hashlib.sha256(f"Grid-FWI-freeze,{pml},{mode},{kind},{group}".encode()).hexdigest()[:7],16)
        vector=audit.direction(parameters,group,seed)
        checks=audit.check_direction(parameters,base,vector,J0,gradients,fields,fixed,mode,kind,pml,obs,args,resolution)
        directions.append({"group":group,"seed":seed,"direction_norm":audit.norm(vector),
                           "active_parameter_names":active_names,"edge_mask_applied":False,"checks":checks})
        print(f"Grid-FWI-freeze pml={pml} {mode} {kind} {group}: "+
              ", ".join(f"{r['term']}={r['status']}" for r in checks),flush=True)
    frozen_records=[]
    for name,value in physical.items():
        if name in active_names:continue
        frozen_records.append({"name":name,"requires_grad":value.requires_grad,"grad_none":value.grad is None,
                               "held_fixed_in_fields":torch.equal(fields()[0 if name=="eps" else 1],initial[name]),
                               "bitwise_unchanged_after_all_FD":torch.equal(value,initial[name]),
                               "original_sha256":tensor_digest(value),"fixed_sha256":tensor_digest(frozen[name])})
    return {"name":"Grid-FWI-freeze","mode":mode,"pml":pml,"objective":kind,
            "active_parameter_names":active_names,"inactive_parameter_names":list(frozen),
            "physical_requires_grad":{name:value.requires_grad for name,value in physical.items()},
            "active_parameter_count":sum(p.numel() for _,p in parameters),
            "J0":dict(zip(("data","tv","total"),J0)),"parameter_dtype":"float64","native_dtype":"float32",
            "gradient_records":records,"combined_backward_comparison":comparison,"clone_only_SGD":step,
            "frozen_checks":frozen_records,"resolution":resolution,"directions":directions,
            "active_gradients_present_and_finite":all(not r["none"] and not r["nonfinite"] for r in records),
            "inactive_physical_tensors_fixed":all(not r["requires_grad"] and r["grad_none"] and
                                                   r["held_fixed_in_fields"] and r["bitwise_unchanged_after_all_FD"]
                                                   for r in frozen_records)}


def write_report(data):
    lines=["# Active/frozen physical-grid FWI audit","",
           "This independent addendum closes the difference between an epsilon-only/sigma-only "
           "**perturbation direction** in a joint model and an epsilon-only/sigma-only **optimization mode**. "
           "The preceding 36 network cases remain unchanged. This script executes 12 actual grid "
           "cases: epsilon_only, sigma_only and joint × PML 0/6 × L1×1e7+TV/MSE×1e4+TV.","",
           "It imports only the sibling audit's public CPU propagation, actual example loss and strict "
           "FD helpers; importing that module does not execute its network audit. The smooth physical "
           "fields, synthetic observations, 20×20 grid, nt=250, source/receivers, stored float32 "
           "wavefield histories and float64 loss accumulation match the existing chain audit. "
           "Every inactive material is a fixed tensor with requires_grad=False, omitted from the "
           "AD/optimizer parameter list. The fields function accepts an active parameter vector and "
           "always returns the same frozen inactive tensor.","",
           "Each case performs a fresh FDTD → combined data+TV loss → one backward. Its actual "
           "combined gradient drives independent forward-only central FD, with all 18 original "
           "steps and no physical-cell mask. Separate data/TV gradients are retained and compared "
           "with the combined result. The unchanged strict criteria require two neighboring "
           "stable, resolved FD values for PASS; measured +/- native material ULP response and "
           "float64 TV arithmetic bounds distinguish FAIL from INCONCLUSIVE. No best-single-step "
           "PASS is allowed. Each single-property case has one direction; each joint case has "
           "epsilon/sigma/joint directions, giving 20 directions and 60 term-direction checks.","",
           "Each case executes torch.optim.SGD.step() on cloned **active** physical parameters. "
           "Frozen clones are requires_grad=False and excluded from the optimizer, and remain "
           "bitwise identical. Active changes are checked against theta-lr*actual_gradient within "
           "float64 rounding tolerance; the step caps maximum proposed update at 1e-10 times "
           "max(1, active parameter scale). All original material tensors are unchanged. Loss "
           "decrease is neither evaluated nor used as a PASS criterion.","",
           f"Case count: {len(data['cases'])}. Term-direction counts: {data['counts']}. "
           f"Execution summary: {data['summary']}.","",
           "| Mode | PML | Loss | Direction | Data | TV | Total | Total AD |",
           "|---|---|---|---|---|---|---|---|" ]
    for c in data["cases"]:
        for direction in c["directions"]:
            checks=direction["checks"]
            lines.append(f"| {c['mode']} | {c['pml']} | {c['objective']} | {direction['group']} | "
                         f"{checks[0]['status']} | {checks[1]['status']} | {checks[2]['status']} | {checks[2]['ad']:.9g} |")
    lines += ["", "PML failures and numerically unresolved conductivity directions remain labelled "
              "FAIL/INCONCLUSIVE. A frozen tensor working correctly does not prove the full PML "
              "propagation derivative correct. Full per-parameter records, all FD curves, empirical "
              "resolution probes, random seeds and source/library hashes are in data/grid_fwi_freeze.json.","",
              "## Reproduce","","```sh",COMMAND,"```","",
              "CPU only, OMP_NUM_THREADS=1. Run log: logs/grid_fwi_freeze.log. No external "
              "experiment/data import, production edits or GPU execution. No optimization "
              "convergence is claimed.",""]
    (AUDIT/"grid_fwi_freeze_audit.md").write_text("\n".join(lines))


def main():
    start=time.perf_counter()
    originals=[AUDIT/name for name in ("network_verification.py","network_audit.md","data/network_results.json",
                                       "logs/network_verification.log")]
    old_hashes={str(path.relative_to(AUDIT)):digest(path) for path in originals}
    native=Path(audit.DeepGPR.get_deepgpr_library_path("cpu"));native_start=digest(native)
    coords,args=audit.configure()
    cases=[]
    for pml in (0,6):
        obs=audit.observed(coords,pml,args)
        for kind in ("L1x1e7","MSEx1e4"):
            for mode in ("epsilon_only","sigma_only","joint"):
                cases.append(run_case(coords,args,pml,obs,mode,kind))
    checks=[r for c in cases for direction in c["directions"] for r in direction["checks"]]
    metadata={"command":COMMAND,"utc_finished":datetime.now(timezone.utc).isoformat(),
              "python":sys.executable,"python_version":platform.python_version(),"platform":platform.platform(),
              "torch_version":torch.__version__,"numpy_version":np.__version__,"device":"cpu",
              "OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS"),"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_threads":torch.get_num_threads(),"seed":audit.SEED,"shape":audit.SHAPE,"nt":audit.NT,
              "dx":audit.DX,"dt":audit.DT,"FD_steps":audit.STEPS,"FD_only_forward":True,"boundary_mask":False,
              "source_location":args["source_location"].tolist(),"receiver_location":args["receiver_location"].tolist(),
              "DeepGPR_import":audit.DeepGPR.__file__,"audit_helper_import":audit.__file__,"external_imports":[],
              "script_sha256":digest(Path(__file__)),"helper_sha256":digest(Path(audit.__file__)),
              "native_library":{"path":str(native),"sha256_at_start":native_start,"sha256_at_end":digest(native)},
              "previous_network_artifact_sha256":old_hashes,
              "previous_network_artifacts_unchanged":all(digest(p)==old_hashes[str(p.relative_to(AUDIT))] for p in originals),
              "elapsed_seconds":time.perf_counter()-start}
    summary={"actual_combined_backward_cases":sum(c["combined_backward_comparison"]["fresh_FDTD_graph"] and
                                                   c["combined_backward_comparison"]["total_backward_calls"]==1 for c in cases),
             "combined_vs_separate_gradient_agreement_cases":sum(c["combined_backward_comparison"]["all_parameters_agree"] for c in cases),
             "active_gradients_present_finite_cases":sum(c["active_gradients_present_and_finite"] for c in cases),
             "fixed_inactive_tensors_verified_cases":sum(c["inactive_physical_tensors_fixed"] for c in cases),
             "actual_clone_SGD_only_active_updated_cases":sum(c["clone_only_SGD"]["only_active_clones_updated"] for c in cases),
             "original_physical_tensors_unchanged_cases":sum(c["clone_only_SGD"]["all_originals_unchanged"] for c in cases),
             "single_property_cases_with_frozen_tensor":sum(bool(c["inactive_parameter_names"]) for c in cases),
             "directions":sum(len(c["directions"]) for c in cases),"term_direction_checks":len(checks),
             "loss_descent_evaluated":False}
    data=audit.safe({"metadata":metadata,"summary":summary,"cases":cases,
                     "counts":{status:sum(r["status"]==status for r in checks)
                               for status in ("PASS","FAIL","INCONCLUSIVE")}})
    (AUDIT/"data/grid_fwi_freeze.json").write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    write_report(data)
    print("COUNTS",data["counts"],"SUMMARY",summary,"elapsed",metadata["elapsed_seconds"],flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
