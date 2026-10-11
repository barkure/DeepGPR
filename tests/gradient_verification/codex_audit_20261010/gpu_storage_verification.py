"""Serial fresh synthetic CUDA storage/checkpoint audit (no old result reuse).

Do not execute while another agent owns the GPU. Output: data/gpu_storage.json.
Every FD objective is forward-only; same-run FD sweeps are shared only between
storage configurations whose receiver data equal the same exact forward.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
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

SEED = 2026101021
BASE = dict(model_gradient_sampling_interval=1,
            wavefield_storage_dtype=torch.float32,
            wavefield_rhs_history="stored",
            wavefield_history_region="extended",
            wavefield_compression="none")


def metric(actual, reference):
    actual, reference = actual.double(), reference.double()
    a_norm, r_norm = float(actual.norm()), float(reference.norm())
    diff = float((actual - reference).norm())
    rel = diff / r_norm if r_norm else (0. if not diff else math.inf)
    cosine = float((actual * reference).sum()) / (a_norm * r_norm) if a_norm and r_norm else None
    return dict(relative_l2=rel, cosine=cosine, norm=a_norm, reference_norm=r_norm,
                max_abs=float((actual-reference).abs().max()) if actual.numel() else 0.)


def physical_faces(shape):
    ring = torch.zeros(shape, dtype=torch.bool)
    for dim in range(len(shape)):
        sl = [slice(None)] * len(shape)
        sl[dim] = 0
        ring[tuple(sl)] = True
        sl[dim] = -1
        ring[tuple(sl)] = True
    return ring


def initial_states(problem, path):
    device = gv.path_device(path)
    e,h,p = DeepGPR.checkpoint_initial_field(
        device=device, dx=problem.dx, dt=problem.dt,
        source_amplitudes=problem.wavelet.to(device),
        source_location=problem.src_loc.to(device), receiver_location=problem.rec_loc.to(device),
        er=problem.eps.to(device), se=problem.sig.to(device),
        pmlthick=problem.pml, fdtd_order=problem.order)
    return tuple((*e,*h,*p))


def buffer_record(tensor):
    if tensor is None:
        return None
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                pinned=tensor.is_pinned() if tensor.device.type == "cpu" else False,
                bytes=tensor.numel()*tensor.element_size())


def effective_record(data):
    ctx = data.grad_fn
    cfg = getattr(ctx, "config", None)
    if cfg is None:
        return dict(available=False)
    return dict(available=True, tm2d_fast_path=cfg.tm2d_fast_path,
                sampling_interval=cfg.sampling_interval, reconstruct_rhs=ctx.reconstruct_rhs,
                async_requested=cfg.use_async_offload, async_effective=ctx.use_async_offload,
                storage_dtype=str(cfg.storage.dtype), compression=cfg.storage.compression,
                conversion_backend=cfg.storage.conversion_backend,
                int8_reduction_backend=cfg.storage.int8_reduction_backend,
                compression_block=cfg.storage.block_size, history_region=cfg.history_region,
                E_saved=buffer_record(ctx.E_saved), R_saved=buffer_record(ctx.R_saved),
                E_final=buffer_record(ctx.E_final))


def solve(problem, path, eps, sig, options, k=None, mode=None, calls=None):
    if k is None:
        out=gv.compute(problem,path,eps,sig,history=True,**options)
        return out[-1],(*out[1],*out[2],*out[3]),effective_record(out[-1])
    state=initial_states(problem,path)
    original_initial=state
    first_state=[s.detach().clone() for s in state]
    device=gv.path_device(path)
    source=problem.wavelet.to(device)
    size,rem=divmod(problem.nt,k)
    sizes=[size+(i<rem) for i in range(k)]

    def segment(e,s,wave,*boundary):
        if calls is not None:
            calls.append(dict(nt=wave.shape[1],grad_enabled=torch.is_grad_enabled()))
        work=tuple(t.clone() for t in boundary)
        out=gv.compute(problem,path,e,s,source=wave,states=work,history=True,**options)
        return (*out[1],*out[2],*out[3],out[-1])

    traces=[]
    start=0
    for n in sizes:
        wave=source[:,start:start+n]
        start+=n
        out=checkpoint(segment,eps,sig,wave,*state,use_reentrant=(mode=="reentrant"))
        state,data=out[:-1],out[-1]
        traces.append(data)
    return torch.cat(traces,1),state,dict(checkpoint=True,k=k,mode=mode,sizes=sizes,
        initial_inputs_preserved=all(torch.equal(a,b) for a,b in zip(original_initial,first_state)))


def ad_run(problem,path,options,k=None,mode=None):
    device=gv.path_device(path)
    torch.cuda.synchronize()
    baseline_allocated=torch.cuda.memory_allocated()
    baseline_reserved=torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    eps=problem.eps.to(device=device,dtype=torch.float32).detach().clone().requires_grad_()
    sig=problem.sig.to(device=device,dtype=torch.float32).detach().clone().requires_grad_()
    calls=[]
    data,states,effective=solve(problem,path,eps,sig,options,k,mode,calls)
    loss=gv.misfit(problem,data)
    torch.cuda.synchronize()
    forward_done=time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_done=time.perf_counter()
    measured=dict(forward_seconds=forward_done-started,
                  backward_seconds=backward_done-forward_done,
                  total_seconds=backward_done-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  baseline_allocated_bytes=baseline_allocated,
                  baseline_reserved_bytes=baseline_reserved,
                  memory_scope="PyTorch allocator; excludes native cudaMalloc and process/driver allocation")
    return dict(J=float(loss),data=data.detach().cpu().clone(),
                states=[s.detach().cpu().clone() for s in states],
                eps=eps.grad.detach().cpu().clone(),sig=sig.grad.detach().cpu().clone(),
                effective=effective,calls=calls,measured=measured)


def forward_fd_sweeps(problem,path):
    """Fresh forward-only objective sweep. Compact and full-edge are separate.

    Values are reusable within this fresh run when receiver data are bitwise
    equal, because storage is absent from these independent forward calls.
    """
    rng=torch.Generator().manual_seed(SEED+problem.dims)
    shape=problem.eps.shape
    coords=torch.meshgrid(*[torch.arange(n) for n in shape],indexing="ij")
    compact=torch.ones(shape,dtype=torch.bool)
    for coord,n in zip(coords,shape):
        compact &= (coord>=3)&(coord<n-3)
    faces=physical_faces(shape)
    sweeps=[]
    for parameter in ("eps","sig"):
        for region,mask in (("interior_compact_support",compact),("all_outer_physical_faces",faces)):
            direction=torch.randn(shape,generator=rng,dtype=torch.float64)*mask
            direction/=direction.abs().max()
            direction*=4. if parameter=="eps" else .01
            values=[]
            for h in (.3,.1,.03,.01,.003,.001,.0003):
                loss=[]
                for sign in (-1,1):
                    e,s=problem.eps.clone(),problem.sig.clone()
                    if parameter=="eps":e+=sign*h*direction
                    else:s+=sign*h*direction
                    # NO history, NO autograd, NO native backward, NO stored data.
                    with torch.no_grad():
                        value=gv.objective_forward(problem,path,e,s)
                    loss.append(value)
                values.append(dict(relative_step=h,Jminus=loss[0],Jplus=loss[1],
                                   g_fd=(loss[1]-loss[0])/(2*h)))
            sweeps.append(dict(parameter=parameter,region=region,direction=direction,
                               active_cells=int(mask.sum()),total_cells=mask.numel(),values=values))
    return sweeps


def fd_compare(actual,sweeps,forward_bitwise):
    rows=[]
    for sweep in sweeps:
        gad=float((actual[sweep["parameter"]].double()*sweep["direction"]).sum())
        errors=[]
        for value in sweep["values"]:
            gfd=value["g_fd"]
            err=abs(gad-gfd)/max(abs(gad),abs(gfd),1e-30)
            errors.append(dict(relative_step=value["relative_step"],g_fd=gfd,relative_error=err))
        best=min(v["relative_error"] for v in errors)
        acceptable=sum(v["relative_error"]<.01 for v in errors)
        rows.append(dict(parameter=sweep["parameter"],region=sweep["region"],g_ad=gad,
                          errors=errors,best_relative_error=best,
                          criterion="best <0.5% and at least 2 h values <1%",
                          verdict=("PASS" if best<.005 and acceptable>=2 else "FAIL")
                          if forward_bitwise else "NOT_COMPARABLE_FORWARD_CHANGED",
                          fd_provenance="same-run independent forward-only sweep; shared only for identical receivers"))
    return rows


def comparison(actual,reference,shape):
    faces=physical_faces(shape)
    state_rows=[dict(name=name,bitwise=torch.equal(a,b),**metric(a,b))
                for name,a,b in zip(STATE_NAMES,actual["states"],reference["states"])]
    return dict(J=actual["J"],reference_J=reference["J"],
                waveform_bitwise=torch.equal(actual["data"],reference["data"]),
                waveform=metric(actual["data"],reference["data"]),
                all_30_states_bitwise=all(r["bitwise"] for r in state_rows),
                states=state_rows,eps=metric(actual["eps"],reference["eps"]),
                sig=metric(actual["sig"],reference["sig"]),
                eps_outer_faces=metric(actual["eps"][faces],reference["eps"][faces]),
                sig_outer_faces=metric(actual["sig"][faces],reference["sig"][faces]),
                effective=actual["effective"],calls=actual["calls"],measured=actual["measured"])


def storage_configs():
    yield "stored_fp32_baseline",{},"exact_baseline"
    for name,options in (
        ("fp32_auto",dict(wavefield_rhs_history="auto")),
        ("fp32_reconstructed",dict(wavefield_rhs_history="reconstructed")),
        ("fp32_physical_stored",dict(wavefield_history_region="physical")),
        ("fp32_physical_auto",dict(wavefield_history_region="physical",wavefield_rhs_history="auto"))):
        yield name,options,"exact_storage"
    for dtype,label in ((torch.float16,"fp16"),(torch.bfloat16,"bf16")):
        for backend in ("legacy","native_scalar","native_vec2"):
            for rhs in ("stored","auto","reconstructed"):
                yield f"{label}_{backend}_{rhs}",dict(wavefield_storage_dtype=dtype,
                    wavefield_conversion_backend=backend,wavefield_rhs_history=rhs),"approximate_storage"
    for dtype,label in ((torch.float32,"fp32"),(torch.float16,"fp16"),(torch.bfloat16,"bf16")):
        for rhs in ("stored","auto"):
            yield f"async_{label}_{rhs}",dict(use_async_offload=True,wavefield_storage_dtype=dtype,
                  wavefield_rhs_history=rhs),"exact_storage" if dtype==torch.float32 else "approximate_storage"
    for rhs in ("stored","auto","reconstructed"):
        yield f"int8_{rhs}",dict(wavefield_compression="int8",wavefield_rhs_history=rhs),"approximate_storage"
    yield "int8_physical_stored",dict(wavefield_compression="int8",wavefield_history_region="physical"),"approximate_storage"
    for sampling in (3,5):
        yield f"sampling_{sampling}",dict(model_gradient_sampling_interval=sampling),"approximate_sampling"


def save(payload):
    out=HERE/"data/gpu_storage.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(gv.to_jsonable(payload),indent=2)+"\n")


def main():
    torch.set_num_threads(1)
    gv.seed_all(SEED)
    if os.environ.get("OMP_NUM_THREADS")!="1":raise RuntimeError("requires OMP_NUM_THREADS=1")
    if not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable; this script does not mark CPU runs as GPU evidence")
    environment=gv.environment()
    environment["seed"]=SEED
    result=dict(environment=environment,executable=sys.executable,seed=SEED,
                invocation="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "+sys.executable+" "+str(Path(__file__).relative_to(gv.REPO)),
                native_cuda_sha256=hashlib.sha256((gv.REPO/"src/DeepGPR/lib/deepgpr.so").read_bytes()).hexdigest(),
                tests=[],finite_difference_sweeps=[],configuration_rejections=[],
                public_derivative_status="Full public forward regenerates PML; complete edge derivative is independently tested and FAIL is retained. Fixed-PML equivalence is a separate claim.",
                approximate_verdict_policy="MEASURED_APPROXIMATION; no 10x historical benchmark tolerance and no universal accuracy guarantee")
    lib=get_deepgpr_lib("cuda")
    symbols=("deepgpr_supports_external_pml","deepgpr_supports_rhs_reconstruction",
             "deepgpr_supports_physical_history","deepgpr_supports_conversion_backends",
             "deepgpr_supports_int8_wavefield","deepgpr_supports_int8_reduction_backends")
    result["capabilities"]={s:int(getattr(lib,s)()) if hasattr(lib,s) else None for s in symbols}
    configs=list(storage_configs())
    failures=0
    for dims,path in ((2,"cuda_fast"),(3,"cuda_general")):
        if dims==2:
            p=gv.make_problem_2d("D",shape=(40,40),nt=400,pml=6,order=2,obs_path=path)
        else:
            p=gv.make_problem_3d("D",shape=(14,14,10),nt=200,pml=3,order=2,obs_path=path)
        print("geometry",dims,"shape",tuple(p.eps.shape),"nt",p.nt,"begin FD",flush=True)
        fd=forward_fd_sweeps(p,path)
        result["finite_difference_sweeps"].append(dict(dims=dims,path=path,shape=list(p.eps.shape),
            nt=p.nt,pml=p.pml,dx=p.dx,dt=p.dt,order=p.order,
            sweeps=[dict(parameter=s["parameter"],region=s["region"],
                         active_cells=s["active_cells"],total_cells=s["total_cells"],
                         direction=s["direction"].tolist(),values=s["values"]) for s in fd]))
        reference=None
        for name,options,kind in configs:
            options=dict(BASE,**options)
            row=dict(dims=dims,path=path,name=name,kind=kind,
                     requested_options={k:str(v) for k,v in options.items()})
            try:
                actual=ad_run(p,path,options)
                if name=="stored_fp32_baseline":reference=actual
                if reference is None:raise RuntimeError("Exact stored-FP32 baseline failed; comparison is unavailable")
                row.update(comparison(actual,reference,p.eps.shape))
                row["finite_differences"]=fd_compare(actual,fd,row["waveform_bitwise"])
                if kind.startswith("exact"):
                    ok=(row["waveform_bitwise"] and row["all_30_states_bitwise"]
                        and max(row["eps"]["relative_l2"],row["sig"]["relative_l2"])<1e-5)
                    row["storage_equivalence_verdict"]="PASS" if ok else "FAIL"
                    failures+=not ok
                else:
                    row["storage_equivalence_verdict"]=("MEASURED_APPROXIMATION" if row["waveform_bitwise"] and row["all_30_states_bitwise"] else "FAIL_FORWARD_CHANGED")
                    failures+=not row["waveform_bitwise"]
                print(dims,name,row["storage_equivalence_verdict"],"eps/sig",
                      row["eps"]["relative_l2"],row["sig"]["relative_l2"],
                      "FD",[(f["parameter"],f["region"],f["verdict"],f["best_relative_error"]) for f in row["finite_differences"]],
                      "effective",{k:row["effective"].get(k) for k in ("reconstruct_rhs","async_effective","conversion_backend")},flush=True)
            except Exception as exc:
                row.update(storage_equivalence_verdict="EXECUTION_ERROR",error=repr(exc))
                failures+=1
                print(dims,name,"EXECUTION_ERROR",repr(exc),flush=True)
            result["tests"].append(row)
            save(result)
        for k in (2,3):
            for mode in ("reentrant","nonreentrant"):
                row=dict(dims=dims,path=path,name=f"checkpoint_K{k}_{mode}",kind="checkpoint")
                try:
                    actual=ad_run(p,path,dict(BASE),k=k,mode=mode)
                    row.update(comparison(actual,reference,p.eps.shape))
                    row["finite_differences"]=fd_compare(actual,fd,row["waveform_bitwise"])
                    ok=(row["waveform_bitwise"] and row["all_30_states_bitwise"]
                        and max(row["eps"]["relative_l2"],row["sig"]["relative_l2"])<1e-5
                        and len(row["calls"])==2*k
                        and row["effective"]["initial_inputs_preserved"])
                    row["storage_equivalence_verdict"]="PASS" if ok else "FAIL"
                    failures+=not ok
                    print(dims,row["name"],row["storage_equivalence_verdict"],"eps/sig",
                          row["eps"]["relative_l2"],row["sig"]["relative_l2"],"calls",len(row["calls"]),flush=True)
                except Exception as exc:
                    row.update(storage_equivalence_verdict="EXECUTION_ERROR",error=repr(exc))
                    failures+=1
                    print(dims,row["name"],"EXECUTION_ERROR",repr(exc),flush=True)
                result["tests"].append(row)
                save(result)
        for name,bad in (
            ("int8_plus_fp16",dict(wavefield_compression="int8",wavefield_storage_dtype=torch.float16)),
            ("int8_plus_async",dict(wavefield_compression="int8",use_async_offload=True)),
            ("zfp",dict(wavefield_compression="zfp"))):
            row=dict(dims=dims,path=path,name=name,expected_rejection=True)
            try:
                ad_run(p,path,dict(BASE,**bad))
                row.update(rejected=False,verdict="FAIL_UNEXPECTED_ACCEPTANCE")
                failures+=1
            except (ValueError,NotImplementedError) as exc:
                row.update(rejected=True,exception=repr(exc),verdict="PASS_VALIDATION")
            result["configuration_rejections"].append(row)
        save(result)
    result["execution_failures"]=failures
    result["complete_public_derivative_failures"]=sum(
        f["verdict"]=="FAIL" for r in result["tests"] if r["kind"]=="exact_baseline"
        for f in r.get("finite_differences",[]))
    save(result)
    print("wrote",HERE/"data/gpu_storage.json","execution/storage failures",failures,
          "public-forward FD failures",result["complete_public_derivative_failures"],flush=True)
    return bool(failures or result["complete_public_derivative_failures"])


if __name__=="__main__":
    raise SystemExit(main())
