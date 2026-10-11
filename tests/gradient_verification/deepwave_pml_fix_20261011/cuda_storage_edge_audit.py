"""Synthetic CUDA history/storage PML-edge audit; preserves earlier evidence.

No external data, experiment modules, production edits or performance claims.
General/TM2D routing is selected by a test-only override of the route predicate;
all propagation and backward equations are the current public native solver.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime,timezone
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'src'))
import torch
import DeepGPR

modeling=importlib.import_module('DeepGPR.solver.modeling')
SHAPE=(20,20);NT=220;PML=6;DT=3e-11;DX=.02;SEED=20261011
STEPS=(.3,.1,.03,.01,.003,.001,.0003,.0001,.00003,.00001)
COMMAND=('OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '
         '/home/ll/anaconda3/envs/deepgpr/bin/python -u '+str(Path(__file__).resolve()))


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return value


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    torch.set_num_threads(1);torch.manual_seed(SEED)
    x,y=torch.meshgrid(torch.linspace(-1,1,20,dtype=torch.float64),
                       torch.linspace(-1,1,20,dtype=torch.float64),indexing='ij')
    eps=(4.+.10*x+.05*y).to('cuda')
    sigma=(.010+.0003*x+.0002*y).to('cuda')
    source=DeepGPR.ricker(4e8,NT,DT,1.1e-9,dtype=torch.float32,device='cuda')*1e-4
    # Waveforms are shared by the shots; the first dimension counts sources
    # per shot rather than shots in the acquisition geometry.
    source=source.reshape(1,NT,1)
    common={'device':'cuda','dx':DX,'dt':DT,'pmlthick':PML,'pml_max_velocity':299792458.,
            'fdtd_order':4,'mode':2,'model_gradient_sampling_interval':1,
            'wavefield_history_region':'extended','source_amplitudes':source,
            'source_location':torch.tensor([[[1,3,0]],[[17,17,0]]],dtype=torch.int32,device='cuda'),
            'receiver_location':torch.tensor([[[0,2,0],[1,2,0],[18,18,0],[19,18,0],[8,8,0]]]*2,
                                            dtype=torch.int32,device='cuda')}
    bump=torch.exp(-((x-.1)**2+(y+.25)**2)/.3).to('cuda')
    return eps,sigma,common,eps+.07*bump+.025,sigma+.0004*bump+.0001


def solve(eps,sigma,common,path,history=False,storage=None):
    options=dict(common);options['save_wavefield_history']=history
    if storage:options.update(storage)
    with patch.object(modeling,'_tm2d_fast_path_applies',return_value=(path=='tm2d')):
        return DeepGPR.compute(eps_r=eps.float(),sigma=sigma.float(),**options)


def objective(data,target,scale):
    return .5*((data.double()-target)/scale).square().mean()


def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))


def direction_specs(eps,sigma):
    generator=torch.Generator().manual_seed(SEED+1)
    edge=torch.zeros(SHAPE,dtype=torch.float64);edge[0,:]=1;edge[-1,:]=1;edge[:,0]=1;edge[:,-1]=1
    corners=torch.zeros_like(edge);corners[0,0]=1;corners[0,-1]=.8;corners[-1,0]=.6;corners[-1,-1]=.7
    # Positive masks avoid cancellations and isolate material replication at
    # faces/corners. The joint-all direction includes every physical cell.
    all_cells=torch.randn(SHAPE,generator=generator,dtype=torch.float64)
    all_cells/=all_cells.abs().max()
    rows=[]
    for name,mask,active in [('epsilon_edge',edge,'eps'),('sigma_edge',edge,'sigma'),
                             ('joint_edge',edge,'both'),('epsilon_corner',corners,'eps'),
                             ('sigma_corner',corners,'sigma'),('joint_all',all_cells,'both')]:
        mask=mask.to('cuda')
        de=mask*.1 if active in ('eps','both') else torch.zeros_like(eps)
        ds=mask*.001 if active in ('sigma','both') else torch.zeros_like(sigma)
        rows.append((name,de,ds))
    return rows


def forward_fd(eps,sigma,common,path,target,scale,specs):
    curves=[]
    def evaluate(e,s):
        with torch.no_grad():return float(objective(solve(e,s,common,path)[-1],target,scale))
    baseline=evaluate(eps,sigma)
    repeat=[evaluate(eps,sigma) for _ in range(3)]
    ulps=[]
    for v in (eps,sigma):
        f=v.float();ulps.append((torch.nextafter(f,torch.full_like(f,float('inf')))-f).double())
    for index,(name,de,ds) in enumerate(specs):
        generator=torch.Generator().manual_seed(SEED+20+index)
        probes=[]
        for _ in range(8):
            e=eps;sg=sigma
            if bool(de.count_nonzero()):
                signs=(torch.randint(0,2,SHAPE,generator=generator).to('cuda')*2-1).double()
                e=e+signs*ulps[0]*(de!=0)
            if bool(ds.count_nonzero()):
                signs=(torch.randint(0,2,SHAPE,generator=generator).to('cuda')*2-1).double()
                sg=sg+signs*ulps[1]*(ds!=0)
            probes.append(evaluate(e,sg))
        bound=max([abs(j-baseline) for j in probes+repeat]+[64*torch.finfo(torch.float64).eps*max(abs(baseline),1.)])
        rows=[]
        for h in STEPS:
            jp=evaluate(eps+h*de,sigma+h*ds);jm=evaluate(eps-h*de,sigma-h*ds)
            rows.append({'h':h,'J_plus':jp,'J_minus':jm,'fd':(jp-jm)/(2*h)})
        curves.append({'direction':name,'active_material_ulp_probes':probes,'J0':baseline,
                       'deterministic_repeats':repeat,'value_resolution_bound':bound,
                       'epsilon_direction_norm':float(de.norm()),'sigma_direction_norm':float(ds.norm()),
                       'epsilon_changed_cells':int(de.count_nonzero()),'sigma_changed_cells':int(ds.count_nonzero()),
                       'physical_edges_included':True,'boundary_exclusion_mask':False,'rows':rows})
        print('FORWARD_FD',path,name,flush=True)
    return curves


def classify(ad,curve):
    rows=[dict(r,absolute_error=abs(r['fd']-ad),relative_error=abs(r['fd']-ad)/max(abs(r['fd']),abs(ad),1e-30))
          for r in curve['rows']]
    pairs=[];noise=curve['value_resolution_bound']
    for a,b in zip(rows,rows[1:]):
        scale=max(abs(a['fd']),abs(b['fd']),abs(ad),1e-30);unc=noise/min(a['h'],b['h'])
        stable=abs(a['fd']-b['fd'])<=.005*scale;resolved=scale>10*unc
        pair={'hs':[a['h'],b['h']],'fds':[a['fd'],b['fd']],
              'stable_fd':stable,'resolved':resolved,'uncertainty_bound':unc,
              'max_relative_error':max(a['relative_error'],b['relative_error']),
              'max_absolute_error':max(a['absolute_error'],b['absolute_error'])}
        pair['strict_pass']=stable and resolved and pair['max_relative_error']<=.002
        pair['stable_bias']=stable and resolved and min(a['absolute_error'],b['absolute_error'])>.005*scale+3*unc
        counter=[]
        for r in rows:
            if r['h']>=b['h']:continue
            local=max(abs(r['fd']),abs(ad),1e-30);u=noise/r['h']
            if local>10*u and (r['absolute_error']<=.005*local or
                              (local>100*u and r['absolute_error']<=3*u)):counter.append(r['h'])
        pair['finer_counterevidence_h']=counter;pairs.append(pair)
    if any(p['strict_pass'] for p in pairs):status='PASS'
    elif any(p['stable_bias'] and not p['finer_counterevidence_h'] for p in pairs):status='FAIL'
    else:status='INCONCLUSIVE'
    return {'ad':ad,'status':status,'fd_reference':curve['direction'],
            'value_resolution_bound':noise,'adjacent_pairs':pairs,'error_curve':rows}


def run_ad(eps,sigma,common,path,target,scale,storage,name,curves,specs,reference):
    e=eps.detach().clone().requires_grad_();s=sigma.detach().clone().requires_grad_()
    result=solve(e,s,common,path,True,storage);node=result[-1].grad_fn
    history={key:{'device':str(getattr(node,key).device),'dtype':str(getattr(node,key).dtype),
                  'shape':list(getattr(node,key).shape),'bytes':getattr(node,key).numel()*getattr(node,key).element_size()}
             for key in ('E_saved','R_saved','E_final') if getattr(node,key) is not None}
    selected_fast=node.config.tm2d_fast_path;loss=objective(result[-1],target,scale)
    loss.backward();torch.cuda.synchronize()
    assert e.grad is not None and s.grad is not None
    gradients=(e.grad.detach().clone(),s.grad.detach().clone())
    values=[]
    for field,g in zip(('eps','sigma'),gradients):
        values.append({'name':field,'none':False,'nan_count':int(torch.isnan(g).sum()),
                       'inf_count':int(torch.isinf(g).sum()),'norm':float(g.norm()),
                       'corners':[float(g[i,j]) for i,j in ((0,0),(0,19),(19,0),(19,19))]})
    checks=[]
    for curve,(label,de,ds) in zip(curves,specs):
        check=classify(float((gradients[0]*de).sum()+(gradients[1]*ds).sum()),curve)
        check['direction']=label;checks.append(check)
    comparisons=None
    if reference:
        comparisons={'receiver_relative_l2':relative(result[-1].detach().double(),reference[0]),
                     'epsilon_gradient_relative_l2':relative(gradients[0],reference[1][0]),
                     'sigma_gradient_relative_l2':relative(gradients[1],reference[1][1])}
    row={'name':name,'path':path,'selected_tm2d_fast':selected_fast,'storage_options':{
             key:str(value) if isinstance(value,torch.dtype) else value for key,value in storage.items()},
         'loss':float(loss.detach()),'histories':history,'use_async_offload':node.use_async_offload,
         'gradient_records':values,'fp32_reference_comparison':comparisons,'checks':checks,
         'physical_field_dtype':'float64; differentiably cast to native float32',
         'material_gradient_cells_include_all_physical_edges':True}
    print('CASE',path,name,{status:sum(r['status']==status for r in checks)
                           for status in ('PASS','FAIL','INCONCLUSIVE')},flush=True)
    return row,(result[-1].detach().double(),gradients)


def write_report(data):
    lines=['# CUDA storage and PML-edge material-gradient audit','',
           'Synthetic 20×20 2D Ez-TM, two shots, nt=220, PML=6, order=4, dx=.02, dt=3e-11. '
           'The PML reference speed is explicitly fixed at 299792458 m/s for AD and every FD. '
           'No external data/modules or old evidence were modified. This is a correctness audit; '
           'concurrent GPU activity makes elapsed time unsuitable for performance conclusions.','',
           'General and TM2D paths use the real public compute/backward with a test-only route predicate override. '
           'Six physical directions include every boundary face, corners, and all cells. FD uses fresh '
           'save_wavefield_history=False forwards at 10 steps; storage cannot change receiver forwards, '
           'so each path shares its forward-only curves across formats. Eight +/- one-native-ULP probes '
           'per direction and three repeats measure resolution. Strict PASS requires adjacent FD stability '
           '<=.005, signal >10× resolution/h, and both AD relative errors <=.002. Stable resolved bias '
           '> .005 plus 3× resolution/h is FAIL unless finer resolved counterevidence exists; otherwise '
           'INCONCLUSIVE. Lossy history discrepancies are retained, never relabelled exact PASS.','',
           'FP32 and FP16/BF16 legacy/native_scalar/native_vec2 each cover stored/reconstructed RHS '
           'and resident/synchronous history or asynchronous pinned-CPU offload. INT8 covers '
           'stored/reconstructed GPU-resident history; INT8 CPU offload is unsupported and checked '
           'for explicit rejection. All material histories cover the extended grid.','',
           f"Cases: {len(data['cases'])}. Direction checks: {data['counts']}. Unsupported contracts: {data['unsupported_contracts']}.",'',
           '| Path | Storage | PASS | FAIL | INCONCLUSIVE | ε gradient relative to FP32 | σ gradient relative to FP32 |',
           '|---|---|---|---|---|---|---|']
    for c in data['cases']:
        count={status:sum(r['status']==status for r in c['checks']) for status in ('PASS','FAIL','INCONCLUSIVE')}
        p=c['fp32_reference_comparison']
        lines.append(f"| {c['path']} | {c['name']} | {count['PASS']} | {count['FAIL']} | {count['INCONCLUSIVE']} | "
                     f"{p['epsilon_gradient_relative_l2'] if p else 0:.6g} | {p['sigma_gradient_relative_l2'] if p else 0:.6g} |")
    lines += ['', 'Complete shared raw J+/J− curves, resolution probes, case-specific AD/errors, '
              'history devices/bytes, corner gradients, hashes and environment are in data/cuda_storage_edge.json.',
              '', '```sh',COMMAND,'```','', 'Run log: logs/cuda_storage_edge.log.']
    (HERE/'cuda_storage_edge_audit.md').write_text('\n'.join(lines)+'\n')


def main():
    assert torch.cuda.is_available(),'GPU execution required'
    start=time.perf_counter();eps,sigma,common,truth_e,truth_s=setup()
    lib=Path(DeepGPR.get_deepgpr_library_path('cuda'));native_start=sha(lib)
    assert DeepGPR.get_deepgpr_lib(torch.device('cuda')).deepgpr_supports_pml_material_gradients()==1
    formats=[('fp32',torch.float32,'legacy','none')]+[
        (name+'_'+backend,dtype,backend,'none') for name,dtype in [('fp16',torch.float16),('bf16',torch.bfloat16)]
        for backend in ('legacy','native_scalar','native_vec2')]+[('int8',torch.float32,'legacy','int8')]
    specs=direction_specs(eps,sigma);all_curves={};cases=[];contracts=[]
    for path in ('tm2d','general'):
        with torch.no_grad():
            target=solve(truth_e,truth_s,common,path)[-1].double()
            base=solve(eps,sigma,common,path)[-1].double();scale=base.abs().max().clamp_min(1e-12)
        curves=forward_fd(eps,sigma,common,path,target,scale,specs);all_curves[path]=curves
        reference=None
        for label,dtype,backend,compression in formats:
            for rhs in ('stored','reconstructed'):
                for offload in ((False,True) if compression=='none' else (False,)):
                    storage={'wavefield_storage_dtype':dtype,'wavefield_conversion_backend':backend,
                             'wavefield_compression':compression,'wavefield_rhs_history':rhs,
                             'use_async_offload':offload}
                    if compression=='int8':storage['wavefield_compression_block_size']=(8,8)
                    name=label+'/'+rhs+('/async_cpu' if offload else '/resident')
                    row,value=run_ad(eps,sigma,common,path,target,scale,storage,name,curves,specs,reference)
                    if reference is None:reference=value
                    cases.append(row)
        for rhs in ('stored','reconstructed'):
            try:solve(eps,sigma,common,path,True,{'wavefield_compression':'int8','wavefield_rhs_history':rhs,'use_async_offload':True})
            except ValueError as exc:contracts.append({'path':path,'rhs':rhs,'int8_async_cpu':'REJECTED_AS_UNSUPPORTED','error':str(exc)})
            else:raise AssertionError('INT8 asynchronous offload must be rejected')
    checks=[r for c in cases for r in c['checks']]
    data=safe({'metadata':{'command':COMMAND,'python':sys.executable,'python_version':platform.python_version(),
               'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0),'OMP_NUM_THREADS':os.getenv('OMP_NUM_THREADS'),
               'UTC':datetime.now(timezone.utc).isoformat(),'shape':SHAPE,'nt':NT,'pml':PML,'dt':DT,'dx':DX,
               'pml_max_velocity':common['pml_max_velocity'],'steps':STEPS,'seed':SEED,'external_imports':[],
               'source_location':common['source_location'].tolist(),'receiver_location':common['receiver_location'].tolist(),
               'native_library':str(lib),'native_sha256_at_start':native_start,'native_sha256_at_end':sha(lib),
               'script_sha256':sha(Path(__file__)),'elapsed_seconds':time.perf_counter()-start,
               'correctness_only_concurrent_GPU_workload':True},
              'cases':cases,'forward_only_fd':all_curves,'unsupported_contracts':contracts,
              'counts':{status:sum(r['status']==status for r in checks) for status in ('PASS','FAIL','INCONCLUSIVE')}})
    (HERE/'data/cuda_storage_edge.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    write_report(data);print('COUNTS',data['counts'],'cases',len(cases),flush=True)


if __name__=='__main__':main()
