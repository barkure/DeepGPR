"""Dump forward/adjoint outputs of the current native libraries for a before/after comparison.

Usage: python regression_dump.py OUT.pt          (dump)
       python regression_dump.py --compare A.pt B.pt [--json NAME]

Cases: 2D (CPU, CUDA fast path, CUDA general path) and 3D (CPU, CUDA), orders
2/4/8, CPML configurations including asymmetric / disabled faces, zero initial
states (the physical use) - receiver data, the 30 final states, eps / sigma /
source gradients, and the E history.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gv_common as gv  # noqa: E402
import torch  # noqa: E402


def cases():
    out = []
    for order in (2, 4, 8):
        for pml in (10, [3, 7, 0, 5]):
            out.append(("2d", order, pml))
    for order in (2, 4):
        for pml in (4, [2, 3, 0, 4, 1, 2]):
            out.append(("3d", order, pml))
    return out


def dump(path_out):
    gv.seed_all()
    paths2 = ["cpu"] + (["cuda_fast", "cuda_general"] if gv.cuda_available() else [])
    paths3 = ["cpu"] + (["cuda_general"] if gv.cuda_available() else [])
    result = {}
    for dims, order, pml in cases():
        problem = (gv.make_problem_2d("D", shape=(32, 30), nt=300, order=order, pml=pml) if dims == "2d"
                   else gv.make_problem_3d("D", shape=(12, 12, 10), nt=120, order=order, pml=pml))
        for path in (paths2 if dims == "2d" else paths3):
            ad = gv.objective_ad(problem, path, problem.eps, problem.sig, wrt=("eps", "sig", "src"))
            fwd = gv.compute(problem, path, problem.eps, problem.sig, history=True,
                             model_gradient_sampling_interval=1, wavefield_storage_dtype=torch.float32)
            states = [t.detach().cpu() for t in (*fwd[1], *fwd[2], *fwd[3])]
            key = f"{dims}|order{order}|pml{pml}|{path}"
            result[key] = dict(data=ad.data, grad_eps=ad.grad_eps, grad_sig=ad.grad_sig, grad_src=ad.grad_src,
                               states=states, history=fwd[0].detach().cpu(), J=ad.J)
            print("dumped", key, flush=True)
    torch.save(result, path_out)


def compare(a_path, b_path, json_name=None):
    a, b = torch.load(a_path), torch.load(b_path)
    rows = []
    for key in a:
        x, y = a[key], b[key]
        row = dict(case=key,
                   data_bitwise=bool(torch.equal(x["data"], y["data"])),
                   states_bitwise=all(torch.equal(p, q) for p, q in zip(x["states"], y["states"])),
                   history_bitwise=bool(torch.equal(x["history"], y["history"])))
        for g in ("grad_eps", "grad_sig", "grad_src"):
            row[g + "_bitwise"] = bool(torch.equal(x[g], y[g]))
            row[g + "_rel"] = gv.relative_l2(y[g], x[g])
        rows.append(row)
        print(row, flush=True)
    if json_name:
        gv.save_json(json_name, dict(before=str(a_path), after=str(b_path), rows=rows))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        name = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
        compare(sys.argv[2], sys.argv[3], name)
    else:
        dump(sys.argv[1])
