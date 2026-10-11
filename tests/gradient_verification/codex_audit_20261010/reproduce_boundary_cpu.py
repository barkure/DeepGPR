"""Small CPU confirmation of the existing t04 physical-boundary failure.

This intentionally preserves and reuses the existing synthetic model and
observed-data construction, and never writes to the original t01..t06 data.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gv_common as gv  # noqa: E402
import torch  # noqa: E402
from DeepGPR.solver import modeling  # noqa: E402


def main():
    torch.set_num_threads(1)
    problem = gv.make_problem_2d("G", nt=400, pml=10, obs_path="cpu")
    ad = gv.objective_ad(problem, "cpu", problem.eps, problem.sig)
    rng = torch.Generator().manual_seed(4)
    # Match the rng consumption preceding G_total_derivative in t04_pml.py.
    gv.directions(problem.eps.shape, problem.pml, "random", rng)
    gv.directions(problem.eps.shape, problem.pml, "random", rng)
    direction = torch.randn(problem.eps.shape, generator=rng, dtype=torch.float64)
    direction /= direction.abs().max()
    direction *= 1.0 - gv.edge_mask(problem.eps.shape, problem.pml)
    perturbation = direction * float(problem.eps.mean())
    g_ad = float((ad.grad_eps * perturbation).sum())
    eps_full = gv.pad_replicate(problem.eps, problem.pml)
    sig_full = gv.pad_replicate(problem.sig, problem.pml)
    full_direction = gv.embed_physical(perturbation, problem.pml)
    frozen_zero = gv.objective_frozen_pml(problem, "cpu", eps_full, sig_full)[0]
    rows = []
    for h in (0.03, 0.01, 0.003, 0.001):
        jp = gv.objective_forward(problem, "cpu", problem.eps + h * perturbation, problem.sig)
        jm = gv.objective_forward(problem, "cpu", problem.eps - h * perturbation, problem.sig)
        fp = gv.objective_frozen_pml(problem, "cpu", eps_full + h * full_direction, sig_full)[0]
        fm = gv.objective_frozen_pml(problem, "cpu", eps_full - h * full_direction, sig_full)[0]
        rows.append({"h": h, "total_compute_fd": (jp - jm) / (2 * h),
                     "fixed_boundary_fd": (fp - fm) / (2 * h),
                     "J_plus": jp, "J_minus": jm,
                     "total_first_order_remainder": abs(jp - ad.J - h * g_ad)})
    # A second forward-only experiment isolates the replicate-material term:
    # rebuild the extended materials for every perturbation, but reuse exactly
    # the CPML coefficient arrays obtained from the baseline model.
    original_coefficients = modeling.build_pml_coeffs
    coefficient_cache = []

    def fixed_coefficients(*args, **kwargs):
        if not coefficient_cache:
            coefficient_cache.extend(original_coefficients(*args, **kwargs))
        return tuple(coefficient_cache)

    try:
        modeling.build_pml_coeffs = fixed_coefficients
        gv.objective_forward(problem, "cpu", problem.eps, problem.sig)
        for row in rows:
            h = row["h"]
            jp = gv.objective_forward(problem, "cpu", problem.eps + h * perturbation, problem.sig)
            jm = gv.objective_forward(problem, "cpu", problem.eps - h * perturbation, problem.sig)
            row["replicated_material_fixed_coefficients_fd"] = (jp - jm) / (2 * h)
            row["missing_replication_term_fd"] = row["replicated_material_fixed_coefficients_fd"] - row["fixed_boundary_fd"]
            row["missing_coefficient_term_fd"] = row["total_compute_fd"] - row["replicated_material_fixed_coefficients_fd"]
    finally:
        modeling.build_pml_coeffs = original_coefficients
    result = {"path": "cpu", "shape": list(problem.eps.shape), "nt": problem.nt,
              "pml": problem.pml, "threads": torch.get_num_threads(),
              "precision": "native float32, objective and directions float64",
              "J0": ad.J, "J0_frozen": frozen_zero, "g_ad": g_ad,
              "direction_nonzero_cells": int(direction.count_nonzero()), "rows": rows}
    target = HERE / "boundary_cpu_reproduction.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
