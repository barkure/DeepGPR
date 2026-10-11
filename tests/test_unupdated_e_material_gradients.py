"""Material derivatives at Yee components whose base E update is inactive.

The retained initial field is independent of the material. Boundary sources
and CPML corrections still depend on the electric RHS coefficient, so their
derivatives must be retained rather than dropping the complete component.
"""

import contextlib
import sys
import unittest
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import DeepGPR  # noqa: E402
from DeepGPR.solver import modeling  # noqa: E402


@contextlib.contextmanager
def _path(path):
    original = modeling._tm2d_fast_path_applies
    if path == "cuda_general":
        modeling._tm2d_fast_path_applies = lambda *args, **kwargs: False
    try:
        yield
    finally:
        modeling._tm2d_fast_path_applies = original


def material_case(path="cpu", component=2, dims=2, rhs="reconstructed",
                  source_value=0.0, cpml_correction=False, compression=None,
                  storage_dtype=torch.float32, conversion_backend="auto"):
    device = torch.device("cpu" if path == "cpu" else "cuda")
    shape = (12, 12) if dims == 2 else (12, 12, 8)
    # Ex is inactive at y=0; Ey and Ez are inactive at x=0.
    cell = (4, 0, 3) if component == 0 else (0, 4, 3)
    if dims == 2 or cpml_correction:
        cell = (cell[0], cell[1], 0)
    pml = [0, 0, 0, 0, 2, 0] if cpml_correction else 0
    eps = torch.full(shape, 4.0, dtype=torch.float32, device=device)
    sigma = torch.full(shape, 0.01, dtype=torch.float32, device=device)
    source = torch.full((1, 1, 1), source_value, dtype=torch.float32, device=device)
    location = torch.tensor([[cell]], dtype=torch.int32, device=device)
    dt, dx = 2.5e-11, 0.02
    initial = DeepGPR.checkpoint_initial_field(
        device=device, dx=dx, dt=dt, source_amplitudes=source,
        source_location=location, receiver_location=location,
        er=eps, se=sigma, pmlthick=pml, fdtd_order=2,
    )
    states = [s.detach().clone() for group in initial for s in group]
    extended_cell = (cell[0], cell[1], cell[2] + (2 if cpml_correction else 0))
    states[component][(0, *extended_cell)] = 1.0
    if cpml_correction:
        # z0 electric Phi2 -> Ey; q=0 has RA=RB=RE=1, RF=0.
        states[6 + 4 * 4 + 1][0, 0, 4, 0] = 0.5

    options = dict(
        device=device, dx=dx, dt=dt, source_amplitudes=source,
        source_location=location, receiver_location=location, pmlthick=pml,
        fdtd_order=2, mode=2 if dims == 2 else 3,
        source_direction=component, receiver_component=component,
        model_gradient_sampling_interval=1, wavefield_storage_dtype=storage_dtype,
        wavefield_rhs_history=rhs,
        wavefield_conversion_backend=conversion_backend,
    )
    if compression:
        options.update(wavefield_compression=compression,
                       wavefield_compression_block_size=(4, 4) if dims == 2 else (4, 4, 1))

    def objective(er, se, backward=False):
        owned = [s.detach().clone() for s in states]
        with _path(path):
            result = DeepGPR.compute(
                eps_r=er, sigma=se, E=tuple(owned[:3]), H=tuple(owned[3:6]),
                PML=tuple(owned[6:]), **options,
            )
        data = result[-1]
        fast = bool(data.grad_fn.config.tm2d_fast_path) if data.grad_fn else None
        loss = 0.5 * data.double().square().sum()
        if backward:
            loss.backward()
        return float(loss.detach()), float(data.detach().flatten()[0]), fast

    er = eps.detach().clone().requires_grad_(True)
    se = sigma.detach().clone().requires_grad_(True)
    loss, data, fast = objective(er, se, backward=True)
    physical_cell = cell[:dims]
    finite_differences = {}
    for parameter, baseline, h in (("eps", eps, 0.04), ("sigma", sigma, 0.003)):
        direction = torch.zeros_like(baseline)
        direction[physical_cell] = h
        if parameter == "eps":
            plus, _, _ = objective(eps + direction, sigma)
            minus, _, _ = objective(eps - direction, sigma)
        else:
            plus, _, _ = objective(eps, sigma + direction)
            minus, _, _ = objective(eps, sigma - direction)
        finite_differences[parameter] = (plus - minus) / (2 * h)

    # One-step analytic map: Enew = Eold + cb * correction.
    # Sources contribute -source/dx^2; the selected CPML auxiliary adds +0.5.
    correction = (0.5 if cpml_correction else 0.0) - source_value / dx**2
    epsilon0 = DeepGPR.config.constants.VACUUM_PERMITTIVITY
    cb = 1.0 / (epsilon0 * 4.0 / dt + 0.5 * 0.01)
    analytic_eps = data * correction * (-epsilon0 / dt) * cb**2
    analytic_sigma = data * correction * (-0.5) * cb**2
    return dict(
        path=path, component=component, dims=dims, rhs=rhs,
        source_value=source_value, cpml_correction=cpml_correction,
        compression=compression, storage_dtype=str(storage_dtype),
        conversion_backend=conversion_backend, cell=list(physical_cell), J=loss, data=data,
        tm2d_fast_path=fast,
        ad_eps=float(er.grad[physical_cell]), ad_sigma=float(se.grad[physical_cell]),
        fd_eps=finite_differences["eps"], fd_sigma=finite_differences["sigma"],
        analytic_eps=analytic_eps, analytic_sigma=analytic_sigma,
    )


class UnupdatedElectricMaterialGradientTests(unittest.TestCase):
    def check_case(self, row):
        relative = 0.008 if row["compression"] or row["storage_dtype"] != "torch.float32" else 0.0005
        for parameter in ("eps", "sigma"):
            expected = row[f"analytic_{parameter}"]
            tolerance = max(2.0e-6, abs(expected) * relative)
            self.assertAlmostEqual(row[f"ad_{parameter}"], expected, delta=tolerance, msg=str(row))
            self.assertAlmostEqual(row[f"fd_{parameter}"], expected,
                                   delta=max(2.0e-5, abs(expected) * 0.001), msg=str(row))
        if row["path"] == "cuda_fast":
            self.assertTrue(row["tm2d_fast_path"])
        elif row["path"] == "cuda_general":
            self.assertFalse(row["tm2d_fast_path"])

    def cases(self, path, components):
        for dims, component in components:
            for rhs in ("stored", "reconstructed"):
                for source in (0.0, 0.0002):
                    with self.subTest(path=path, dims=dims, component=component, rhs=rhs, source=source):
                        self.check_case(material_case(path, component, dims, rhs, source))

    def test_cpu_unupdated_components_and_boundary_sources(self):
        self.cases("cpu", ((2, 2), (3, 0), (3, 1), (3, 2)))

    def test_cpu_unupdated_component_keeps_cpml_correction(self):
        for rhs in ("stored", "reconstructed"):
            self.check_case(material_case("cpu", 1, 3, rhs, cpml_correction=True))

    def test_cpu_lossy_rhs_history_keeps_actual_component_coefficient(self):
        for dtype in (torch.float16, torch.bfloat16):
            for source in (0.0, 0.0002):
                with self.subTest(dtype=dtype, source=source):
                    self.check_case(material_case("cpu", rhs="stored", source_value=source,
                                                  storage_dtype=dtype))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fast_unupdated_component_and_boundary_source(self):
        self.cases("cuda_fast", ((2, 2),))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_general_unupdated_components_and_boundary_sources(self):
        self.cases("cuda_general", ((2, 2), (3, 0), (3, 1), (3, 2)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_general_unupdated_component_keeps_cpml_correction(self):
        for rhs in ("stored", "reconstructed"):
            self.check_case(material_case("cuda_general", 1, 3, rhs, cpml_correction=True))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_int8_unupdated_component_and_boundary_source(self):
        for path in ("cuda_fast", "cuda_general"):
            for rhs in ("stored", "reconstructed"):
                for source in (0.0, 0.0002):
                    with self.subTest(path=path, rhs=rhs, source=source):
                        self.check_case(material_case(path, 2, 2, rhs, source, compression="int8"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_lossy_rhs_history_keeps_actual_component_coefficient(self):
        for path in ("cuda_fast", "cuda_general"):
            for backend in ("legacy", "native_scalar", "native_vec2"):
                for dtype in (torch.float16, torch.bfloat16):
                    for source in (0.0, 0.0002):
                        with self.subTest(path=path, backend=backend, dtype=dtype, source=source):
                            self.check_case(material_case(path, rhs="stored", source_value=source,
                                                          storage_dtype=dtype, conversion_backend=backend))


if __name__ == "__main__":
    unittest.main()
