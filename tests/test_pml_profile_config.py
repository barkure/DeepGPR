"""Fixed absorber profiles, public aliases and stale-library CPU regressions."""

import importlib
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import DeepGPR  # noqa: E402
from DeepGPR.config.constants import SPEED_OF_LIGHT  # noqa: E402
from DeepGPR.native.loader import get_deepgpr_lib  # noqa: E402
from DeepGPR.solver.autograd import _check_forward_capabilities  # noqa: E402
from DeepGPR.solver.pml import build_pml_coeffs  # noqa: E402
from DeepGPR.utils.exceptions import NativeLibraryError  # noqa: E402

autograd_module = importlib.import_module("DeepGPR.solver.autograd")
modeling_module = importlib.import_module("DeepGPR.solver.modeling")


def profile_arguments():
    eps = torch.full((8, 9, 6), 4.0, dtype=torch.float32)
    return dict(
        eps_r=eps,
        mu_r=torch.ones_like(eps),
        dt=1.0e-11,
        dx=(0.02, 0.017, 0.015),
        nx=8,
        ny=9,
        nz=6,
        pmlthick=torch.tensor([2, 1, 0, 2, 1, 0], dtype=torch.int32),
        device="cpu",
        dtype=torch.float32,
    )


def cpu_problem(requires_grad=False):
    eps = torch.full((6, 7), 4.0, dtype=torch.float32, requires_grad=requires_grad)
    sigma = torch.full_like(eps, 2.0e-4)
    source = torch.zeros((1, 6, 1), dtype=torch.float32)
    source[0, 1, 0] = 1.0
    return dict(
        device="cpu",
        dx=0.02,
        dt=1.0e-11,
        eps_r=eps,
        sigma=sigma,
        source_amplitudes=source,
        source_location=torch.tensor([[[2, 2, 0]]]),
        receiver_location=torch.tensor([[[2, 3, 0]]]),
        pmlthick=2,
        wavefield_rhs_history="stored",
    )


class LibraryWithoutPMLMaterialProbe:
    """Expose the current CPU ABI except the new optional capability symbol."""

    def __init__(self, library):
        self.library = library

    def __getattr__(self, name):
        if name == "deepgpr_supports_pml_material_gradients":
            raise AttributeError(name)
        return getattr(self.library, name)


class PMLProfileConfigTests(unittest.TestCase):
    def test_compute_rejects_invalid_speed_before_model_initialization(self):
        invalid = (
            0.0, -1.0, float("nan"), float("inf"), -float("inf"),
            True, False, None, "invalid", 1.0 + 2.0j,
            torch.tensor(SPEED_OF_LIGHT, requires_grad=True),
        )
        with patch.object(modeling_module, "initialization") as initialize:
            for speed in invalid:
                with self.subTest(speed=speed):
                    with self.assertRaises((TypeError, ValueError)):
                        DeepGPR.compute(device="cpu", pml_max_velocity=speed)
            initialize.assert_not_called()

    def test_low_level_builder_rejects_invalid_speed(self):
        invalid = (
            0.0, -1.0, float("nan"), float("inf"), -float("inf"),
            True, False, None, "invalid", 1.0 + 2.0j,
            torch.tensor(SPEED_OF_LIGHT, requires_grad=True),
        )
        for speed in invalid:
            with self.subTest(speed=speed):
                with self.assertRaises((TypeError, ValueError)):
                    build_pml_coeffs(**profile_arguments(), max_velocity=speed)

    def test_unrepresentable_float32_profiles_are_rejected(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            for speed in (1.0e308, 1.0e-320, 1.0e-250):
                with self.subTest(speed=speed):
                    with self.assertRaisesRegex(ValueError, "unrepresentable"):
                        build_pml_coeffs(**profile_arguments(), max_velocity=speed)
        finally:
            torch.set_default_dtype(previous)

    def test_extreme_speeds_never_return_nonfinite_coefficients(self):
        previous = torch.get_default_dtype()
        try:
            for default_dtype in (torch.float32, torch.float64):
                torch.set_default_dtype(default_dtype)
                for speed in (1.0e-320, 1.0e-250, 1.0e30, 1.0e308):
                    with self.subTest(default_dtype=default_dtype, speed=speed):
                        try:
                            arrays = build_pml_coeffs(**profile_arguments(), max_velocity=speed)
                        except ValueError:
                            continue
                        for coefficient in arrays[6:]:
                            self.assertTrue(bool(torch.isfinite(coefficient).all()))
        finally:
            torch.set_default_dtype(previous)

    def test_profiles_are_independent_of_changing_boundary_materials(self):
        args = profile_arguments()
        reference = build_pml_coeffs(**args, max_velocity=2.0e8)
        args["eps_r"] = torch.linspace(1.0, 15.0, args["eps_r"].numel()).reshape(
            args["eps_r"].shape
        ).requires_grad_()
        args["mu_r"] = torch.full_like(args["mu_r"], 3.0, requires_grad=True)
        changed = build_pml_coeffs(**args, max_velocity=2.0e8)
        for before, after in zip(reference, changed):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertFalse(after.requires_grad)
        slower = build_pml_coeffs(**args, max_velocity=1.0e8)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(reference[6:], slower[6:])))

    def test_default_reference_matches_explicit_vacuum_speed(self):
        default = build_pml_coeffs(**profile_arguments())
        explicit = build_pml_coeffs(**profile_arguments(), max_velocity=SPEED_OF_LIGHT)
        scalar_tensor = build_pml_coeffs(
            **profile_arguments(), max_velocity=torch.tensor(SPEED_OF_LIGHT, dtype=torch.float64)
        )
        for a, b, c in zip(default, explicit, scalar_tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_compute_aliases_expose_same_keyword_only_parameter(self):
        # Some existing test modules reload DeepGPR during discovery. Resolve
        # all aliases from the same module generation at execution time.
        current_compute = importlib.import_module("DeepGPR").compute
        aliases = (
            current_compute,
            importlib.import_module("DeepGPR.compute2").compute,
            importlib.import_module("DeepGPR.solver").compute,
            importlib.import_module("DeepGPR.solver.modeling").compute,
        )
        for function in aliases:
            self.assertIs(function, current_compute)
            parameter = inspect.signature(function).parameters["pml_max_velocity"]
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(parameter.default, SPEED_OF_LIGHT)

    def test_legacy_profile_wrappers_forward_reference_velocity(self):
        args = profile_arguments()
        expected = build_pml_coeffs(**args, max_velocity=1.7e8)
        args["er"] = args.pop("eps_r")
        args["mr"] = args.pop("mu_r")
        for module_name in ("DeepGPR", "DeepGPR.common", "DeepGPR.solver"):
            wrapper = importlib.import_module(module_name).buildpmlcoeffs
            for actual, reference in zip(wrapper(**args, max_velocity=1.7e8), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_capability_probe_is_required_only_for_pml_material_gradients(self):
        config = SimpleNamespace(
            pmlthick=torch.tensor([2, 2, 2, 2, 0, 0], dtype=torch.int32),
            storage=SimpleNamespace(
                conversion_backend="legacy", compression="none", int8_reduction_backend="current"
            ),
            tm2d_fast_path=False,
            history_region="extended",
            reconstruct_rhs=False,
            save_wavefield_history=True,
        )
        stale = SimpleNamespace(deepgpr_supports_external_pml=lambda: 1)
        _check_forward_capabilities(stale, config, needs_model_gradient=False)
        with self.assertRaisesRegex(NativeLibraryError, "omits PML material sensitivities"):
            _check_forward_capabilities(stale, config, needs_model_gradient=True)
        stale.deepgpr_supports_pml_material_gradients = lambda: 0
        with self.assertRaises(NativeLibraryError):
            _check_forward_capabilities(stale, config, needs_model_gradient=True)
        stale.deepgpr_supports_pml_material_gradients = lambda: 1
        _check_forward_capabilities(stale, config, needs_model_gradient=True)
        config.pmlthick.zero_()
        _check_forward_capabilities(SimpleNamespace(), config, needs_model_gradient=True)

    def test_public_compute_rejects_stale_library_before_native_forward(self):
        library = get_deepgpr_lib("cpu")
        stale = LibraryWithoutPMLMaterialProbe(library)
        with patch.object(autograd_module, "get_deepgpr_lib", return_value=stale), patch.object(
            autograd_module, "invoke"
        ) as invoke:
            with self.assertRaisesRegex(NativeLibraryError, "omits PML material sensitivities"):
                DeepGPR.compute(**cpu_problem(requires_grad=True))
            invoke.assert_not_called()

    def test_stale_probe_does_not_block_forward_only_with_physical_history(self):
        library = get_deepgpr_lib("cpu")
        stale = LibraryWithoutPMLMaterialProbe(library)
        args = cpu_problem(requires_grad=True)
        with patch.object(autograd_module, "get_deepgpr_lib", return_value=stale):
            result = DeepGPR.compute(
                **args, save_wavefield_history=False, wavefield_history_region="physical"
            )
        self.assertEqual(result[0].numel(), 0)
        self.assertTrue(bool(torch.isfinite(result[-1]).all()))
        self.assertGreater(float(result[-1].detach().abs().max()), 0)


if __name__ == "__main__":
    unittest.main()
