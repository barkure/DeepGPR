"""Async workspaces must not retain field values or alias concurrent calls."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock
import warnings

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import DeepGPR
from DeepGPR.native import loader


class ResourceCacheApiTests(unittest.TestCase):
    def test_cpu_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            DeepGPR.clear_cuda_resource_cache("cpu")

    def test_older_library_without_cache_is_a_noop(self):
        with mock.patch.object(torch.cuda, "device") as context, \
                mock.patch.object(loader, "get_deepgpr_lib", return_value=object()):
            DeepGPR.clear_cuda_resource_cache(1)
        context.assert_called_once_with(torch.device("cuda:1"))
        context.return_value.__exit__.assert_called_once()


def run_case(shape=(17, 19), dtype=torch.float32, rhs="stored", offload=True,
             material=True, device="cuda:0", scale=1.0, interval=1, nt=24):
    device = torch.device(device)
    three_d = len(shape) == 3
    eps = (torch.arange(torch.tensor(shape).prod().item(), device=device)
           .reshape(shape).float() % 13 * 0.02 + 4).requires_grad_(material)
    sigma = torch.full(shape, 2e-4, device=device, requires_grad=material)
    dt = 1.5e-11
    source = (scale * DeepGPR.ricker(8e8, nt, dt, 1e-10).to(device)
              .reshape(1, nt, 1)).requires_grad_()
    src = torch.tensor([[[4, 4, 3 if three_d else 0]]], device=device)
    rec = torch.tensor([[[5, 5, 3 if three_d else 0]]], device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = DeepGPR.compute(
            device=device, dx=0.02, dt=dt, eps_r=eps, sigma=sigma,
            source_amplitudes=source, source_location=src, receiver_location=rec,
            pmlthick=2, mode=3 if three_d else 2, fdtd_order=4,
            source_direction=0 if three_d else 2,
            use_async_offload=offload, wavefield_storage_dtype=dtype,
            wavefield_rhs_history=rhs, model_gradient_sampling_interval=interval,
        )
    node = out[-1].grad_fn
    values = [out[-1], *out[1], *out[2], *out[3]]
    for name in ("E_saved", "R_saved", "E_final"):
        value = getattr(node, name, None)
        if isinstance(value, torch.Tensor):
            values.append(value)
    values = [value.detach().cpu().clone() for value in values]
    out[-1].square().sum().backward()
    gradients = [source.grad]
    if material:
        gradients += [eps.grad, sigma.grad]
    return values, [value.detach().cpu().clone() for value in gradients]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaResourceCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(DeepGPR.get_deepgpr_lib("cuda"), "deepgpr_clear_cuda_resource_cache"):
            raise unittest.SkipTest("Rebuild the CUDA backend with resource-cache support")

    def tearDown(self):
        DeepGPR.clear_cuda_resource_cache()

    def assert_results(self, expected, actual, three_d=False):
        for group, (left, right) in enumerate(zip(expected, actual)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertTrue(torch.isfinite(b).all())
                # The general 3D adjoint uses atomics; forwards and 2D are exact.
                atomic = three_d and group == 1
                atol = 1e-6 * a.abs().max().item() if atomic else 0
                torch.testing.assert_close(
                    a, b, rtol=2e-5 if atomic else 0, atol=atol,
                )
                if atomic:
                    self.assertLessEqual((a - b).norm().item(), 1e-6 * a.norm().item())

    def test_reuse_across_shapes_storage_and_history_layouts(self):
        cases = [
            dict(shape=(17, 19), dtype=torch.float32, rhs="stored"),
            dict(shape=(23, 27), dtype=torch.float16, rhs="stored", interval=2),
            dict(shape=(11, 13, 9), dtype=torch.bfloat16, rhs="stored"),
            dict(shape=(13, 15), dtype=torch.float32, rhs="reconstructed"),
            dict(shape=(19, 21), dtype=torch.float16, rhs="reconstructed"),
            dict(shape=(17, 19), material=False),
        ]
        references = [run_case(offload=False, **case) for case in cases]
        for _ in range(2):
            for case, expected in zip(cases, references):
                with self.subTest(case=case):
                    self.assert_results(expected, run_case(**case), len(case["shape"]) == 3)

    def test_clear_and_nondefault_stream(self):
        expected = run_case()
        for _ in range(3):
            DeepGPR.clear_cuda_resource_cache()
            with torch.cuda.stream(torch.cuda.Stream()):
                actual = run_case()
            self.assert_results(expected, actual)

    def test_concurrent_calls_and_clear(self):
        scales = (0.7, 1.3)
        expected = [run_case(scale=scale) for scale in scales]
        barrier = threading.Barrier(3)

        def worker(scale):
            with torch.cuda.device(0), torch.cuda.stream(torch.cuda.Stream()):
                barrier.wait(timeout=30)
                return run_case(scale=scale)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(3):
                futures = [pool.submit(worker, scale) for scale in scales]
                barrier.wait(timeout=30)
                DeepGPR.clear_cuda_resource_cache()
                for reference, future in zip(expected, futures):
                    self.assert_results(reference, future.result(timeout=60))

    @unittest.skipUnless(torch.cuda.device_count() > 1, "Requires two CUDA devices")
    def test_device_isolation_and_restoration(self):
        original = torch.cuda.current_device()
        expected = run_case(device="cuda:1")
        run_case(device="cuda:0", shape=(23, 25))
        DeepGPR.clear_cuda_resource_cache("cuda:1")
        self.assertEqual(torch.cuda.current_device(), original)
        self.assert_results(expected, run_case(device="cuda:1"))
        DeepGPR.clear_cuda_resource_cache("cuda:1")


if __name__ == "__main__":
    unittest.main()
