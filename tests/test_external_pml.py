"""Physical-model API, differentiable PML extension, and checkpoint regressions."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import DeepGPR

common = importlib.import_module('DeepGPR.common')
# compute() and the autograd bridge moved; patch the modules that use the names.
compute_module = importlib.import_module('DeepGPR.solver.modeling')
autograd_module = importlib.import_module('DeepGPR.solver.autograd')


def problem(shape=(8, 9), pml=(2, 3, 1, 2), device='cpu'):
    model = torch.linspace(3.0, 5.0, steps=int(torch.tensor(shape).prod()), device=device).reshape(shape)
    sigma = torch.full_like(model, 2.e-4)
    mu = torch.linspace(1., 1.2, steps=model.numel(), device=device).reshape(shape)
    nt = 35
    source = torch.zeros((1, nt, 1), device=device)
    source[0, 1:4, 0] = torch.tensor([1., -.5, .25], device=device)
    z = 0 if len(shape) == 2 or shape[2] == 1 else 2
    args = dict(device=device, dx=(.02, .017, .015), dt=1.e-11,
                source_amplitudes=source,
                source_location=torch.tensor([[[0, 3, z]], [[3, 4, z]]], device=device),
                receiver_location=torch.tensor([[[0, 3, z], [3, 4, z]], [[3, 4, z], [2, 3, z]]], device=device),
                pmlthick=pml, mu_r=mu, mode=2 if z == 0 else 3)
    return model, sigma, args


def pad_model(model, pml):
    model = model.reshape(*model.shape[:2], -1)
    return F.pad(model[None, None], (pml[4], pml[5], pml[2], pml[3], pml[0], pml[1]), mode='replicate')[0, 0]


def raw_compute(model, sigma, args, pml):
    """Reference native call with explicitly supplied solver-grid materials."""
    original = common.initialization

    def prepared(*values):
        values = list(values)
        values[-2] = 0  # Validate full-grid coordinates without extending again.
        result = list(original(*values))
        result[14] = torch.tensor(pml, dtype=torch.int32)
        return tuple(result)

    with patch.object(compute_module, 'initialization', prepared), patch.object(
        compute_module, '_shift_locations', lambda loc, p, device: loc
    ):
        return DeepGPR.compute(eps_r=model, sigma=sigma, **args)


class ExternalPMLTests(unittest.TestCase):
    def test_replicated_material_values_and_folded_backward(self):
        torch.manual_seed(612)
        for shape, pml in [
            ((3, 4, 1), (5, 2, 0, 3, 0, 0)),
            ((3, 4, 2), (1, 2, 3, 0, 2, 1)),
            ((1, 2, 1), (3, 4, 2, 1, 2, 3)),
            ((3, 4, 2), (0, 0, 0, 0, 0, 0)),
        ]:
            with self.subTest(shape=shape):
                model = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float64).reshape(shape).requires_grad_()
                reference = model.detach().clone().requires_grad_()
                extended = common._ExtendModel.apply(model, pml)
                padded = pad_model(reference, pml)
                torch.testing.assert_close(extended, padded, rtol=0, atol=0)
                dual = torch.randn_like(extended)
                extended.backward(dual)
                padded.backward(dual)
                torch.testing.assert_close(model.grad, reference.grad, rtol=1.e-13, atol=1.e-13)

                # With a unit dual, a corner receives the product of the
                # replication counts of all its incident faces.
                model.grad = None
                reference.grad = None
                common._ExtendModel.apply(model, pml).sum().backward()
                pad_model(reference, pml).sum().backward()
                torch.testing.assert_close(model.grad, reference.grad, rtol=0, atol=0)

    def test_automatic_extension_matches_manual_solver_grid(self):
        for shape, pml in [((8, 9), (2, 3, 1, 2, 0, 0)), ((8, 9, 1), (0, 3, 2, 0, 0, 0)), ((6, 7, 5), (2, 1, 1, 2, 1, 3))]:
            for order in (2, 4, 8):
                with self.subTest(shape=shape, order=order):
                    model, sigma, args = problem(shape, pml)
                    args['fdtd_order'] = order
                    model.requires_grad_()
                    sigma.requires_grad_()
                    loc_before = args['source_location'].clone()
                    auto = DeepGPR.compute(eps_r=model, sigma=sigma, **args)
                    auto[-1].square().sum().backward()
                    full_model = pad_model(model.detach(), pml).requires_grad_()
                    full_sigma = pad_model(sigma.detach(), pml).requires_grad_()
                    offset = torch.tensor(pml[::2])
                    raw_args = {**args, 'mu_r': pad_model(args['mu_r'], pml),
                                'source_location': args['source_location'] + offset,
                                'receiver_location': args['receiver_location'] + offset}
                    raw = raw_compute(full_model, full_sigma, raw_args, pml)
                    raw[-1].square().sum().backward()
                    for a, b in zip((auto[0], *auto[1], *auto[2], *auto[3], auto[-1]),
                                    (raw[0], *raw[1], *raw[2], *raw[3], raw[-1])):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    for a, b in [(model.grad, full_model.grad), (sigma.grad, full_sigma.grad)]:
                        reference = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
                        pad_model(reference, pml).backward(b.double())
                        # The independent PyTorch transpose accumulates in a
                        # different order from the deterministic native fold.
                        torch.testing.assert_close(a.double(), reference.grad, rtol=2.e-6, atol=1.e-5)
                    torch.testing.assert_close(args['source_location'], loc_before)
                    self.assertGreater(model.grad[0].abs().max().item(), 0)

    def test_model_edge_derivative_reextends_materials_on_every_forward(self):
        model, sigma, args = problem()
        sigma.fill_(.01)
        model.requires_grad_()
        sigma.requires_grad_()
        result = DeepGPR.compute(eps_r=model, sigma=sigma, **args)
        scale = result[-1].detach().double().abs().max()
        (.5 * (result[-1].double() / scale).square().sum()).backward()
        for name, values, steps in [
            ('eps', model, (.04, .02, .01, .005)),
            ('sigma', sigma, (.003, .0015, .00075, .000375)),
        ]:
            errors = []
            for h in steps:
                losses = []
                for sign in (1, -1):
                    er, se = model.detach().clone(), sigma.detach().clone()
                    (er if name == 'eps' else se)[0, 3] += sign*h
                    with torch.no_grad():
                        data = DeepGPR.compute(eps_r=er, sigma=se, save_wavefield_history=False, **args)[-1]
                    losses.append(.5 * (data.double() / scale).square().sum())
                fd = float((losses[0] - losses[1]) / (2*h))
                ad = float(values.grad[0, 3])
                errors.append(abs(fd - ad) / max(abs(fd), abs(ad), 1.e-20))
            with self.subTest(parameter=name, errors=errors):
                self.assertTrue(any(max(a, b) < .003 for a, b in zip(errors[:-1], errors[1:])))

    def test_checkpoint_recomputation_and_fwi_updates(self):
        for shape, pml, reentrant in [
            ((8, 9), (2, 3, 1, 2), True),
            ((6, 7, 5), (1, 2, 2, 1, 1, 2), True),
            ((8, 9), (2, 3, 1, 2), False),
            ((6, 7, 5), (1, 2, 2, 1, 1, 2), False),
        ]:
            with self.subTest(shape=shape, reentrant=reentrant):
                base, sigma_base, args = problem(shape, pml)
                model = base.clone().requires_grad_()
                sigma = sigma_base.clone().requires_grad_()
                optimizer = torch.optim.SGD([model, sigma], lr=1.e-7)
                for iteration in range(2):
                    ref_model = model.detach().clone().requires_grad_()
                    ref_sigma = sigma.detach().clone().requires_grad_()
                    ref_source = args['source_amplitudes'].clone().requires_grad_()
                    full = DeepGPR.compute(eps_r=ref_model, sigma=ref_sigma, **{**args, 'source_amplitudes': ref_source})
                    full[-1].square().sum().backward()
                    source = args['source_amplitudes'].clone().requires_grad_()
                    initial_args = {k:v for k,v in args.items() if k not in ('mode', 'mu_r')}
                    E, H, PML = DeepGPR.checkpoint_initial_field(er=model, se=sigma, mr=args['mu_r'], **initial_args)
                    state = (*E, *H, *PML)
                    traces = []
                    for start in range(0, source.shape[1], 12):
                        def segment(er, se, wave, *boundary):
                            work = tuple(t.clone() for t in boundary)
                            r = DeepGPR.compute(eps_r=er, sigma=se, **{**args, 'source_amplitudes': wave},
                                                E=work[:3], H=work[3:6], PML=work[6:])
                            return (*r[1], *r[2], *r[3], r[-1])
                        out = checkpoint(segment, model, sigma, source[:, start:start+12], *state, use_reentrant=reentrant)
                        state, data = out[:-1], out[-1]
                        traces.append(data)
                    data = torch.cat(traces, dim=1)
                    torch.testing.assert_close(data, full[-1], rtol=0, atol=0)
                    for a, b in zip(state, (*full[1], *full[2], *full[3])):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    optimizer.zero_grad()
                    data.square().sum().backward()
                    for a,b in [(model.grad, ref_model.grad), (sigma.grad, ref_sigma.grad), (source.grad, ref_source.grad)]:
                        # Segmented and full adjoints sum in a different order; scale
                        # atol with the gradient (|g| ~ 1e6-1e7 here) for float32 round-off.
                        torch.testing.assert_close(a, b, rtol=2.e-5, atol=max(1.e-5, 1.e-7 * float(b.abs().max())))
                    optimizer.step()
                    with torch.no_grad():
                        model.clamp_(min=1.0)
                        sigma.clamp_(min=0.0)
                self.assertFalse(torch.equal(model, base))

    def test_physical_coordinate_and_state_validation(self):
        model, sigma, args = problem()
        # Thickness is independent of physical model size; no overlap rejection.
        r = DeepGPR.compute(eps_r=model, sigma=sigma, **{**args, 'pmlthick': 12})
        self.assertEqual(tuple(r[1][0].shape), (2, 33, 34, 2))
        with self.assertRaisesRegex(ValueError, 'out of range'):
            DeepGPR.compute(eps_r=model, sigma=sigma, **{**args, 'source_location': args['source_location'] + torch.tensor([8, 0, 0])})
        with self.assertRaisesRegex(ValueError, 'z boundaries'):
            DeepGPR.compute(eps_r=model, sigma=sigma, **{**args, 'pmlthick': [1]*6})
        with self.assertRaisesRegex(ValueError, 'Field shape mismatch'):
            DeepGPR.compute(eps_r=model, sigma=sigma, E=r[1], **args)
        r = DeepGPR.compute(eps_r=model, sigma=sigma, **{**args, 'pmlthick': 0, 'save_wavefield_history': False})
        self.assertEqual(tuple(r[1][0].shape), (2, 9, 10, 2))
        self.assertEqual(r[0].numel(), 0)

    def test_absorption_speed_must_be_fixed_positive_and_finite(self):
        model, sigma, args = problem()
        for speed in (0., -1., float('nan'), float('inf'), True,
                      torch.tensor(2.e8, requires_grad=True)):
            with self.subTest(speed=speed):
                with self.assertRaises((ValueError, TypeError)):
                    DeepGPR.compute(eps_r=model, sigma=sigma, pml_max_velocity=speed, **args)

    def test_checkpoint_shot_subset_with_disabled_faces(self):
        for shape, pml in [((8, 9), (0, 2, 1, 0)), ((6, 7, 5), (0, 1, 0, 2, 0, 1)), ((8, 9), 0)]:
            model, sigma, args = problem(shape, pml)
            initial_args = {k:v for k,v in args.items() if k not in ('mode', 'mu_r')}
            E, H, PML = DeepGPR.checkpoint_initial_field(er=model, se=sigma, mr=args['mu_r'], per_nstep=1, **initial_args)
            result = DeepGPR.compute(eps_r=model, sigma=sigma, E=E, H=H, PML=PML, **{
                **args, 'source_location': args['source_location'][:1],
                'receiver_location': args['receiver_location'][:1]})
            self.assertEqual(result[-1].shape[0], 1)
            with self.assertRaisesRegex(ValueError, 'per_nstep'):
                DeepGPR.checkpoint_initial_field(er=model, se=sigma, per_nstep=3, **initial_args)

    def test_cfl_uses_axes_activated_by_extension(self):
        location = torch.tensor([[[0, 2, 0]]])
        with self.assertRaisesRegex(ValueError, 'CFL'):
            common.initialization('cpu', torch.full((1, 5), 4.), torch.zeros((1, 5)),
                                  None, torch.zeros((1, 2, 1)), location, location,
                                  (.001, .02, .02), 1.e-11, 2)

    def test_stale_native_library_is_rejected(self):
        model, sigma, args = problem()
        with patch.object(autograd_module, 'get_deepgpr_lib', return_value=object()):
            with self.assertRaisesRegex(RuntimeError, 'external-PML'):
                DeepGPR.compute(eps_r=model, sigma=sigma, **args)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_physical_model_gradient_parity(self):
        outputs = []
        for device in ('cpu', 'cuda'):
            model, sigma, args = problem(device=device)
            model.requires_grad_()
            sigma.requires_grad_()
            result = DeepGPR.compute(eps_r=model, sigma=sigma, **args)
            result[-1].square().sum().backward()
            outputs.append([t.detach().cpu() for t in (result[-1], model.grad, sigma.grad)])
        for a,b in zip(*outputs):
            torch.testing.assert_close(a,b,rtol=2.e-4,atol=1.e-5)


if __name__ == '__main__':
    unittest.main()
