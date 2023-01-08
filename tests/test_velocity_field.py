import sys
import unittest
import torch

sys.path.append("..")

# import vector_field.VectorField

from src.gp_flows.velocity_field import VelocityField
from src.tool_box import compute_div
from src.case import Case


def init_v(v):
    """
    for vi in v.v:
        for d in vi.denses:
            d.bias = torch.nn.Parameter(torch.randn(d.bias.shape))
            d.weight = torch.nn.Parameter(torch.randn(d.weight.shape))
    """
    for d in v.v.denses:
        d.bias = torch.nn.Parameter(torch.randn(d.bias.shape))
        d.weight = torch.nn.Parameter(torch.randn(d.weight.shape))


class TestVelocityField(unittest.TestCase):
    def setUp(self):
        seed_int = 123
        torch.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Only with cpu
        torch.use_deterministic_algorithms(True)

    @classmethod
    def setUpClass(cls):
        pass

    def test_incompressible(self):
        dim = 5
        nb_neurons = [5, 5, 5]
        x = torch.rand(50, dim)

        v1 = VelocityField(
            dim,
            nb_neurons,
            time_state=Case.continuous_time,
            incompressible=True,
        )
        init_v(v1)
        div = compute_div(v1, x)
        self.assertAlmostEqual(torch.norm(div), 0.0, delta=5e-5)

        v2 = VelocityField(
            dim,
            nb_neurons,
            time_state=Case.continuous_time,
            incompressible=True,
        )
        init_v(v2)
        div = compute_div(v2, x)
        self.assertAlmostEqual(torch.norm(div), 0.0, delta=5e-5)

        v3 = VelocityField(
            dim,
            nb_neurons,
            time_state=Case.continuous_time,
            incompressible=True,
        )
        init_v(v3)
        div = compute_div(v3, x)
        self.assertAlmostEqual(torch.norm(div), 0.0, delta=5e-5)

        v3 = VelocityField(
            dim,
            nb_neurons,
            time_state=Case.continuous_time,
            incompressible=True,
        )
        init_v(v3)
        div = compute_div(v3, x)
        self.assertAlmostEqual(torch.norm(div), 0.0, delta=5e-5)

        v4 = VelocityField(
            dim,
            nb_neurons,
            time_state=Case.continuous_time,
            incompressible=True,
            nb_div_blocks=3,
        )
        init_v(v4)

        div = compute_div(v4, x)
        self.assertAlmostEqual(torch.norm(div), 0.0, delta=5e-5)

        """
        v5 = VelocityField(
            dim,
            nb_neurons,
            time_state=Case.continuous_time,
            incompressible=True,
        )
        init_v(v5)
        div = compute_div(v5, x)
        self.assertAlmostEqual(torch.norm(div), 0.0, delta=5e-5)
        """

    def test_derivatives(self):
        dtype = torch.double
        torch.set_default_dtype(dtype)
        dim = 4
        t = torch.tensor(0.5)
        nb_neurons = [5, 5, 5]
        x = torch.rand(50, dim)

        v = VelocityField(
            dim,
            nb_neurons,
            time_state=Case.continuous_time,
            incompressible=True,
        )
        init_v(v)
        x.requires_grad_(True)
        t.requires_grad_(True)
        lagrangian_deriv = v.lagrangian_deriv(x, t)

        dt = 1e-8
        approx_lagrangian_deriv = v.approx_lagrangian_deriv(x, t, dt)
        err = torch.norm(
            lagrangian_deriv - approx_lagrangian_deriv, dim=1
        ).max()
        self.assertAlmostEqual(err, 0.0, delta=1e-3)

        # err = torch.norm(approx_lagrangian_deriv-approx_tot_deriv, dim=1).max()
        # self.assertAlmostEqual(err, 0., delta=1e-6)


if __name__ == "__main__":
    unittest.main()
