import sys

sys.path.append("..")

import unittest
from src.gp_flows.vector_field import VectorField
import torch


class TestVectorField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_grad(self):
        dim = 4
        v = VectorField(dim, 1, nb_neurons=[5, 5, 5], bias=None)
        x = torch.rand(50, dim).requires_grad_(True)
        with torch.enable_grad():
            v_x = v.eval(x)
        grad1 = torch.autograd.grad(
            v_x, x, torch.ones_like(v_x), create_graph=True
        )[0]
        grad2 = v.jacobian(x).squeeze(1)
        self.assertAlmostEqual(torch.norm(grad1 - grad2), 0.0, delta=5e-6)


if __name__ == "__main__":
    unittest.main()
