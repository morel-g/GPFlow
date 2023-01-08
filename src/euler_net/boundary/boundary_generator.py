import torch
import numpy as np
from case import Case
from precision import torch_float_precision


class BoundaryGenerator:
    def __init__(self, d, boundary_dict):
        self.d = d
        self.boundary_dict = boundary_dict
        self.device = "cpu"
        self.boundary_case = boundary_dict["case"]

    def generate(self, nb_points, neumann=False, periodic=False):
        if self.boundary_case == Case.circle:
            return self.generate_circle(nb_points, neumann)
        elif self.boundary_case == Case.rectangle:
            return self.generate_rectangle(nb_points, periodic)
        else:
            raise RuntimeError("Unknown boundary domain for generation.")

    def generate_circle(self, nb_points, neumann=False):
        if self.d != 2:
            RuntimeError("Generate circle works only for d=2.")
        r = self.boundary_dict["radius"]
        theta, w = np.polynomial.legendre.leggauss(
            50
        )  # torch.rand(nb_points, 1, device=self.device) * 2. * np.pi
        theta *= np.pi
        theta = torch.tensor(
            theta, device=self.device, dtype=torch_float_precision
        )
        w = (
            torch.tensor(w, device=self.device, dtype=torch_float_precision)
            * np.pi
        )
        """
        if not neumann:
            return torch.cat((r * torch.cos(theta), r * torch.sin(theta)), dim=1)
        else:
        """
        normal = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        return r * normal, normal, w

    def generate_rectangle(self, nb_points, periodic=False):
        if self.d != 2 or self.d != 3:
            RuntimeError("Generate circle works only for d=2 for the moment.")

        d = self.d
        bounds = self.boundary_dict["bounds"]
        if d == 2:
            x_min, x_max, y_min, y_max = (
                bounds[0][0],
                bounds[0][1],
                bounds[1][0],
                bounds[1][1],
            )
            if (x_max - x_min) != y_max - y_min:
                raise RuntimeError(
                    "Boundary generator assume that x_max-x_min = y_max-y_min"
                )
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = (
                bounds[0][0],
                bounds[0][1],
                bounds[1][0],
                bounds[1][1],
                bounds[2][0],
                bounds[2][1],
            )
            if (x_max - x_min) != y_max - y_min or (
                x_max - x_min
            ) != z_max - z_min:
                raise RuntimeError(
                    "Boundary generator assume that x_max-x_min = y_max-y_min and x_max-x_min = z_max-z_min"
                )

        x, w = np.polynomial.legendre.leggauss(25)
        np_bounds = np.array(bounds)
        if d == 2:
            x_var = ((np_bounds[0, 1] - np_bounds[0, 0]) / 2.0) * x + (
                (np_bounds[0, 1] + np_bounds[0, 0]) / 2.0
            )
            x_var = np.expand_dims(x_var, -1)
            w = w * ((np_bounds[0, 1] - np_bounds[0, 0]) / 2.0)
        elif d == 3:
            size = x.shape[0]
            x1, x2 = np.expand_dims(x, axis=1), np.expand_dims(x, axis=0)
            x1, x2 = (
                np.repeat(x1, size, axis=1).reshape(-1),
                np.repeat(x2, size, axis=0).reshape(-1),
            )
            xx = np.stack((x1, x2), axis=1)
            x_var = ((np_bounds[0, 1] - np_bounds[0, 0]) / 2.0) * xx + (
                (np_bounds[0, 1] + np_bounds[0, 0]) / 2.0
            )
            w = w * ((np_bounds[0, 1] - np_bounds[0, 0]) / 2.0)
            w = np.outer(w, w).reshape(-1)
        else:
            RuntimeError("Dimension not implemented.")
        x_var = torch.tensor(
            x_var, device=self.device, dtype=torch_float_precision
        )
        ones = torch.ones(
            (x_var.shape[0], 1),
            device=self.device,
            dtype=torch_float_precision,
        )

        boundary_val = None
        n = None
        for i, b in enumerate(bounds):
            min_val = torch.cat(
                (x_var[..., :i], b[0] * ones, x_var[..., i:]), dim=-1
            )
            max_val = torch.cat(
                (x_var[..., :i], b[1] * ones, x_var[..., i:]), dim=-1
            )

            ni = torch.zeros((x_var.shape[0], d))
            ni[:, i] = 1
            if i == 0:
                boundary_val = torch.cat((min_val, max_val), dim=0)
                n = torch.cat((-ni, ni), dim=0)
            else:
                boundary_val = torch.cat(
                    (boundary_val, min_val, max_val), dim=0
                )
                n = torch.cat((n, -ni, ni), dim=0)
        w = torch.tensor(w, device=self.device, dtype=torch_float_precision)
        # w = w.unsqueeze(-1).repeat(1,1,2*self.d)
        w = w.repeat(2 * self.d)
        return boundary_val, n, w
