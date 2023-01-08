import torch
from ..case import Case
from .poly_tools.polynomials import Polynomials
from .poly_tools.gradient_manager import GradientManager
from .poly_tools.differential_op import DifferentialOp
import numpy as np


class PolyVelocity(torch.nn.Module):
    def __init__(self, nb_layers, velocity_dict, dt):
        super(PolyVelocity, self).__init__()
        self.nb_layers = nb_layers
        self.velocity_dict = velocity_dict

        self.p = Polynomials(
            2,
            self.velocity_dict["order_poly"],
            self.velocity_dict["poly_case"],
            apply_bc=True,
            boundary_dict=self.velocity_dict["boundary_dict"],
        )
        self.grad_manager = GradientManager(self.p)
        self.dt = dt
        self.nb_layer = nb_layers
        self.init_basis_gpu = False
        self.len_coef = 0
        self.ode_method = Case.RK4
        self.clip = not True

        self.init_alpha()
        self.params = self.alpha
        self.differential_op = DifferentialOp(
            self,
            order=self.velocity_dict["order_scheme"],
            coef_scheme=self.velocity_dict["coef_scheme"],
        )

    def reinit_cuda(self):
        self.p.reinit_cuda()
        self.grad_manager.reinit_cuda()
        self.init_basis_gpu = False

    def init_alpha(self):
        velocity_dict = self.velocity_dict

        self.training_id = 0

        if self.ode_method != Case.RK4:
            self.len_coef = self.nb_layers + 1
        else:
            self.len_coef = 2 * self.nb_layers + 1
        # if self.opt_init_only:
        if self.ode_method != Case.RK4:
            self.alpha_fix = [
                torch.zeros(self.p.nb_basis) for i in range(self.nb_layers + 1)
            ]
        else:
            self.alpha_fix = [
                torch.zeros(self.p.nb_basis)
                for i in range(2 * self.nb_layers + 1)
            ]
        if (
            "init" in velocity_dict
            and velocity_dict["init"] == Case.random_init
        ):
            self.alpha_fix[0] = 0.5 * torch.rand(self.p.nb_basis) - 1.0
        else:
            self.alpha_fix[0] = (
                velocity_dict["alpha init"]
                if "alpha init" in velocity_dict
                else torch.zeros(self.p.nb_basis)
            )
        if self.ode_method != Case.RK4:
            self.alpha = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(torch.zeros(self.p.nb_basis))]
            )
        else:
            self.alpha = torch.nn.ParameterList(
                [torch.nn.parameter.Parameter(torch.zeros(self.p.nb_basis))]
            )
        [a.data.uniform_(-1e-2, 1e-2) for a in self.alpha]

        # [a.data.uniform_(-1e-2, 1e-2) for a in self.alpha]

    def initialize(self, t):
        self.euler_time_resolution = t.clone()

    def init_discrete_alpha(self, i):
        # Initialize discete alpha_i from the previous time step
        if i <= 0:
            raise RuntimeError(
                "Cannot initialize discrete alpha for index ", i
            )
        else:
            if i == 1:
                with torch.no_grad():
                    self.alpha[i].data = self.alpha_fix[0].clone()
                    self.id_training = i
            else:
                with torch.no_grad():
                    self.alpha[i].data = self.alpha[i - 1].data.clone()
                    self.id_training = i
                    self.alpha_fix[i - 1] = self.alpha[i - 1].data.clone()

            self.init_basis_gpu = False

    def set_poly(self, p, bound=0.05):
        self.p = p
        self.init_alpha(bound)

    def discrete_id_from_t(self, time):
        t = time.item()
        dt = self.dt
        if (
            self.ode_method != Case.RK4
            and self.ode_method != Case.middle_point
        ):
            i = round(t / dt)
        elif self.ode_method == Case.middle_point:
            eps = 1e-5
            i = round((t + (dt / 2.0) - eps) / dt)
        else:
            i = round(2.0 * t / dt)
        return i

    def set_alpha(self, alpha, t):
        i = self.discrete_id_from_t(t)
        self.alpha_fix[i] = alpha

    def init_cuda_alpha_fix(self, t):
        for j in range(self.len_coef):
            # if self.opt_init_only:
            self.alpha_fix[j] = self.alpha_fix[j].type_as(t)
            if j < len(self.alpha):
                self.alpha[j].data = self.alpha[j].data.type_as(t)
        # self.alpha_init = self.alpha_init.type_as(t)
        self.init_basis_gpu = True

    def evaluate_alpha_list(self, t, nb_el):
        if nb_el == 0:
            return self.evaluate_alpha(t).unsqueeze(0)
        else:
            if not self.init_basis_gpu:
                self.init_cuda_alpha_fix(t)
            i = self.discrete_id_from_t(t)
            if i - nb_el < 0:
                raise RuntimeError(
                    "Value of i-nb_el should be greater or equal to 0."
                )
            if i - nb_el == 0 and not (self.velocity_dict["euler_only"]):
                alpha_list = self.evaluate_alpha(torch.tensor([0.0]))
                nb_el = nb_el - 1
                alpha_list = torch.cat(
                    (
                        alpha_list.unsqueeze(dim=0),
                        torch.stack(self.alpha_fix[i - nb_el : i + 1]),
                    ),
                    dim=0,
                )
            else:
                alpha_list = torch.stack(self.alpha_fix[i - nb_el : i + 1])

            return alpha_list

    def evaluate_alpha(self, t):
        if not self.init_basis_gpu:
            self.init_cuda_alpha_fix(t)

        i = self.discrete_id_from_t(t)

        if self.velocity_dict["euler_only"]:
            a = self.alpha_fix[i]
        else:
            if self.velocity_dict["euler_only"] and i == 0:
                a = self.alpha_fix[i]
            elif i != 0:
                a = self.alpha_fix[i]
            else:
                a = self.alpha[i]

        return a

    def grad(self, x, t):
        alpha = self.evaluate_alpha(t)
        return self.grad_manager.compute_grad(x, alpha)

    def deriv_time(self, x, t):
        a = self.evaluate_alpha_prime(t)
        return self.p.eval(x, a)

    def __call__(self, x, t=0):
        # if self.training:
        if self.euler_time_resolution < t - 1e-5:
            if self.ode_method != Case.RK4:
                dt_step = self.dt
            else:
                dt_step = self.dt / 2.0
            if self.euler_time_resolution + dt_step - t < -1e-5:
                raise RuntimeError("Try to solve euler with a time too high.")
            self.differential_op.solve_euler(t - dt_step, dt_step)
            self.euler_time_resolution += dt_step

        a = self.evaluate_alpha(t)
        v = self.p.eval(x, a)

        if self.clip:
            v = clip_velocity(self.velocity_dict["boundary_dict"], x, v)

        return v


def clip_velocity(boundary_dict, x, v):
    if boundary_dict["case"] == Case.circle:
        r = boundary_dict["radius"]
        scale = (torch.norm(x, dim=1) <= r) * 1.0
        v = scale.unsqueeze(1) * v
    elif boundary_dict["case"] == Case.rectangle:
        x_max = abs(np.array(boundary_dict["bounds"])).max() + 1e-3
        scale = (abs(x).max(dim=1)[0] < x_max) * 1.0
        v = scale.unsqueeze(1) * v
    else:
        RuntimeError(
            "Can't clip the velocity for the domain ", boundary_dict["case"]
        )
    return v
