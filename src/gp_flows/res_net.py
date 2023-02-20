import torch
from .velocity_field import VelocityField
from .ode_solver import ODESolver
from ..case import Case
from ..precision import torch_float_precision, eps_precision
from ..euler_net.poly_velocity import PolyVelocity

# from functorch import jacrev, grad, vmap
import numpy as np
from .compose import Compose
from functorch import vjp

def get_gp_flow(
    dim,
    velocity_dict,
    nb_layers_gp,
    nb_blocks=1,
    T_final=1.0,
    ode_params={"method": Case.RK4},
    euler_dict={"use_euler": False},
    euler_spectral_velocity_dict=None,
):
    """Construct a GP flow.

    Args:
        dim: The dimension.
        velocity_dict: The velocity dictionary associated with
        the flow.
        nb_layers_gp: The number of layers of the GP flow.
        nb_blocks: The number of blocks of the GP flow. Default to 1.
        T_final: The final time of the simulation. Default to 1.0.
        ode_params: The ode parameters. Default to {"method": Case.RK4}.
        euler_dict: The euler dictionary. Default to {"use_euler": False}.
        euler_spectral_velocity_dict: Dictionary neede if solving Euler with a
        spectral method. Default to None.

    Returns:
        The corresponding GP flow.
    """
    if nb_blocks != 1:
        flow = Compose(
            [
                ResNet(
                    dim,
                    velocity_dict,
                    nb_layers_gp,
                    T_final,
                    ode_params,
                    euler_dict,
                    euler_spectral_velocity_dict,
                )
                for i in range(nb_blocks)
            ]
        )
    else:
        flow = ResNet(
            dim,
            velocity_dict,
            nb_layers_gp,
            T_final,
            ode_params,
            euler_dict,
            euler_spectral_velocity_dict,
        )

    return flow


class ResNet(torch.nn.Module):
    def __init__(
        self,
        dim,
        velocity_dict,
        nb_layers,
        T_final=1.0,
        ode_params={"method": Case.RK4},
        euler_dict={"use_euler": False},
        euler_spectral_velocity_dict=None,
        compute_divergence=False,
    ):
        """A ResNet to solve numerically an ODE.

        Args:
            dim: dimension of the points.
            velocity_dict: a dictionary containing useful informations about"
            the velocity.
            nb_layers: number of layers i.e. number of time steps to solve the
            ODE
            T_final: final time of the simulation. Defaults to 1.0.
            ode_params: parameters of the ODE solver. Defaults to
            {"method": Case.RK4}.
            euler_dict: a dictionary used when solving Euler. Defaults to
            {"use_euler": False}.
            euler_spectral_velocity_dict: dictionary used when solving Euler
            in dimension 2. Defaults to None.
            compute_divergence: to compute and store the divergence of the
            velocity field. Defaults to False.

        Raises:
            RuntimeError: when trying to solve Euler in dim > 2.
            RuntimeError: when trying to solve Euler with a spectral method
            but no spectral dict available.
        """
        super(ResNet, self).__init__()

        self.velocity_dict = velocity_dict
        self.nb_layers = nb_layers
        self.loop_coef = 1
        self.remove_epochs = 0
        self.T_final = T_final
        self.dt = T_final / self.nb_layers
        self.set_nb_layers_eval(nb_layers)
        self.constraint_value = 0.0
        self.init_trajectories()
        self.euler_dict = euler_dict
        if (
            self.euler_dict["use_euler"]
            and self.euler_dict["case"] == Case.penalization
        ):
            self.euler_coef_penalization = self.euler_dict["coef_penalization"]
        else:
            self.euler_coef_penalization = None

        if (
            self.euler_dict["use_euler"]
            and self.euler_dict["case"] == Case.spectral_method
        ):
            if dim != 2:
                raise RuntimeError("Euler solve only implemented for d=2.")
            if euler_spectral_velocity_dict is None:
                raise RuntimeError(
                    "Try to solve Euler with a spectral method \
                    but no spectral_velocity_dict provided."
                )
            # Take nb_layers_gp as the number of layers for the spectral method
            self.v = PolyVelocity(
                nb_layers=nb_layers,
                velocity_dict=euler_spectral_velocity_dict,
                dt=self.dt,
            )
        else:

            self.v = VelocityField(
                dim,
                velocity_dict.get("nb_neurons", [10]),
                activ_func=velocity_dict.get("activ_func", Case.tanh),
                time_state=velocity_dict.get(
                    "time_state", Case.continuous_time
                ),
                incompressible=velocity_dict.get("incompressible", True),
                nb_div_blocks=velocity_dict.get("nb_div_blocks", 1),
            )

        if torch_float_precision == torch.double:
            self.v.double()
        self.ode_solver = None

        self.ode_solver = ODESolver(
            self.v,
            ode_params.get("method", Case.RK4),
            ode_params.get("theta", 0.0),
            ode_params.get("solver", Case.fix_point),
            ode_params.get("tol", 1e-8),
            ode_params.get("nb_it_max", 1e2),
            ode_params.get("solve_method", Case.standard),
        )

        self.compute_divergence = compute_divergence

    def init_trajectories(self):
        self.x_trajectories = []
        self.v_trajectories = []

    def set_nb_layers_eval(self, nb_layers):
        self.nb_layers_eval = nb_layers
        self.dt_eval = self.T_final / self.nb_layers_eval

    def forward(
        self,
        x,
        training=False,
        current_epoch=0,
        reverse=False,
        save_trajectories=False,
        id_final_layer=-1,
    ):
        self.init_trajectories()
        x = self.solve(
            x,
            current_epoch=current_epoch,
            reverse=reverse,
            training=training,
            save_trajectories=save_trajectories,
            id_final_layer=id_final_layer,
        )  # and not training)))

        return x

    def solve(
        self,
        x,
        current_epoch=0,
        reverse=False,
        training=True,
        save_trajectories=False,
        id_final_layer=-1,
    ):
        t = torch.tensor(0).type_as(x)
        if training:
            dt_solve = self.dt
            nb_layers = self.nb_layers
        else:
            dt_solve = self.dt_eval
            nb_layers = self.nb_layers_eval

        dt = torch.tensor(dt_solve).type_as(x)
        if reverse:
            t = torch.tensor(dt_solve * self.nb_layers).type_as(x)
            dt = -dt

        x_full = self.ode_solver.solve(x, t, dt, nb_layers)
        self.process_ode_solution(
            x_full,
            t,
            dt,
            nb_layers,
            current_epoch=current_epoch,
            reverse=reverse,
            save_trajectories=save_trajectories,
        )
        x = x_full[id_final_layer]

        return x

    def process_ode_solution(
        self,
        x_full,
        t,
        dt,
        nb_layers,
        current_epoch=0,
        reverse=False,
        save_trajectories=False,
    ):
        x = x_full[0]

        self.length_trajectory = torch.zeros(x.shape)
        if save_trajectories:
            self.x_trajectories = []
            self.v_trajectories = []
            x_start = torch.clone(x)
            self.x_trajectories.append(x)
            if not reverse:
                self.v_trajectories.append(self.v(x_start, t))
            else:
                self.v_trajectories.append(-self.v(x_start, t))

        self.init_constraint(x)
        for i in range(nb_layers):
            x = x_full[i + 1]
            self.compute_constraint(x, t, current_epoch)
            t = t + dt
            if save_trajectories:
                if (
                    i % max(self.nb_layers // 40, 1) == 0
                    or i == self.nb_layers - 1
                ):
                    self.x_trajectories.append(x)
                    if not reverse:
                        self.v_trajectories.append(self.v(x_start, t))
                    else:
                        self.v_trajectories.append(-self.v(x_start, t))

    def get_trajectories(self):
        x_trajectories = [
            self.x_trajectories[i].detach().cpu().numpy()
            for i in range(len(self.x_trajectories))
        ]
        v_trajectories = [
            self.v_trajectories[i].detach().cpu().numpy()
            for i in range(len(self.v_trajectories))
        ]
        return np.array(x_trajectories), np.array(v_trajectories)

    def init_constraint(self, x):
        self.regularization_value = torch.tensor(0.0).type_as(x)
        # self.euler_pen_value = torch.tensor(0.).type_as(x)
        self.lagrangian = torch.tensor(0.0).type_as(x)
        self.divergence = torch.zeros(x.shape[0]).type_as(x)

    def compute_constraint(self, x, t, epoch):
        self.lagrangian += (
            self.dt * self.compute_lagrangian(x, t).sum(-1).mean()
        )
        if (
            self.euler_dict["use_euler"]
            and self.euler_dict["case"] == Case.penalization
            and self.euler_coef_penalization > 2 * eps_precision
        ):
            self.regularization_value += (
                self.euler_coef_penalization * self.compute_euler_pen(x, t)
            )

        if self.compute_divergence:
            dt = self.dt
            """
            # Simpson method
            x1 = x + (dt / 2.0) * self.v(x, t)
            div1 = self.v.compute_divergence(x1, t + dt / 2.0)
            self.divergence += (2.0 * dt / 3.0) * div1

            if (t - 0.0) < eps_precision:
                self.divergence += (dt / 6.0) * self.v.compute_divergence(x, t)
            elif (self.T_final - t - self.dt) < eps_precision:
                x2 = x1 + (dt / 2.0) * self.v(x1, t + (dt / 2.0))
                self.divergence += (dt / 6.0) * self.v.compute_divergence(
                    x2, t + dt
                )
            else:
                self.divergence += (
                    (1.0 / 3.0) * dt * self.v.compute_divergence(x, t)
                )

            # self.divergence += self.dt * self.v.compute_divergence(x, t)
            """
            k1 = self.v(x, t)
            k2 = self.v(x + (dt / 2.0) * k1, t + dt / 2.0)
            k3 = self.v(x + (dt / 2.0) * k2, t + dt / 2.0)
            # k4 = self.v(x + dt * k3, t + dt)
            div1 = self.v.compute_divergence(x, t)
            div2 = self.v.compute_divergence(x + (dt / 2.0) * k1, t + dt / 2.0)
            div3 = self.v.compute_divergence(x + (dt / 2.0) * k2, t + dt / 2.0)
            div4 = self.v.compute_divergence(x + dt * k3, t + dt)
            self.divergence += (dt / 6.0) * (
                div1 + 2.0 * div2 + 2.0 * div3 + div4
            )

    def compute_euler_pen(self, y, t):
        """Compute Euler penalization

        Args:
            y: The points where the penalization is computed.
            t: The time variable.

        Returns:
            Value of Euler penalization.
        """
        nb_batch = 1
        z, w = (
            torch.randn(nb_batch, y.shape[1]).type_as(y),
            torch.randn(nb_batch, y.shape[1]).type_as(y),
        )
        batch_size = (
            self.euler_dict["batch_size"]
            if self.euler_dict.get("batch_size", None) is not None
            else y.shape[0]
        )
        z, w = (
            z.repeat_interleave(batch_size, dim=0),
            w.repeat_interleave(batch_size, dim=0),
        )
        x = y.repeat(nb_batch, 1)

        x.requires_grad_(True)
        dt = 2 * np.sqrt(eps_precision)
        # approx_lagrangian_deriv = True
        scheme = Case.order_1

        # if not approx_lagrangian_deriv:
        #     with torch.enable_grad():
        #         tot_deriv = self.v.lagrangian_deriv(x, t)

        #     z_w = (
        #         torch.autograd.grad(tot_deriv, x, z, create_graph=True)[0] * w
        #     ).sum(-1)
        #     w_z = (
        #         torch.autograd.grad(tot_deriv, x, w, create_graph=True)[0] * z
        #     ).sum(-1)
        # else:
        
        # with torch.enable_grad():
        #     lagrange_deriv = self.v.approx_lagrangian_deriv(
        #         x, t, dt=dt, scheme=scheme
        #     )
        _, vjp_fun = vjp(lambda xi: self.v.approx_lagrangian_deriv(
                xi, t, dt=dt, scheme=scheme
            ), x)   
        z_w = (vjp_fun(z)[0] * w).sum(-1)
        w_z = (vjp_fun(z)[0] * w).sum(-1)
        # z_w = (
        #     torch.autograd.grad(lagrange_deriv, x, z, create_graph=True)[0] * w
        # ).sum(-1)
        # w_z = (
        #     torch.autograd.grad(lagrange_deriv, x, w, create_graph=True)[0] * z
        # ).sum(-1)
        return ((z_w - w_z) ** 2).mean()

    def compute_lagrangian(self, x, t):
        return 0.5 * self.v(x, t) ** 2

    def regularization_coef(self):
        return self.regularization_value
