import torch
from case import Case
import copy
from gp_flows.solvers import anderson, fix_point, gradient_descent
import numpy as np


class OdeOrderMethod:
    def __init__(
        self,
        v,
        ode_method,
        differential_op=None,
        theta=0.0,
        solver=Case.fix_point,
        tol=1e-8,
        nb_it_max=1e2,
    ):
        self.v = v
        self.ode_method = ode_method
        self.differential_op = differential_op
        self.theta = theta
        self.tol = tol
        self.nb_it_max = nb_it_max
        self.brownian_motion = False  # True
        self.gaussian = None
        self.shape = (0, 0)
        if solver == Case.fix_point:
            self.solver = fix_point
        elif solver == Case.anderson:
            self.solver = anderson
        else:
            raise RuntimeError("Unknown solver for the ode")

    def motion(self, x, dt):
        if self.gaussian == None or self.shape != x.shape:
            self.gaussian = (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros_like(x).type_as(x),
                    abs(dt.detach().cpu().numpy()[0])
                    * torch.eye(x.shape[-1]).type_as(x),
                )
            )
            self.shape = x.shape
        return torch.sqrt(torch.tensor(2.0)) * self.gaussian.sample()

    def compute_step(self, x, t, dt, solve_euler):
        if self.brownian_motion:
            with torch.no_grad():
                b_motion = self.motion(x.detach(), dt)
        time_step = t[0]
        v = lambda x, t: self.v(x, t, time_step=time_step)
        if self.ode_method == Case.euler:
            if solve_euler:
                self.differential_op.solve_euler(t, dt)

            theta = self.theta
            if abs(theta) < 1e-5:
                # Explicit case
                x = x + dt * self.v(x, t)
            else:
                xn = x.clone()

                vn = v(xn, t)
                # x_prev = xn
                x = xn + dt * vn

                f = lambda x: xn + dt * (
                    (1 - theta) * vn + theta * v(x, t + dt)
                )
                x = self.solver(
                    f, x, nb_it_max=self.nb_it_max, tol=self.tol
                )  # anderson(f, x, nb_it_max=nb_it_max, tol=tol)#
                err = torch.norm(x - f(x))

                if err > self.tol:
                    print("-" * 20)
                    print(
                        "ERROR: implicit scheme does not converge error = ",
                        err,
                    )
                    print("-" * 20)
                    err_points = torch.norm(x - f(x), dim=-1)
                    x_conv = ((err_points <= self.tol) * 1).unsqueeze(-1)
                    x = x_conv * x + (1 - x_conv) * (xn + dt * vn)
        elif self.ode_method == Case.RK4:
            if solve_euler:
                self.differential_op.solve_euler(t, dt / 2.0)
                self.differential_op.solve_euler(t + dt / 2.0, dt / 2.0)
            x0 = x.clone()
            K = dt * v(x0, t)
            x = x + (1.0 / 6.0) * K

            K = dt * v(x0 + 0.5 * K, t + (dt / 2.0))
            x += (2.0 / 6.0) * K

            K = dt * v(x0 + 0.5 * K, t + (dt / 2.0))
            x += (2.0 / 6.0) * K

            K = dt * v(x0 + K, t + dt)
            x += (1.0 / 6.0) * K

            return x
            """
            k1 = self.v(x, t)
            k2 = self.v(x + (dt/2.)*k1, t+dt/2.)
            k3 = self.v(x + (dt/2.)*k2, t+dt/2.)
            k4 = self.v(x + dt*k3, t+dt)
            x = x + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
            """
        else:
            RuntimeError("Ode method ", self.ode_method, " not implemented.")
        if self.brownian_motion:
            x = x + 0.1 * b_motion
        return x
