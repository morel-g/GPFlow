import torch
from ..case import Case
from .solvers import anderson, fix_point


class OdeOrderMethod:
    def __init__(
        self,
        v,
        ode_method,
        theta=0.0,
        solver=Case.fix_point,
        tol=1e-8,
        nb_it_max=1e2,
    ):
        self.v = v
        self.ode_method = ode_method
        self.theta = theta
        self.tol = tol
        self.nb_it_max = nb_it_max
        self.gaussian = None
        self.shape = (0, 0)
        # self.clip = clip
        if solver == Case.fix_point:
            self.solver = fix_point
        elif solver == Case.anderson:
            self.solver = anderson
        else:
            raise RuntimeError("Unknown solver for the ode:", solver)

    def compute_step(self, x, t, dt):
        if t == 0:
            self.v.initialize(t) if callable(
                getattr(self.v, "initialize", None)
            ) else None
        # v = lambda x, t: self.v(x, t)
        if self.ode_method == Case.euler:
            theta = self.theta
            if abs(theta) < 1e-5:
                # Explicit case
                x = x + dt * self.v(x, t)
            else:
                xn = x.clone()

                vn = self.v(xn, t)
                x = xn + dt * vn

                def f(x):
                    return xn + dt * (
                        (1 - theta) * vn + theta * self.v(x, t + dt)
                    )

                x = self.solver(f, x, nb_it_max=self.nb_it_max, tol=self.tol)
                err = torch.norm(x - f(x))

                if err > self.tol:
                    print("-" * 20)
                    print(
                        "WARNING: implicit scheme does not converge. Error = ",
                        err,
                    )
                    print("-" * 20)
                    err_points = torch.norm(x - f(x), dim=-1)
                    x_conv = ((err_points <= self.tol) * 1).unsqueeze(-1)
                    x = x_conv * x + (1 - x_conv) * (xn + dt * vn)
        elif self.ode_method == Case.RK4:
            k1 = self.v(x, t)
            k2 = self.v(x + (dt / 2.0) * k1, t + dt / 2.0)
            k3 = self.v(x + (dt / 2.0) * k2, t + dt / 2.0)
            k4 = self.v(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        else:
            RuntimeError("Ode method ", self.ode_method, " not implemented.")

        # if self.clip is not None:
        #    x = torch.clip(x, self.clip[0], self.clip[1])

        return x
