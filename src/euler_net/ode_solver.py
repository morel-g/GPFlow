import torch
from case import Case
from gp_flows.ode_order_method import OdeOrderMethod

# import pydevd


class ODESolver:
    def __init__(self, v, ode_params, differential_op=None, deq_model=False):
        self.v = v
        self.ode_params = ode_params
        self.differential_op = differential_op
        self.ode_order_method = OdeOrderMethod(
            v,
            ode_params["method"],
            differential_op=differential_op,
            theta=ode_params["theta"],
            solver=ode_params["solver"],
            tol=ode_params["tol"],
            nb_it_max=ode_params["nb_it_max"],
        )
        self.deq_model = deq_model

    def solve(self, z0, t0, dt, nb_steps, solve_euler=False):
        if not self.deq_model:
            z = ode_solve(
                z0,
                t0,
                dt,
                nb_steps,
                self.ode_order_method.compute_step,
                solve_euler=solve_euler,
            )
        else:
            if self.ode_params["method"] != Case.euler:
                raise RuntimeError(
                    "DEQ model only implemented for euler method."
                )
            params = flatten(self.v.params.parameters(), self.v)
            z = ODEDEQ.apply(
                z0,
                t0,
                dt,
                nb_steps,
                self.ode_order_method,
                solve_euler,
                params,
            )

        return z


def ode_solve(z0, t0, dt, nb_steps, f, *args, **kwargs):
    t = t0
    zi = z0

    z = torch.zeros((nb_steps + 1,) + z0.shape).type_as(z0)
    z[0] = z0.clone()

    for i in range(nb_steps):
        zi = f(zi, t, dt, *args, **kwargs)
        z[i + 1] = zi.clone()
        t = t + dt
    return z


class ODEDEQ(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, z0, t0, dt, nb_steps, ode_order_method, solve_euler, params
    ):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        ctx.nb_steps = nb_steps
        ctx.v = ode_order_method.v
        ctx.solver = ode_order_method.solver
        ctx.theta = ode_order_method.theta
        ctx.ode_order_method = ode_order_method
        ctx.tol, ctx.nb_it_max = (
            ode_order_method.tol,
            ode_order_method.nb_it_max,
        )
        ctx.t_list = t0, dt, nb_steps

        with torch.no_grad():
            z = ode_solve(
                z0.clone().detach(),
                t0,
                dt,
                nb_steps,
                ode_order_method.compute_step,
                solve_euler=solve_euler,
            )

        ctx.save_for_backward(z.clone(), params)
        return z

    @staticmethod
    def backward(ctx, dy):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        v = ctx.v
        nb_steps = ctx.nb_steps
        theta = ctx.theta
        t0, dt, nb_steps = ctx.t_list
        z, params = ctx.saved_tensors
        ode_order_method = ctx.ode_order_method

        def compute_current_g(it, g, g_prev, vn, zn, retain_graph=True):
            if it == nb_steps:
                return (
                    theta
                    * dt
                    * torch.autograd.grad(
                        vn, zn, g, retain_graph=retain_graph
                    )[0]
                    + dy[-1]
                )
            else:
                return (
                    dt
                    * torch.autograd.grad(
                        vn,
                        zn,
                        theta * g + (1 - theta) * g_prev,
                        retain_graph=retain_graph,
                    )[0]
                    + g_prev
                )

        gi = torch.zeros_like(dy[-1])
        # g = torch.zeros((nb_steps,) + gi.shape).type_as(z)

        grad = torch.zeros_like(params)

        t = torch.tensor([dt * nb_steps]).type_as(z)
        tol, nb_it_max = ctx.tol, ctx.nb_it_max

        if ode_order_method.differential_op is None:
            stop = 0
        else:
            stop = -1

        for i in range(nb_steps, stop, -1):
            g_prev = gi.clone()
            zn, tn = z[i].clone(), t
            zn2, tn2 = z[i - 1].clone(), t - dt
            zn.requires_grad = True
            with torch.enable_grad():
                vn = v(zn, tn)
                vn2 = v(zn2, tn2)

            def f(g):
                return compute_current_g(i, g, g_prev, vn, zn)

            gi = ctx.solver(
                f, x0=torch.zeros_like(gi), tol=tol, nb_it_max=nb_it_max
            )

            if ode_order_method.differential_op is None:
                grad = grad + (
                    theta
                    * dt
                    * flatten(
                        torch.autograd.grad(
                            vn, v.params.parameters(), gi, allow_unused=True
                        ),
                        v,
                    )
                    + (1 - theta)
                    * dt
                    * flatten(
                        torch.autograd.grad(
                            vn2, v.params.parameters(), gi, allow_unused=True
                        ),
                        v,
                    )
                )
            else:
                v_basis = ode_order_method.v.p.eval_basis(zn)
                if i != 0:
                    y = dt * (theta * gi + (1 - theta) * g_prev)
                else:
                    y = dt * (1 - theta) * g_prev

                f1 = (v_basis * y.unsqueeze(-1)).sum(0).sum(0) + grad
                if i != 0:
                    grad = ode_order_method.differential_op.euler_backward(
                        f1, t, dt
                    )
                else:
                    grad = f1
            t = t - dt
        return None, None, None, None, None, None, grad


def flatten(x, v):
    flat = []
    for i, xi in enumerate(x):
        if xi is not None:
            flat.append(xi.flatten())
        else:
            flat.append(torch.zeros_like(v.params[i]))
    return torch.cat(flat)
