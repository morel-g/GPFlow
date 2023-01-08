import torch

from ..case import Case
from ..precision import eps_precision
from ..tool_box import time_dependent_var
from .vector_field import VectorField


def construct_eye_pad(dim, nb_blocks, device):
    # To fill the n first rows and columns of a matrix with zeros
    eye_pad = torch.eye(dim, device=device)
    eye_pad = eye_pad.reshape((1, dim, dim))
    eye_pad = eye_pad.repeat(nb_blocks, 1, 1)
    for i in range(nb_blocks - 1):
        eye_pad[i + 1 :, i, i] = 0.0
    eye_pad = eye_pad.unsqueeze(0)
    return eye_pad


def divergence_free_vector_func(v, x, bound, eye_pad=None, nb_blocks=1):
    """
    Compute the divergence free vectors associated with a vector v.
    The function v(x) is supposed to have size dim * nb_blocks.

    Args:
        v: The initial vector field from which the gradient is taken to
        construct the non divergence function. The vector v(x) should have
        size dim * nb_blocks.
        x: The position to evaluate the divergence free vector.
        bound: Boundary of the domain to apply the boundary conditions
        $u \cdot n =0$.
        nb_blocks: Number of divergence block. Defaults to 1.

    Returns:
        A divergence free vector evaluated in x which satisfy the boundary
        conditions $u \cdot n =0$.
    """

    b = bound
    device = x.device
    # eye_pad.to(device)
    x_space = x[:, :-1]
    dim = x_space.shape[-1]
    if eye_pad is None:
        eye_pad = construct_eye_pad(dim, nb_blocks, device)

    grad_f = v.jacobian(x).squeeze(1)
    grad_f = grad_f[:, :, :-1]
    f = v.eval(x)
    f = f.reshape((f.shape[0],) + (dim, nb_blocks))

    # To fill the n first components of a vector with zeros
    vect_pad = torch.ones(dim, nb_blocks, device=device)
    vect_pad = torch.tril(vect_pad).unsqueeze(0)

    f_pad = f * vect_pad
    M = f_pad.unsqueeze(-2) - f_pad.unsqueeze(-3)
    M = torch.transpose(M.unsqueeze(1), -1, -4).squeeze(-1)
    M = torch.matmul(torch.matmul(eye_pad, M), eye_pad).sum(1)

    grad_f = torch.transpose(
        grad_f.reshape((grad_f.shape[0], dim, nb_blocks, dim)), -2, -3
    )
    # Fill with zeros rows and columns of the grad of the functions v^n
    grad_blocks = torch.matmul(torch.matmul(eye_pad, grad_f), eye_pad)
    diag_blocks = torch.diagonal(grad_blocks, dim1=-2, dim2=-1)

    mat_1 = 2 * torch.matmul(M, x_space.unsqueeze(-1)).squeeze(-1)
    mat_2 = torch.matmul(
        grad_blocks.sum(-3), (x_space**2 - b).unsqueeze(-1)
    ).squeeze(-1)
    mat_3 = (
        ((x_space**2 - b).unsqueeze(1) * diag_blocks)
        .sum(-1)
        .unsqueeze(-1)
        .repeat(1, 1, dim)
    )
    mat_3 = (vect_pad * torch.transpose(mat_3, -1, -2)).sum(-1)
    return (x_space**2 - b) * (mat_1 + mat_2 - mat_3)


class VelocityField(torch.nn.Module):
    def __init__(
        self,
        dim,
        nb_neurons,
        activ_func=Case.tanh,
        time_state=Case.continuous_time,
        incompressible=False,
        nb_div_blocks=1,
    ):
        super(VelocityField, self).__init__()

        self.time_state = time_state
        self.incompressible = incompressible
        self.bound = 1 - eps_precision

        self.dim = dim
        self.dim_in = dim
        self.dim_out = dim

        self.nb_div_blocks = nb_div_blocks
        self.eye_pad = None

        if Case.continuous_time == self.time_state:
            # Add one entry for the time variable.
            self.v = VectorField(
                self.dim_in + 1,
                self.dim_out * self.nb_div_blocks,
                nb_neurons,
                func_name=activ_func,
            )
        else:
            self.v = VectorField(
                self.dim_in, self.dim_out, nb_neurons, func_name=activ_func
            )

    def __call__(self, x, t=0):
        x = time_dependent_var(x, t, self.time_state)

        if not self.incompressible:
            v = self.v(x)
        else:
            v = self.divergence_free_velocity(x)

        return v

    def divergence_free_velocity(self, x):
        device = x.device
        self.eye_pad = (
            construct_eye_pad(self.dim, self.nb_div_blocks, device)
            if self.eye_pad is None
            else self.eye_pad.to(device)
        )
        v_x = divergence_free_vector_func(
            self.v,
            x,
            self.bound,
            self.eye_pad,
            nb_blocks=self.nb_div_blocks,
        )
        return v_x

    def lagrangian_deriv(self, x, t):
        """Compute the lagrangian derivative $\partial_t v + v \cdot \nabla v$ with torch.autograd function.

        Args:
            x: the space variable.
            t: the time varible.

        Returns:
            Lagrangian derivative evaluate at the point (x,t)
        """
        with torch.enable_grad():
            v_x = self(x, t)
        grad_v_v = torch.autograd.functional.jvp(
            lambda x: self(x, t), x, v_x, create_graph=True
        )[1]
        dt_v = torch.autograd.functional.jvp(
            lambda t: self(x, t), t, torch.tensor(1.0), create_graph=True
        )[1]
        return dt_v + grad_v_v

    def approx_lagrangian_deriv(self, x, t, dt, scheme=Case.order_1):
        """Numerical approximation of the lagrangian derivative.

        Args:
            x: the space variable.
            t: the time varible.
            dt: time step used in the discretization.
            scheme: Discretization scheme ot use. Choices are Case.order_1,
                Case.order_2, Case.RK4_space. Defaults to Case.order_1.

        Raises:
            RuntimeError: If the scheme is not known.

        Returns:
            An approximation of the lagrangian derivative at the point (x,t).
        """
        if scheme == Case.order_1:
            # Order 1
            v_x = self(x, t)
            x2 = x + dt * v_x
            v_x2 = self(x2, t + dt)
            return (v_x2 - v_x) / dt
        elif scheme == Case.order_2:
            # Order 2
            k1 = self(x, t)
            k2 = self(x + 0.5 * dt * k1, t + 0.5 * dt)
            x2 = x + (dt / 2.0) * (k1 + k2)
            return 2 * (-0.5 * self(x2, t + dt) + 2 * k2 + -1.5 * k1) / (dt)
        elif scheme == Case.RK4_space:
            # RK4 for space and order 1 for the time
            k1 = self(x, t)
            k2 = self(x + (dt / 2.0) * k1, t + dt / 2.0)
            k3 = self(x + (dt / 2.0) * k2, t + dt / 2.0)
            k4 = self(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            return (self(x, t + dt) - k1) / dt
        else:
            raise RuntimeError("Scheme not implemented " + scheme)

    def compute_divergence(self, x, t):
        """Compute the divergence of the velocity field for debugginf prupose.

        Args:
            x: the space variable.
            t: the time varible.

        Returns:
            The divergence evaluate at the point (x,t)
        """
        x = time_dependent_var(x, t, self.time_state)
        jacobian = self.v.jacobian(x)
        div = torch.diagonal(jacobian, dim1=-2, dim2=-1).sum(-1)
        return div
