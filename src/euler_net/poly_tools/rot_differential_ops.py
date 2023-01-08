from os import error
import sys

from numpy.core.fromnumeric import cumsum
sys.path.append('../')
from poly_tools.polynomials import Polynomials
from poly_tools.gradient_manager import GradientManager
#from precision import torch_precision
from case import Case
import torch

class RotDifferentialOp():
    def __init__(self, v, f_init=None, dt=1., nu=0.):
        self.nu = nu
        self.v = v
        self.f_init = f_init
        self.grad = GradientManager(self.v.p)
        self.apply_bc = self.v.p.apply_bc
        if self.apply_bc:
            self.boundary_manager = self.v.p.boundary_manager
        self.dt = dt
        self.init_basis_gpu = False
        self.grad_to_rot = torch.tensor([1., -1.])
        self.init_rot_arrays()

    def reinit_cuda(self):
        self.grad.reinit_cuda()
        self.init_basis_gpu = False

    def init_cuda(self, x):
        self.deriv_grad_deg, self.deriv_grad_coef = (self.deriv_grad_deg.type_as(x),
                                                    self.deriv_grad_coef.type_as(x))
        self.grad_to_rot = self.grad_to_rot.type_as(x)
        if self.apply_bc:
            self.boundary_manager.boundary_matrix = self.boundary_manager.boundary_matrix.type_as(x)
        self.init_basis_gpu = True

    def rot_from_grad(grad_vn, grad_vn2, dt):
        Id = torch.eye(2, 2).reshape(1, 2, 2)
        Id = Id.repeat(grad_vn.shape[0], 1, 1).type_as(grad_vn)

        rot_matrix = torch.tensor([[0., -1.], [1., 0.]]).type_as(grad_vn)

        gradient = (1./dt) * (torch.matmul(grad_vn2, Id + dt*grad_vn)-grad_vn)
        
        return (rot_matrix * gradient).sum(dim=(1, 2))
    
    def compute_rot_Dt_v(self, x_n, x_n2, t):
        if not self.init_basis_gpu:
            self.init_cuda(x_n)
        # Compute the rotational of (D/Dt)v(t,x)
        dt = self.dt
        
        if t < 1e-5 and self.v.init_case == Case.exact_init:
            grad_vn = self.f_init.compute_grad(x_n, t)
        else:
            alpha_n = self.v.evaluate_alpha(t)
            if self.apply_bc:
                alpha_n = self.boundary_manager.apply_boundary_conditions(alpha_n)
            grad_vn = self.grad.compute_grad(x_n, alpha_n)
 
        alpha_n2 = self.v.evaluate_alpha(t+dt)
        if self.apply_bc:
            alpha_n2 = self.boundary_manager.apply_boundary_conditions(alpha_n2)        

        grad_vn2 = self.grad.compute_grad(x_n2, alpha_n2)
        if self.nu>1e-5:
            return DifferentialOp.rot_from_grad(grad_vn, grad_vn2, dt) - self.nu * self.grad.compute_rot_laplacian(x_n, alpha_n)
        else:
            return DifferentialOp.rot_from_grad(grad_vn, grad_vn2, dt)

    def init_rot_arrays(self):
        grad_deg = self.grad.grad_degrees
        grad_coef = self.grad.grad_coef
        dy_grad_deg_v1, dy_grad_coef_v1 = GradientManager.compute_deriv(
            grad_deg[:, 0, :, :], grad_coef[:, 0, :], torch.tensor([0, 1]))
        dx_grad_deg_v2, dx_grad_coef_v2 = GradientManager.compute_deriv(
            grad_deg[:, 1, :, :], grad_coef[:, 1, :], torch.tensor([1, 0]))

        self.deriv_grad_deg, self.deriv_grad_coef = (torch.stack((dy_grad_deg_v1, dx_grad_deg_v2), dim=0),
                torch.stack((dy_grad_coef_v1, dx_grad_coef_v2), dim=0))

    def rot_euler_eq_discrete_time(self, xn, xn2, t):

        if not self.init_basis_gpu:
            self.init_cuda(xn)
        
        alpha_n = self.v.evaluate_alpha(t)
 
        alpha_n2 = self.v.evaluate_alpha(t+self.dt)
        if self.apply_bc:
            alphas = torch.stack((alpha_n, alpha_n2), dim=1)
            alphas = self.boundary_manager.apply_boundary_conditions(alphas)
            alpha_n = alphas[:, 0]
            alpha_n2 = alphas[:, 1]

        grad_vn2 = self.grad.compute_grad(xn, alpha_n2)
        vn2 = self.v(xn, t+self.dt)
        deriv_grad_n2 = Polynomials.eval_poly_arrays(self.deriv_grad_deg, self.deriv_grad_coef, xn, alpha_n2)
        
        if self.v.discrete_id_from_t(t) == 0 and self.v.init_case == Case.exact_init:
            vn = self.f_init(xn, 0.)
            grad_vn = self.f_init.compute_grad(xn, 0.)
            deriv_grad_n = self.f_init.deriv_grad(xn, 0.)
        else:
            grad_vn = self.grad.compute_grad(xn, alpha_n)
            deriv_grad_n = Polynomials.eval_poly_arrays(self.deriv_grad_deg, self.deriv_grad_coef, xn, alpha_n)
            vn = self.v(xn, t)            

        dt_grad_v = (grad_vn2 - grad_vn) / self.dt
        if self.nu>1e-5:
            return (dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(vn2, grad_vn2, deriv_grad_n2) 
                    - self.nu* self.grad.compute_rot_laplacian(xn, alpha_n2))
        else:
            return dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(vn, grad_vn, deriv_grad_n) 
        #return dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(vn2, grad_vn2, deriv_grad_n2)

    def rot_euler_eq(self, x, t):
        if not self.init_basis_gpu:
            self.init_cuda(x)

        alpha = self.v.evaluate_alpha(t)
        alpha_prime = self.v.evaluate_alpha_prime(t)
        alphas = torch.stack((alpha, alpha_prime), dim=1)
        if self.apply_bc:
            alphas = self.boundary_manager.apply_boundary_conditions(alphas)
            alpha = alphas[:, 0]
        grads = self.grad.compute_grad(x, alphas)
        grad_v = grads[:, :, :, 0]
        dt_grad_v = grads[:, :, :, 1]
        deriv_grad = Polynomials.eval_poly_arrays(self.deriv_grad_deg, self.deriv_grad_coef, x, alpha)
        v = self.v(x, t)
        if self.f_init is not None:
            v += self.f_init(x, torch.tensor(0.))
            grad_v += self.f_init.compute_grad(x, torch.tensor(0.))
            deriv_grad += self.f_init.deriv_grad(x, torch.tensor(0.))
        if self.nu> 1e-5:
            #alpha2 = self.v.evaluate_alpha(t+ self.dt)
            #grad_v2 = self.grad.compute_grad(x, alpha2)
            #deriv_grad2 = Polynomials.eval_poly_arrays(self.deriv_grad_deg, self.deriv_grad_coef, x, alpha2)

            return (dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(v, grad_v, deriv_grad)
                    - self.nu* (self.grad.compute_rot_laplacian(x, alpha)+self.f_init.rot_laplacian(x, torch.tensor(0.))))
            #return (dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + (self.rot_v_grad_v(v, grad_v, deriv_grad)
            #        - self.nu* self.grad.compute_rot_laplacian(x, alpha) + self.rot_v_grad_v(v, grad_v2, deriv_grad2)
            #        - self.nu* self.grad.compute_rot_laplacian(x, alpha2))/2.)

            #return (dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(v, grad_v2, deriv_grad2)
            #        - self.nu* self.grad.compute_rot_laplacian(x, alpha2))
        else:
            return dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(v, grad_v, deriv_grad)
    """
    def rot_euler_eq(self, x, t):
        if not self.init_basis_gpu:
            self.init_cuda(x)

        alpha = self.v.evaluate_alpha(t)
        alpha_prime = self.v.evaluate_alpha_prime(t)
        alphas = torch.stack((alpha, alpha_prime), dim=1)
        if self.apply_bc:
            alphas = self.boundary_manager.apply_boundary_conditions(alphas)
            alpha = alphas[:, 0]
        grads = self.grad.compute_grad(x, alphas)
        grad_v = grads[:, :, :, 0]
        dt_grad_v = grads[:, :, :, 1]
        deriv_grad = Polynomials.eval_poly_arrays(self.deriv_grad_deg, self.deriv_grad_coef, x, alpha)
        v = self.v(x, t)
        if self.f_init is not None:
            v += self.f_init(x, torch.tensor(0.))
            grad_v += self.f_init.compute_grad(x, torch.tensor(0.))
            deriv_grad += self.f_init.deriv_grad(x, torch.tensor(0.))
        if self.nu> 1e-5:
            alpha2 = self.v.evaluate_alpha(t+ self.dt)
            return (dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(v, grad_v, deriv_grad)
                    - self.nu* self.grad.compute_rot_laplacian(x, alpha2))
        else:
            return dt_grad_v[:, 0, 1] - dt_grad_v[:, 1, 0] + self.rot_v_grad_v(v, grad_v, deriv_grad)
        """

    def rot_v_grad_v(self, v, grad_v, deriv_grad):
        """
        if t < 1e-5:
            grad_v = self.f_init.compute_grad(x, t)
            v = self.f_init.eval(x, t)
            # Definir rot_grad
        else:
            alpha = self.v.evaluate_alpha(t)
            grad_v = self.grad.compute_grad(x, alpha)
            deriv_grad = Polynomials.eval_poly_arrays(self.deriv_grad_deg, self.deriv_grad_coef, x, alpha)
            v = self.v(x, t)
        v = v.unsqueeze(-1)
        grad_grad_1 = torch.matmul(grad_v[:,0,:].unsqueeze(1), grad_v[:, :, 1].unsqueeze(-1))
        grad_grad_2 = torch.matmul(grad_v[:,1,:].unsqueeze(1), grad_v[:, :, 0].unsqueeze(-1))
        grad_grad = torch.cat((grad_grad_1, grad_grad_2), 1)   
        return (torch.matmul(deriv_grad, v) + grad_grad).squeeze()
        """
        v = v.unsqueeze(-1)
        grad_grad_1 = torch.matmul(grad_v[:,0,:].unsqueeze(1), grad_v[:, :, 1].unsqueeze(-1))
        grad_grad_2 = torch.matmul(grad_v[:,1,:].unsqueeze(1), grad_v[:, :, 0].unsqueeze(-1))
        grad_grad = grad_grad_1 - grad_grad_2
        
        return (torch.matmul(torch.matmul(deriv_grad, v).squeeze(), self.grad_to_rot) + grad_grad.squeeze())

    def compute_advection(self, x_n, t):
        if t < 1e-5:
            grad_vn = self.f_init.compute_grad(x_n, t)
            vn = self.f_init.eval(x_n, t)
        else:
            alpha_n = self.v.evaluate_alpha(t)
            grad_vn = self.grad.compute_grad(x_n, alpha_n)
            vn = self.v(x_n, t)
        vn = vn.reshape(vn.shape + (1,))
        return torch.squeeze(torch.matmul(grad_vn, vn), -1)

    def compute_euler_op(self, x_n, t):
        if self.nu> 1e-5:
            alpha_n = self.v.evaluate_alpha(t)
            if self.apply_bc:
                alpha_n = self.boundary_manager.apply_boundary_conditions(alpha_n)
            return self.v.deriv_time(x_n, t) + self.compute_advection(x_n, t) - self.nu * self.grad.compute_rot_laplacian(x_n, alpha_n)
        else:
            return self.v.deriv_time(x_n, t) + self.compute_advection(x_n, t)


    
    

    