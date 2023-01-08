
from ...case import Case
from .gradient_manager import GradientManager
from .polynomials import Polynomials
from .poly_tools import PolyTools
import torch
import copy


class TestSpace():
    def __init__(self, p, pressure_dict=None, debug=False):
        self.p = p
        self.pressure_dict = pressure_dict
        if pressure_dict:
            self.pressure_p = Polynomials(d=p.d, n=pressure_dict["order"], poly_case=Case.classic_scalar_poly, apply_bc=False,
                     boundary_dict=p.boundary_dict,
                     orthonormalization=pressure_dict["orthonormalization"])
        else:
            self.pressure_p = None
        if pressure_dict:
            self.pressure = pressure_dict["apply"]        
        else:
            self.pressure = False
        self.debug = debug
        self.initialize()

    def poly_to_test_space(p):
        basis_degrees = p.get_degrees().unsqueeze(0).transpose(0, -1)
        basis_coef = p.get_coef().unsqueeze(0).transpose(0, -1)
        L = p.get_L()
        return [basis_degrees, basis_coef, L]

    def initialize(self):
        self.grad = GradientManager(self.p)        
        L = self.p.get_L()
        
        self.p.basis_poly = TestSpace.poly_to_test_space(self.p)        
        self.grad.set_deg(self.grad.get_deg().unsqueeze(0).transpose(0, -1))
        self.grad.set_coef(self.grad.get_coef().unsqueeze(0).transpose(0, -1))
        self.basis = self.p.basis_poly
        self.grad_basis = self.grad.grad_list
        self.basis_phi_v = PolyTools.product_space(self.basis, self.basis)
        self.basis_phi_v.append(L)
        basis_v_grad_v = TestSpace.v_grad_v(self.basis, self.grad_basis)
        basis_v_grad_v_phi = PolyTools.product_space(basis_v_grad_v, self.basis, dim_to_add=2)
        self.basis_phi_v_grad_v = (torch.transpose(basis_v_grad_v_phi[0], 0, 2),
                                   torch.transpose(basis_v_grad_v_phi[1], 0, 2))        
        self.basis_phi_v_grad_v = [torch.transpose(self.basis_phi_v_grad_v[0], 1, 2),
                                   torch.transpose(self.basis_phi_v_grad_v[1], 1, 2), L]
        
        dx_basis = [self.grad_basis[0][:,:,0,:,:], self.grad_basis[1][:,:,0,:], L]
        dy_basis = [self.grad_basis[0][:,:,1,:,:], self.grad_basis[1][:,:,1,:], L]
        self.dx_phi_dx_v = PolyTools.product_space(dx_basis, dx_basis)
        self.dy_phi_dy_v = PolyTools.product_space(dy_basis, dy_basis)
        self.dx_phi_dx_v.append(L)
        self.dy_phi_dy_v.append(L)

        if self.pressure:
            self.pressure_poly_eval = copy.deepcopy(self.pressure_p)
            self.pressure_p.basis_poly = TestSpace.poly_to_test_space(self.pressure_p)
            self.basis_pressure = self.pressure_p.basis_poly            
            self.div_phi_p = TestSpace.div_phi_p(self.basis_pressure, self.grad_basis)
                
        self.init_integrals()        
        
    def set_L(self, L, L_p=None):
        self.p.set_L(L)
        self.grad.set_L(L)
        self.basis_phi_v[2] = L
        self.basis_phi_v_grad_v[2] = L
        self.dx_phi_dx_v[2] = L
        self.dy_phi_dy_v[2] = L
        
        if self.pressure:            
            self.pressure_p.set_L(L_p)
            self.div_phi_p[2] = [L, L_p]
        
        self.init_integrals()        

    def init_integrals(self):
        self.int_phi_v = PolyTools.integrate(self.basis_phi_v, self.p.boundary_dict)
        self.int_phi_v_grad_v = PolyTools.integrate(self.basis_phi_v_grad_v, self.p.boundary_dict)
        self.int_grad_phi_grad_v = PolyTools.integrate(self.dx_phi_dx_v, self.p.boundary_dict) + PolyTools.integrate(self.dy_phi_dy_v, self.p.boundary_dict)
        self.init_basis_gpu = False
        self.p.L_mul_first_dim = True
        if self.pressure:
            self.int_div_phi_p = PolyTools.integrate(self.div_phi_p, self.p.boundary_dict)
            # To delete?
            self.pressure_p.L_mul_first_dim = True
        if not self.debug:
            del self.basis_phi_v_grad_v
    
    def reinit_cuda(self):
        self.init_basis_gpu = False        
        self.grad.reinit_cuda()

    def init_gpus(self, x):
        self.int_phi_v = self.int_phi_v.type_as(x)
        self.int_phi_v_grad_v = self.int_phi_v_grad_v.type_as(x)
        self.int_grad_phi_grad_v = self.int_grad_phi_grad_v.type_as(x)
        self.int_div_phi_p = self.int_div_phi_p.type_as(x)
        self.init_basis_gpu = True
    
    def product_grad(self, v_x, x):
        alpha = torch.tensor([1.]).type_as(x)

        return (v_x.unsqueeze(1) * self.grad.compute_grad(x, alpha)).sum(dim=-1)

    def product(self, v_x, x):
        if not self.init_basis_gpu:
            self.init_gpus(x)
        
        alpha = torch.tensor([1.]).type_as(x)

        return v_x.unsqueeze(1) * self.p(x, alpha)
    
    def div_phi_p(basis_p, basis_grad):
        deg_p, coef_p, L_p = basis_p
        deg_grad_v, coef_grad_v, L = basis_grad
        
        produc_deg_dx, product_coef_dx = PolyTools.product_space(
           (deg_grad_v[:, 0, 0, :, :].unsqueeze(-3),  coef_grad_v[:, 0, 0, :].unsqueeze(-2), None), basis_p)
        product_deg_dy, product_coef_dy = PolyTools.product_space(
            (deg_grad_v[:, 1, 1, :, :].unsqueeze(-3), coef_grad_v[:, 1, 1, :].unsqueeze(-2), None), basis_p)
        
        div_deg = torch.cat((produc_deg_dx, product_deg_dy), -1)
        div_coef = torch.cat((product_coef_dx, product_coef_dy), -1)

        return [div_deg, div_coef, [L, L_p]]

    def v_grad_v(basis_v, basis_grad):
        """
        Compute the product of two basis spaces corresponding to $(grad v) \cdot v$
        if the last dimension is sum (matrix vector multiplication).
        Denoting $p$ the number of basis functions shape of the inputs are
        (p, 2, 2, 1) for v and (p, 2, 2, 2, 1) for grad_v.
        Return object has shape (p, 2, 2, 2) and should be sum over last dimension
        when evaluationg (that is multiply with alpha = (1., 1.))
        """
        deg_v, coef_v, L = basis_v
        deg_grad_v, coef_grad_v, L = basis_grad

        deg_v1 = deg_v[:, 0, :, :]
        coef_v1 = coef_v[:, 0, :]
        dim = deg_v1.shape[-2]
        coef_v1 = coef_v1.unsqueeze(-2).expand(
            coef_v1.shape[:-1] + (dim,) + (coef_v1.shape[-1],))            
        deg_v1 = deg_v1.unsqueeze(-3).expand(
            deg_v1.shape[:-2]+ (dim,) + deg_v1.shape[-2:])
        deg_v2 = deg_v[:,1,:,:]
        coef_v2 = coef_v[:,1,:]
        coef_v2 = coef_v2.unsqueeze(-2).expand(
            coef_v2.shape[:-1] + (dim,) + (coef_v2.shape[-1],))            
        deg_v2 = deg_v2.unsqueeze(-3).expand(
            deg_v2.shape[:-2]+ (dim,) + deg_v2.shape[-2:])
        product_grad_deg1, product_grad_coef1 = PolyTools.product_space(
            (deg_v1, coef_v1, None),(deg_grad_v[:, :, 0, :, :], coef_grad_v[:, :, 0, :], None))

        product_grad_deg2, product_grad_coef2 = PolyTools.product_space(
            (deg_v2, coef_v2, None),(deg_grad_v[:, :, 1, :, :], coef_grad_v[:, :, 1, :], None))
        
        v_gradv_deg = torch.cat((product_grad_deg1, product_grad_deg2), -1)
        v_gradv_coef = torch.cat((product_grad_coef1, product_grad_coef2), -1)

        return(v_gradv_deg, v_gradv_coef, L)