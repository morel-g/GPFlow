import torch
from .poly_coef_generator import PolyCoefGenerator
from ..boundary.boundary_manager import BoundaryManager
from .poly_tools import PolyTools
from ...case import Case
from ...precision import torch_float_precision

class Polynomials():
    def __init__(self, d=2, n=0, poly_case=Case.classic_poly, apply_bc=False,
                 boundary_dict={}, orthonormalization=True):
        # basis_degrees should be of shape (d,d,n) and basis_coef (d, n) where d is the dimension
        # of the space (e.g. 2 for spirals) and n is the number of basis functions.
        gen_poly = PolyCoefGenerator(d, n, poly_case)
        self.basis_poly = [gen_poly.basis_degrees, gen_poly.basis_coef, None]
        if apply_bc:
            self.boundary_manager = BoundaryManager(
                d, n, gen_poly.basis_degrees, gen_poly.basis_coef, poly_case=poly_case, boundary_dict=boundary_dict)
            self.nb_basis = self.boundary_manager.nb_basis_in            
            self.basis_poly[2] = self.boundary_manager.boundary_matrix.transpose(0, 1)

        else:            
            self.nb_basis = gen_poly.nb_basis
            
        self.L_mul_first_dim = False        
        self.orthonormalization = orthonormalization
        self.poly_case = poly_case
        
        self.boundary_dict = boundary_dict            
        self.d = d
        self.n = n
        self.apply_bc = apply_bc        
        self.initialize()

    def initialize(self):
        if self.basis_poly[2] is not None:
            basis_2 = self.basis_poly[2].clone().detach()#torch.tensor(self.basis_poly[2])
        else:
            basis_2 = None
        self.basis_poly = [torch.tensor(self.basis_poly[0], dtype=torch.int),
            torch.tensor(self.basis_poly[1], dtype=torch_float_precision), basis_2]
        self.init_basis_gpu = False
        if self.orthonormalization:
            L = PolyTools.orthonormalization(self.basis_poly[0], self.basis_poly[1], self.boundary_dict, self.basis_poly[2])
            self.basis_poly[2] = torch.tensor(L.cpu().detach().numpy())

    def get_degrees(self):
        return self.basis_poly[0]

    def get_coef(self):
        return self.basis_poly[1]

    def get_L(self):
        return self.basis_poly[2]

    def set_L(self, L):
        # Set the matrix L to multiply dim 0
        self.basis_poly[2] = L

    def set_basis(self, basis_degrees, basis_coef, n,
                boundary_dict={}, L=None):
        self.basis_poly = [basis_degrees, basis_coef, L]
        self.d = basis_coef.shape[0]
        self.nb_basis = basis_coef.shape[-1]
        self.order_last_id = None        
        self.init_basis_gpu = False
        self.n = n

    def __call__(self, x, alpha):
        return self.eval(x, alpha)

    def reinit_cuda(self):
        self.init_basis_gpu = False

    def init_gpus(self, x):
        basis_0 = self.basis_poly[0].type_as(x)
        basis_1= self.basis_poly[1].type_as(x)
        if self.basis_poly[2] is not None:
            basis_2 = self.basis_poly[2].type_as(x)
        else:
            basis_2 = self.basis_poly[2]
        self.basis_poly = [basis_0, basis_1, basis_2]           
        self.init_basis_gpu = True

    def grad(self, x, alpha):
        return self.grad_manager.compute_grad(x, alpha)

    def eval(self, x, alpha):
        if not self.init_basis_gpu:
            self.init_gpus(x)        

        return PolyTools.eval_poly_tuple(self.basis_poly, x, alpha, L_mul_first_dim=self.L_mul_first_dim)

    def eval_basis(self, x):
        if not self.init_basis_gpu:
            self.init_gpus(x)
        return PolyTools.eval_basis_tuple(self.basis_poly, x, L_mul_first_dim=self.L_mul_first_dim)