from .poly_tools import PolyTools
import torch


class GradientManager():

    def __init__(self, p):
        self.p = p        
        self.init_basis_gpu = False
        self.boundary_dict = p.boundary_dict
        self.apply_bc = p.apply_bc

        self.grad_list = self.init_gradient(p.get_degrees(), p.get_coef(), p.get_L())
        if self.p.d == 2:
            self.init_rot_laplacian()

    def compute_deriv(basis_degrees, basis_coef, deriv_update):
        """
        - The array deriv_update should be composed only on 0 and 1.
        - Only one 1 by row.
        if not np.array_equal(deriv_update, deriv_update.astype(bool)):
            raise RuntimeError("Error deriv update should contain only 0 and 1.")
        if not deriv_update.sum(dim=0) == torch.ones(deriv_update.shape[0]):
            raise RuntimeError("Invalid shape for deriv_updates.")
        """        
        grad_coef = torch.einsum("...i,jik,jk->...jk", deriv_update.float(), basis_degrees.float(), basis_coef)        
        if deriv_update.ndim != 1:
            grad_degrees = torch.tile(basis_degrees, (deriv_update.shape[1], 1, 1, 1))#shape[1]
        else:
            grad_degrees = basis_degrees
        deriv_update = torch.unsqueeze(torch.unsqueeze(deriv_update, dim=-1), -3)#torch.expand_dims(deriv_update, axis=(-3, -1))
        grad_degrees = torch.maximum((grad_degrees-deriv_update), torch.tensor(0))
        return grad_degrees, grad_coef

    def init_gradient(self, basis_degrees, basis_coef, L):        
        grad_degrees, grad_coef = GradientManager.compute_deriv(
            basis_degrees, basis_coef, torch.eye(basis_degrees.shape[-2], dtype=torch.int))
        grad_coef = torch.transpose(grad_coef, 0, 1)
        grad_degrees = torch.transpose(grad_degrees, 0, 1)
        grad_L = L
        
        return [grad_degrees, grad_coef, grad_L]

    def init_rot_laplacian(self):
        grad_deg, grad_coef, grad_L = self.grad_list
        dxx_degrees, dxx_coef = GradientManager.compute_deriv(
            grad_deg[:, 0, :, :], grad_coef[:, 0, :], torch.tensor([1, 0]))
        dyy_degrees, dyy_coef = GradientManager.compute_deriv(
            grad_deg[:, 1, :, :], grad_coef[:, 1, :], torch.tensor([0, 1]))
        self.rot_laplacian_1 = GradientManager.compute_deriv(
            torch.stack((dxx_degrees[0, :, :], dyy_degrees[0, :, :]), dim=0),
            torch.stack((dxx_coef[0, :], dyy_coef[0, :]), dim=0), torch.tensor([0, 1]))
        self.rot_laplacian_2 = GradientManager.compute_deriv(
            torch.stack((dxx_degrees[1, :, :], dyy_degrees[1, :, :]), dim=0),
            torch.stack((dxx_coef[1, :], dyy_coef[1, :]), dim=0), torch.tensor([1, 0]))

    def init_gpus(self, x):
        for i in range(len(self.grad_list)):
            if self.grad_list[i] is not None:
                self.grad_list[i] = self.grad_list[i].type_as(x)
        if self.p.d == 2:
            self.rot_laplacian_1 = (
                self.rot_laplacian_1[0].type_as(x),
                self.rot_laplacian_1[1].type_as(x))
            self.rot_laplacian_2 = (
                self.rot_laplacian_2[0].type_as(x),
                self.rot_laplacian_2[1].type_as(x))
        self.init_basis_gpu = True

    def reinit_cuda(self):
        self.p.reinit_cuda()
        self.init_basis_gpu = False

    def get_deg(self):
        return self.grad_list[0]
    
    def get_coef(self):
        return self.grad_list[1]

    def get_L(self):
        return self.grad_list[2]

    def set_deg(self, deg):
        self.grad_list[0] = deg

    def set_coef(self, coef):
        self.grad_list[1] = coef
    
    def set_L(self, L):
        self.grad_list[2] = L

    def compute_rot_laplacian(self, x, alpha):
        if not self.init_basis_gpu:
            self.init_gpus(x) 
        rot_degrees_1, rot_coef_1 = self.rot_laplacian_1
        rot_degrees_2, rot_coef_2 = self.rot_laplacian_2
        rot_1 = PolyTools.eval_poly_arrays(rot_degrees_1, rot_coef_1, x, alpha)
        rot_2 = PolyTools.eval_poly_arrays(rot_degrees_2, rot_coef_2, x, alpha)
        return rot_1.sum(dim=1) - rot_2.sum(dim=1)      

    def compute_grad(self, x, alpha):
        if not self.init_basis_gpu:
            self.init_gpus(x) 

        return PolyTools.eval_poly_tuple(self.grad_list, x, alpha)