import torch
import numpy as np
from ...case import Case
from itertools import product

class PolyTools():
    def eval_basis_tuple(deg_coef, x, L_mul_first_dim=False):
        if len(deg_coef)==2:
            return PolyTools.eval_basis(
                deg_coef[0], deg_coef[1], x)
        else:
            return PolyTools.eval_basis(
                deg_coef[0], deg_coef[1], x, L=deg_coef[2], L_mul_first_dim=L_mul_first_dim)

    def eval_poly_tuple(deg_coef, x, alpha, L_mul_first_dim=False):
        if len(deg_coef)==2:
            return PolyTools.eval_poly_arrays(
                deg_coef[0], deg_coef[1], x, alpha)
        else:
            return PolyTools.eval_poly_arrays(
                deg_coef[0], deg_coef[1], x, alpha, L=deg_coef[2], L_mul_first_dim=L_mul_first_dim)
        
    def eval_poly_arrays(deg, coef, x, alpha, L=None, L_mul_first_dim=False):
        x = PolyTools.eval_basis(deg, coef, x, L=L, L_mul_first_dim=L_mul_first_dim)
        return torch.matmul(x, alpha)        

    def eval_basis(deg, coef, x, L=None, L_mul_first_dim=False):
        x = x.reshape((x.shape[0],) + (deg.ndim-2)*(1,) + (x.shape[1],) + (1,))
        x = x.expand((x.shape[0],) + deg.shape)
        x = torch.pow(x, deg)        
        x = torch.prod(x, dim=-2)
        x = torch.mul(x, coef)
        if L is not None and not isinstance(L, list):
            if not L_mul_first_dim:                
                #return torch.matmul(x, torch.matmul(torch.transpose(L, 0, -1), alpha))
                return torch.matmul(L, x.unsqueeze(-1)).squeeze(-1)
                
            else:
                x = torch.matmul(L, torch.transpose(x, 1, -1).unsqueeze(-1)).squeeze(-1)
                #x = torch.matmul(L.transpose(0,1), torch.transpose(x, 1, -1).unsqueeze(-1)).squeeze(-1)
                x = torch.transpose(x, 1, -1)
        if isinstance(L, list) and L is not [None]*len(L):
            raise RuntimeError("A list of L is given for evaluating tuple")
        return x
    
    def product_space(basis_space1, basis_space2, dim_to_add=1):
        """
        - Compute the degrees and the coefficient for the product between two
        basis spaces, basis_space1 and basis_space2 which should be tuple of
        the form (degrees, coef)
        - Let $p$ denotes the number of basis functions. The first dimensions
        of basis_space1 (at least one, multiples if this space is already a 
        product of space of basis function) should have size equal to p.
        For basis_space2 the number of dimension with size equal to the number
        of basis functions should be equal to 1 (i.e. we multiply with one
        basis space at a time)
        - Last dimensions of degrees (both for basis_space1 and basis_space2): 
            -1: Monomials to sum when evaluate, 
            -2: power in x and y, 
            -3: Components.
        - Last dimensions of coef:
            -1: Monomials to sum when evaluate, 
            -2: Components.
        - That is its shape of degrees1 could be in 2D (p,p,...,p, 2,2,1)
        and for degrees2 (p,2,2,1)
        Return object add one dimension to basis_space1 (for the product of 
        the two spaces) with size equal to $p$.
        """
        deg1, coef1, _ = basis_space1
        if basis_space2 is not None:
            deg2, coef2, _ = basis_space2
        else:
            deg2, coef2 = deg1, coef1

        product_deg = (deg1.reshape(deg1.shape[0:dim_to_add] + (1,)
                       + deg1.shape[dim_to_add:]))
        product_coef = (coef1.reshape(coef1.shape[0:dim_to_add] + (1,)
                        + coef1.shape[dim_to_add:]))                       
        product_deg = (product_deg.expand(deg1.shape[0:dim_to_add] + (deg2.shape[0],)
                       + deg1.shape[dim_to_add:]))
        product_coef = (product_coef.expand(coef1.shape[0:dim_to_add] + (coef2.shape[0],)
                        + coef1.shape[dim_to_add:]))
        id = 0 if dim_to_add != 0 else 1
        product_deg = product_deg + deg2.unsqueeze(id)
        product_coef = product_coef * coef2.unsqueeze(id)

        return [product_deg, product_coef]

    def integrate_basis_rectangle(deg, coef, bounds):
        # prod = product([0,1],repeat=3)
        # a = (([0,1],)*3)
        # w[v,[0,1,2]] 
        dim = deg.shape[-2]       
        deg = deg.clone()
        coef = coef.clone()
        deg += 1
        coef = coef / deg.prod(-2)
        alpha = torch.ones(deg.shape[-1]).type_as(coef)        
        prod = list(product([0,1],repeat=dim))
        prod_value = torch.tensor(bounds)[range(dim), prod].type_as(coef)
        sign_prod = abs((np.array(prod)-1).sum(axis=1))
        integrals = PolyTools.eval_poly_tuple((deg, coef), prod_value, alpha)
        integrals *= torch.tensor((-1)**sign_prod.reshape((-1,) +(1,)*(integrals.ndim-1))).type_as(coef)
        integral = integrals.sum(0).sum(-1)
        
        return integral   

    def integrate_basis_unit_circle(deg, coef):        
        deg = deg.clone()
        coef = coef.clone()
        even_deg = 1 - (deg%2)
        even_deg = even_deg.prod(-2)
        lgamma_num = torch.lgamma(0.5*(deg+1)).sum(-2)
        lgamma_denom = torch.lgamma((0.5*(deg+1)).sum(-2))
        lgamma = torch.exp(lgamma_num - lgamma_denom) * even_deg
        
        alpha = torch.ones(deg.shape[-1])
        return 2.*lgamma*coef#torch.matmul(2.*lgamma*coef, alpha).sum(-1)
    
    def integrate_basis_circle(deg, coef, r, remove_r=0):
        deg_sum = deg.sum(-2)+1-remove_r
        alpha = torch.ones(deg.shape[-1])
        return torch.matmul((r**deg_sum)*PolyTools.integrate_basis_unit_circle(deg, coef), alpha).sum(-1)

    def integrate_basis_disk(deg, coef, r):        
        deg_sum = deg.sum(-2)+2
        alpha = torch.ones(deg.shape[-1])
        return torch.matmul((r**deg_sum/deg_sum)*PolyTools.integrate_basis_unit_circle(deg, coef), alpha).sum(-1)#(r**deg_sum/deg_sum)*PolyTools.integrate_basis_circle(deg, coef)
        #return (r**deg_sum/deg_sum)*2.*lgamma*coef

    def integrate(basis_product, boundary_dict):
        deg, coef, L = basis_product
        if boundary_dict["case"] == Case.rectangle:
            integral = PolyTools.integrate_basis_rectangle(deg, coef, boundary_dict["bounds"])
        elif boundary_dict["case"] == Case.circle:
            integral = PolyTools.integrate_basis_disk(deg, coef, boundary_dict["radius"])
        else:
            raise RuntimeError("Integrals not implemented for this domain.")  
        if L is None:            
            return integral
        else:
            if isinstance(L, list):                
                if L == [None]*len(L):
                    return integral
                L_list = L
            else:
                L_list = [L]*len(integral.shape)
            for i in range(len(integral.shape)):
                integral = torch.transpose(integral, i, -1).unsqueeze(-1)
                if L_list[i] is not None:
                    integral = torch.matmul(L_list[i], integral)
                integral = torch.transpose(integral.squeeze(-1), i, -1)
            return integral        

    def update_integral(integral_L, integral_basis, L, k):
        for i in range(len(integral_L.shape)):
            integral = torch.transpose(integral_basis, i, -1).unsqueeze(-1)
            integral_L = torch.transpose(integral_L, i, -1).unsqueeze(-1)
            integral_L[:, k, :] = torch.matmul(L[k, :], integral)            
            integral_L = torch.transpose(integral_L.squeeze(-1), i, -1)

    def projection_coef(integral, k, m):
        return integral[k, m] / integral[k, k]

    def orthonormalization(deg_in, coef_in, boundary_dict, M=None):                          
        deg = torch.transpose(deg_in.unsqueeze(0), 0, -1)
        coef = torch.transpose(coef_in.unsqueeze(0), 0, -1)
        basis = deg, coef, None     
        if M is not None:
            nb_basis = M.shape[0]
            L = M
        else:
            nb_basis = deg_in.shape[-1]
            L = torch.eye(nb_basis, nb_basis)
        
        basis_product = PolyTools.product_space(basis, basis)        
        integral_basis = PolyTools.integrate((basis_product[0], basis_product[1], L), boundary_dict)
        integral_L = integral_basis.clone()
                       
        for i in range(nb_basis):
            for j in range(i):
                proj_coef = PolyTools.projection_coef(integral_L, j, i)
                L[i, :] -= proj_coef * L[j, :]
                
            integral_L = PolyTools.integrate((basis_product[0], basis_product[1], L), boundary_dict)            
            norm_coef = torch.sqrt(integral_L[i, i])            
            L[i, :] /= norm_coef
            
            integral_L[i, :] /= norm_coef
            integral_L[:, i] /= norm_coef   

        return L
