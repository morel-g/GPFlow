from ...case import Case
from .circle_boundary import CircleBoundary
from .rectangle_boundary import RectangleBoundary
import torch


class BoundaryManager():
    def __init__(self, d, n, basis_degrees, basis_coef, boundary_dict={"case": Case.circle, "radius": 1},
                 poly_case=Case.classic_poly, debug=False):
        self.n = n
        self.d = d
        self.basis_degrees = basis_degrees
        self.basis_coef = basis_coef
        self.poly_case = poly_case
        if boundary_dict["case"] == Case.circle:
            self.boundary = CircleBoundary(d, n, basis_degrees, basis_coef,
                                           radius=boundary_dict["radius"], poly_case=poly_case,
                                           debug=debug)
        elif boundary_dict["case"] == Case.rectangle:
            self.boundary = RectangleBoundary(d, n, basis_degrees, basis_coef,
                                           bounds=boundary_dict["bounds"], poly_case=poly_case,
                                           debug=debug)
        else:
            raise RuntimeError("Unknown boundary domain.")
        self.nb_basis_out = self.boundary.nb_basis_out
        self.nb_basis_in = self.boundary.nb_basis_in      
        self.boundary_matrix = self.boundary.boundary_matrix

        #self.global_to_local_id = np.zeros(self.nb_basis_out, dtype=int)
        #self.local_to_global_id = np.zeros(self.nb_basis_in, dtype=int)

    def old_apply_boundary_conditions(self, alpha):        
        return torch.matmul(self.boundary_matrix, alpha)
