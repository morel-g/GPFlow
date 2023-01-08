from boundary.boundary_generator import BoundaryGenerator
from case import Case
from poly_tools.functions import Function

class BoundaryPenalization():
    def __init__(self, d, v, penalization_dict, boundary_dict):
        self.case = penalization_dict["case"]
        self.d = d
        self.v = v
        self.penalization_dict = penalization_dict
        self.generator = BoundaryGenerator(d, boundary_dict)
        if self.case == Case.dirichlet or self.case == Case.pressure:
            self.f_boundary = penalization_dict["boundary func"]#Function(case=penalization_dict["boundary func"])

    def set_device(self, device):
        self.generator.device = device

    def compute(self, nb_points, t=0):
        if self.case == Case.neumann:
            x, n = self.generator.generate(nb_points, neumann=True)
            return self.v(x, t) * n
        elif self.case == Case.dirichlet:
            x, n = self.generator.generate(nb_points)
            v_x = self.v(x,t)
            v_dot_n = (v_x*n).sum(dim=1)
            return (v_x - self.f_boundary(x, t))#[v_dot_n<=0]
        elif self.case == Case.periodic:
            x1, x2 = self.generator.generate(nb_points, periodic=True)
            return self.v(x1,t) - self.v(x2,t)
        elif self.case == Case.pressure:
            x, n, w = self.generator.generate(nb_points)
            n = n.unsqueeze(-1)
            return -self.f_boundary.dn_u_pn(x, t, n), x, w
        else:
            raise RuntimeError("Unknown boundary conditions.")
