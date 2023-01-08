if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from numbers import Integral
from poly_velocity import PolyVelocity
from case import Case
import numpy as np
from scipy.integrate import dblquad, tplquad
from poly_tools.differential_op import DifferentialOp
import torch
import quadpy


class ProjectionL2():
    def __init__(self, data, f):
        self.data = data
        self.f = f
        velocity_dict = data.velocity_dict.copy() 
        velocity_dict["time state"] = Case.stationary
        self.v = PolyVelocity(data, velocity_dict=velocity_dict, nb_layers=1)
        self.differential_op = DifferentialOp(self.v, nu=data.nu, order=data.order_scheme,
                                              direct_solve=data.direct_solve, 
                                              coef_scheme=data.coef_scheme)
        print("start projecting")
        self.project()
        print("end projecting")

    def project(self):
        nb_basis = self.v.p.nb_basis
        data = self.data
        
        def f_dot_ei_cube(ei, x):            
            x = torch.transpose(torch.tensor(x),0 ,1)
            t = torch.tensor([0.])
            return (self.f(x, t)*self.v.p(x, ei)).sum(dim=-1).cpu().detach().numpy()        

        def f_dot_ei_circle(ei, x):
            x = torch.transpose(torch.tensor(x),0 ,1)            
            t = torch.tensor([0.])
            return ((self.f(x, t)*self.v.p(x, ei)).sum(dim=-1)).cpu().detach().numpy()

        def f_dot_ei_rectangle2(ei, x1, x2):
            x = torch.tensor([[x1, x2]])
            t = torch.tensor([0.])
            return (self.f(x, t)*self.v.p(x, ei))[0, :].sum().cpu().detach().numpy()

        def f_dot_ei_cube2(ei, x1, x2, x3):
            x = torch.tensor([[x1, x2, x3]])
            t = torch.tensor([0.])
            return (self.f(x, t)*self.v.p(x, ei))[0, :].sum().cpu().detach().numpy()
        """
        def f_dot_ei_circle2(ei, r, theta):
            theta = torch.tensor(theta)
            x = torch.tensor([[r*torch.cos(theta), r*torch.sin(theta)]])
            t = torch.tensor([0.])
            return r*(self.f(x, t)*self.v.p(x, ei)).sum().cpu().detach().numpy()
        """
        rhs = torch.zeros(nb_basis)
        for i in range(nb_basis):
            ei = torch.zeros(nb_basis)
            ei[i] = 1
            if data.boundary_dict["case"] == Case.rectangle:
                bounds = data.boundary_dict["bounds"]
                if data.dim==2:
                    x_min, x_max, y_min, y_max = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
                    scheme = quadpy.c2.get_good_scheme(16)
                    integral = scheme.integrate( lambda x: f_dot_ei_cube(ei, x),
                    [[[x_min, y_min], [x_max, y_min]], [[x_min, y_max], [x_max, y_max]]])
                elif data.dim==3:
                    """
                    scheme = quadpy.c3.get_good_scheme(11)
                    integral = scheme.integrate(
                        lambda x: f_dot_ei_cube(ei, x),
                        quadpy.c3.cube_points(*bounds),
                        )
                    """
                    x_min, x_max, y_min, y_max, z_min, z_max = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1]                  
                    integral, res = tplquad(lambda x, y, z: f_dot_ei_cube2(ei, x, y, z), x_min, x_max, y_min, y_max, z_min, z_max)
                    print("residual = ", res)
                    
                    
                else:
                    RuntimeError("Projection on rectangle not implemented for dim>3")
                #integral, res = dblquad(lambda x, y: f_dot_ei_rectangle(ei, x, y), x_min, x_max, y_min, y_max)
            else:
                r = data.boundary_dict["radius"]
                scheme = quadpy.s2.get_good_scheme(16)
                integral = scheme.integrate(lambda x: f_dot_ei_circle(ei, x), [0.0, 0.0], r)

                #Not working: integral, res = dblquad(lambda r, theta: f_dot_ei_circle2(ei, r, theta), 0, radius, 0, 2.*np.pi)
                #Working: integral, res = dblquad(lambda x, y: f_dot_ei_rectangle2(ei, x, y), -r, r, lambda x: -np.sqrt(r**2-x**2), lambda x: np.sqrt(r**2-x**2))
            rhs[i] = integral
        M = self.differential_op.test_space.int_phi_v
        self.alpha_init = torch.linalg.solve(M, rhs)        
    
    