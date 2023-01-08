from .test_space import TestSpace
from .order_manager import OrderManager
from ...case import Case
import torch
import copy


class DifferentialOp():
    def __init__(self, v, f_init=None, nu=0., order=1,
                 coef_scheme = {"alpha": 0, "beta": 0}):
        self.nu = nu
        self.v = v
        self.pressure_dict = {"apply": True, "order": v.velocity_dict["order_poly"]-1, "orthonormalization": True}#v.velocity_dict["pressure_dict"]
        self.poly_case = v.velocity_dict["poly_case"]
        self.apply_pressure =  self.pressure_dict["apply"] and self.poly_case==Case.classic_poly
        self.f_init = f_init
        self.test_space = TestSpace(copy.deepcopy(v.p), self.pressure_dict)
        self.init_basis_gpu = False        
        self.order = order        
        self.coef_scheme = coef_scheme
        self.boundary_penalization = None

        gamma = self.coef_scheme["gamma"]
        self.non_linear_term = ((1-gamma)*self.test_space.int_phi_v_grad_v 
                            - gamma*torch.transpose(self.test_space.int_phi_v_grad_v, 0, -1))                
        self.M = None
        self.b = None

    def reinit_cuda(self):
        self.v.reinit_cuda()
        self.test_space.reinit_cuda()
        self.init_basis_gpu = False

    def euler_implicit(self, t, boundary_val, dt):
        M = self.test_space.int_phi_v        
        b = self.test_space.int_phi_v_grad_v        
        alpha = self.v.evaluate_alpha(t)
        alpha2 = self.v.evaluate_alpha(t+dt)
        un = torch.matmul(M, alpha)
        un2 = torch.matmul(M, alpha2)
        u_grad_u_n2 = b * (torch.matmul(alpha2.unsqueeze(1), alpha2.unsqueeze(0)))
        b = un2 - (un - dt*(u_grad_u_n2.sum((-1, -2)) + boundary_val))
        return b        

    def solve_euler(self, t, dt, time_interval=None):
        self.test_space.init_gpus(t)
        if self.poly_case == Case.incompressible_poly:
            self.euler(t, dt, time_interval)
        else:
            self.euler_classic_poly(t)        

    def eval_pressure(self, x):
        return self.test_space.pressure_poly_eval(x, self.alpha_pressure)
    
    """
    def eval_boundary(self, t, dt):
        boundary_value, x_boundary, w = self.boundary_penalization.compute(None, t+dt)            
        boundary_val = ((w.unsqueeze(-1).unsqueeze(-1)*self.test_space.product(boundary_value, x_boundary)).sum(dim=0)).sum(dim=-1)
        return  - dt*boundary_val
    """

    def get_high_order_coef(self, t, time_interval):
        order_manager = OrderManager(self.order, time_interval)
        coef_un, coef_rhs, coef_extrapolation = order_manager.get_coef(t)

        coef_un = coef_un.type_as(t).unsqueeze(dim=1)
        coef_extrapolation = coef_extrapolation.type_as(t)

        return coef_un, coef_rhs, coef_extrapolation

    def euler(self, t, dt, time_interval):
        if time_interval == None:
            time_interval = dt
        
        self.non_linear_term = self.non_linear_term.type_as(t)
        nb_basis = self.test_space.p.nb_basis        

        M2, b = self.compute_A_b(t, dt, time_interval)

        if self.boundary_penalization is not None:
            order_manager = OrderManager(self.order, time_interval)
            _, coef_rhs, _ = order_manager.get_coef(t)#self.coef_order_euler(self.order, t)#            
            b +=  coef_rhs*self.eval_boundary(t, dt)
        if self.poly_case == Case.classic_poly:
            nb_basis_pressure = self.test_space.pressure_p.nb_basis
            b = torch.cat((b, torch.zeros(nb_basis_pressure, device=t.device)), 0)
        
        alpha_2 = torch.linalg.solve(M2, b)            
        if self.poly_case == Case.incompressible_poly:        
            self.v.set_alpha(alpha_2, t+dt)
        else:
            self.alpha_pressure = alpha_2[nb_basis:]
            self.v.set_alpha(alpha_2[:nb_basis], t+dt)
        
    def mass_matrix(self, t, dt, coef_rhs, alpha_extrapolate):
        alpha = self.coef_scheme["alpha"]
        beta = self.coef_scheme["beta"]
        
        # Construction of the matrix to invert 
        # Identity in case of orthonormalize basis functions.
        M = self.test_space.int_phi_v.clone().type_as(t)
        #Viscosity term
        if self.nu>1e-5:        
            M += beta*coef_rhs*self.nu*dt*self.test_space.int_grad_phi_grad_v
        # Non linear term        
        M += alpha*coef_rhs*dt*torch.matmul(torch.transpose(self.non_linear_term,-1,1), alpha_extrapolate)

        if self.poly_case == Case.classic_poly:
            nb_basis_pressure = self.test_space.pressure_p.nb_basis             
            A1 = torch.cat((M, -coef_rhs*dt*self.test_space.int_div_phi_p), 1)
            A2 = torch.cat((torch.transpose(coef_rhs*dt*self.test_space.int_div_phi_p, 0 ,1), torch.zeros(nb_basis_pressure, nb_basis_pressure, device=t.device)),1)
            M = torch.cat((A1, A2), 0)
        
        return M

    def rhs(self, dt, coef_un, coef_rhs, coef_extrapolation, alpha_list, alpha_extrapolate):
        alpha = self.coef_scheme["alpha"]
        beta = self.coef_scheme["beta"]

        alpha_n_mean = (coef_un * alpha_list).sum(dim=0)
        
        u_grad_u_n = (self.non_linear_term.unsqueeze(0)*torch.matmul(alpha_list.unsqueeze(-1), alpha_list.unsqueeze(-2)).unsqueeze(1)).sum((-1, -2))
        u_grad_u_n = (1-alpha)*(coef_extrapolation.unsqueeze(-1) * u_grad_u_n).sum(dim=0)
        """        
        for i in range(len(coef_un)):
            id = (len(coef_un)-i-1)
            alpha_i = self.v.evaluate_alpha(t - id*self.dt)

            #alpha_i = alpha_list[i, :]
            #u_i = torch.matmul(M, alpha_i)            
            #u_n += coef_un[i] * u_i

            alpha_n_mean += coef_un[id] * alpha_i
            u_grad_u_i = non_linear_term * (torch.matmul(alpha_i.unsqueeze(1), alpha_i.unsqueeze(0)))
            u_grad_u_n += (1-alpha)*coef_extrapolation[id] *  u_grad_u_i.sum((-1, -2))
            alpha_extrapolate += coef_extrapolation[id]*alpha_i
            
            #u_grad_u_i = phi_v_grad_v * (torch.matmul(alpha_i.unsqueeze(1), alpha_i.unsqueeze(0)))
            #u_grad_u_n += 0.5*coef_extrapolation[i] *  u_grad_u_i.sum((-1, -2))
            #M2 += 0.5*coef_rhs*coef_extrapolation[i]*self.dt*torch.matmul(torch.transpose(phi_v_grad_v,-1,1), alpha_i)
            #M2 += coef_rhs*self.dt*torch.matmul(torch.transpose(phi_v_grad_v, -1, 1), alpha_i)
        """ 
        M = self.test_space.int_phi_v
        u_n = torch.matmul(M, alpha_n_mean)
        if self.nu>1e-5:
            u_n += -(1-beta)*coef_rhs*self.nu*dt*torch.matmul(self.test_space.int_grad_phi_grad_v, alpha_extrapolate)        
        
        b = u_n - coef_rhs*dt*(u_grad_u_n )

        return b


    def compute_A_b(self, t, dt, time_interval):
        coef_un, coef_rhs, coef_extrapolation = self.get_high_order_coef(t, time_interval)

        alpha_list = self.v.evaluate_alpha_list(t, len(coef_un)-1)
        alpha_extrapolate = (coef_extrapolation.unsqueeze(dim=1) * alpha_list).sum(dim=0)

        M = self.mass_matrix(t, dt, coef_rhs, alpha_extrapolate)
        b = self.rhs(dt, coef_un, coef_rhs, coef_extrapolation, alpha_list, alpha_extrapolate)
        
        return M, b 

    

    def euler_backward(self, f, t, dt, time_interval=None):        
        if self.poly_case != Case.incompressible_poly:
            raise RuntimeError("Manual backward for Euler only implemented for incompressible poly.")
        if self.order != 1:
            raise RuntimeError("Manaul backward for Euler only implemented for order 1.")
        if self.nu>1e-5:
            raise RuntimeError("Manaul backward for Euler only implemented for nu=0.")

        self.non_linear_term = self.non_linear_term.type_as(t)        

        if time_interval == None:
            time_interval = dt
        
        coef_un, coef_rhs, coef_extrapolation = self.get_high_order_coef(t, time_interval)
        alpha_list = self.v.evaluate_alpha_list(t-dt, len(coef_un)-1)
        alpha_extrapolate = (coef_extrapolation.unsqueeze(dim=1) * alpha_list).sum(dim=0)        

        #M = self.mass_matrix(t, dt, coef_rhs, alpha_extrapolate)
        M = self.mass_matrix(t, dt, coef_rhs, alpha_extrapolate)
        
        g = torch.linalg.solve(torch.transpose(M, 0 ,1), f.unsqueeze(-1)).squeeze(-1)                

        alpha_np1 = self.v.evaluate_alpha(t)
        alpha_n = self.v.evaluate_alpha(t-dt)
        dM_dp = self.dM_dp(dt, coef_rhs)
        db_dp = self.drhs_dp(dt, alpha_n.unsqueeze(0))

        rhs = db_dp - torch.matmul(dM_dp, alpha_np1.unsqueeze(-1))
        return (g.unsqueeze(0) * rhs.squeeze(-1)).sum(-1)


    def dM_dp(self, dt, coef_rhs):
        alpha = self.coef_scheme["alpha"]        
        nb_basis = self.test_space.p.nb_basis
        # M += alpha*coef_rhs*dt*torch.matmul(torch.transpose(self.non_linear_term,-1,1), alpha_extrapolate)
        Id = torch.eye(nb_basis).type_as(self.non_linear_term)
        dM_dp = alpha*coef_rhs*dt*torch.matmul(torch.transpose(self.non_linear_term,-1,1), Id)

        return torch.transpose(torch.transpose(dM_dp, 0, -1), 1, 2)
        
    def drhs_dp(self, dt, alpha_list):
        """
        u_grad_u_n = (self.non_linear_term.unsqueeze(0)*torch.matmul(alpha_list.unsqueeze(-1), alpha_list.unsqueeze(-2)).unsqueeze(1)).sum((-1, -2))
        u_grad_u_n = (1-alpha)*(coef_extrapolation.unsqueeze(-1) * u_grad_u_n).sum(dim=0)        
        M = self.test_space.int_phi_v
        u_n = torch.matmul(M, alpha_n_mean)
        """

        alpha = self.coef_scheme["alpha"]
        
        nb_basis = self.test_space.p.nb_basis
        Id = torch.eye(nb_basis).type_as(alpha_list)
        
        #u_grad_u_n = (self.non_linear_term.unsqueeze(0)*torch.matmul(alpha_list.unsqueeze(-1), alpha_list.unsqueeze(-2)).unsqueeze(1)).sum((-1, -2))
        #u_grad_u_n = (1-alpha)*(coef_extrapolation.unsqueeze(-1) * u_grad_u_n).sum(dim=0)

        u_grad_u_n = (self.non_linear_term.unsqueeze(0)*(torch.matmul(Id.unsqueeze(-1), alpha_list).unsqueeze(1))).sum((-1, -2))        
        u_grad_u_n += (self.non_linear_term.unsqueeze(0)*((torch.matmul(Id.unsqueeze(-1), alpha_list).transpose(-1,-2)).unsqueeze(1))).sum((-1, -2))
        
        u_grad_u_n = (1-alpha)*(u_grad_u_n)
        
        M = self.test_space.int_phi_v.type_as(alpha_list)
        u_n = torch.transpose(torch.matmul(M, Id), 0, 1)        
        
        db_dp = u_n - dt*(u_grad_u_n)

        return db_dp.unsqueeze(-1)    

    def euler_classic_poly(self, t, dt):        
        order = self.order
        coef_un, coef_rhs, coef_extrapolation = self.coef_order_euler(order, t)
        nb_basis = self.test_space.p.nb_basis
        nb_basis_pressure = self.test_space.pressure_p.nb_basis
        M = self.test_space.int_phi_v
        M2 = self.test_space.int_phi_v + coef_rhs*self.nu*dt*self.test_space.int_grad_phi_grad_v
        
        #cond = torch.linalg.cond(A)
        #print("COndition number = ", cond)
        phi_v_grad_v = self.test_space.int_phi_v_grad_v
        
        alpha = torch.zeros(nb_basis, device=t.device)
        u_grad_u_n = torch.zeros(nb_basis, device=t.device)

        for i in range(len(coef_un)):
            alpha_i = self.v.evaluate_alpha(t - i*dt)
            alpha += coef_un[i] * alpha_i
            
            
            u_grad_u_i = phi_v_grad_v * (torch.matmul(alpha_i.unsqueeze(1), alpha_i.unsqueeze(0)))            
            u_grad_u_n += coef_extrapolation[i] *  u_grad_u_i.sum((-1, -2))                    
                        
        u_n = torch.matmul(M, alpha)
        f = u_n - coef_rhs*dt*(u_grad_u_n)
        if self.boundary_penalization is not None:        
            boundary_value, x_boundary, w = self.boundary_penalization.compute(None, t+dt)            
            boundary_val = ((w.unsqueeze(-1).unsqueeze(-1)*self.test_space.product(boundary_value, x_boundary)).sum(dim=0)).sum(dim=-1)
            f +=  - coef_rhs*dt*boundary_val
        b = torch.cat((f, torch.zeros(nb_basis_pressure, device=t.device)), 0)

        A1 = torch.cat((M2, -coef_rhs*dt*self.test_space.int_div_phi_p), 1)
        A2 = torch.cat((torch.transpose(coef_rhs*dt*self.test_space.int_div_phi_p, 0 ,1), torch.zeros(nb_basis_pressure, nb_basis_pressure, device=t.device)),1)
        A = torch.cat((A1, A2), 0)

        alpha_2 = torch.linalg.solve(A, b)
        self.alpha_pressure = alpha_2[nb_basis:]
        self.v.set_alpha(alpha_2[:nb_basis], t+dt)
    