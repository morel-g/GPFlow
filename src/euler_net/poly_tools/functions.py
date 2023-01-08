from case import Case
import numpy as np
import torch

class Function:
    def __init__(self, case=Case.sin_func, radius=1, scale=1., nu=0., t_init=0.):
        self.case = case
        self.radius = radius
        self.scale = scale
        self.nu = nu
        self.C = self.scale#self.scale * np.pi
        self.t_init = t_init
        self.a = 1.#np.pi
        self.d = 1.#np.pi

    def __call__(self, x, t):
        return self.eval(x, t)

    def eval(self, x, t):
        t = t.clone() + self.t_init
        """
        - Case sin_func: f(x,y) = (-y, x) * sin((2*pi / r**2)*(x**2+y**2)
        - Case TGV_func: f(t,x,y) = (1 − cos(x − t)sin(y − t), 1 + sin(x − t)cos(y − t))
        """
        r = self.radius
        if self.case == Case.sin_func:
            C = (2*np.pi/(r**2))
            x1, x2 = x[:, 0], x[:, 1]
            M = torch.tensor([[0., 1.], [-1., 0.]]).type_as(x)
            y_x = torch.stack((-x2, x1), dim=-1)
            return y_x * torch.sin(C*(x1**2 + x2**2)).unsqueeze(1)
            ##return torch.matmul(x, M) * torch.sin((2*np.pi/r)*(torch.norm(x, dim=-1)**2)).reshape(-1,1)
            #return torch.matmul(x, M) * torch.sin(C*(torch.norm(x, dim=-1)**2)).reshape(-1,1)
        elif self.case == Case.TGV_func:
            C = self.C
            nu = self.nu
            x = C * x.clone()
            t = (C**2)*t.clone()
            #C=1.            
            cos, sin = torch.cos(x-t), torch.sin(x-t)
            return C*(1 + torch.stack((-cos[:, 0]*sin[:,1], sin[:, 0]*cos[: ,1]), dim=1) * torch.exp(-2.*t*nu))
        elif self.case == Case.TGV_func2:
            C = 2.*np.pi
            nu = self.nu
            #C=1.            
            cos, sin = torch.cos(C*(x-t)), torch.sin(C*(x-t))
            return (1./3.) + (2./3.)*torch.stack((cos[:, 0]*sin[:,1], -sin[:, 0]*cos[: ,1]), dim=1)
        elif self.case == Case.Beltrami_3D:
            if x.shape[-1] != 3:
                RuntimeError("Error Beltrami_3D solution is valid only for d = 3")
            a = self.a
            d = self.d
            nu = self.nu
            x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            u1 = -a*(torch.exp(a*x1)*torch.sin(a*x2+d*x3) + torch.exp(a*x3)*torch.cos(a*x1+d*x2))* torch.exp(-(d**2)*t*nu)
            u2 = -a*(torch.exp(a*x2)*torch.sin(a*x3+d*x1) + torch.exp(a*x1)*torch.cos(a*x2+d*x3))* torch.exp(-(d**2)*t*nu)
            u3 = -a*(torch.exp(a*x3)*torch.sin(a*x1+d*x2) + torch.exp(a*x2)*torch.cos(a*x3+d*x1))* torch.exp(-(d**2)*t*nu)
            return torch.stack((u1, u2, u3), dim=1)
        elif self.case == None:
            return torch.zeros(x.shape) + 1e-3*torch.rand(x.shape)
        else:
            raise RuntimeError("Unknown function.") 

    def grad(self, x, t):
        t = t.clone() + self.t_init
        r = self.radius
        if self.case == Case.sin_func:
            x1, x2 = x[:, 0], x[:, 1]
            C = (2*np.pi/(r**2))
            cos, sin = torch.cos(C*(x1**2 + x2**2)), torch.sin(C*(x1**2 + x2**2))
            e1, e2 = torch.tensor([[1., 0.]]).type_as(x), torch.tensor([[0., 1.]]).type_as(x)
            dx_f = e2*sin.unsqueeze(-1) + torch.stack((-x2, x1), dim=-1)*(2*x1*C*cos).unsqueeze(-1)
            dy_f = -e1*sin.unsqueeze(-1) + torch.stack((-x2, x1), dim=-1)*(2*x2*C*cos).unsqueeze(-1)

            return torch.stack((dx_f, dy_f), dim=-1)
        elif self.case == Case.TGV_func:
            C = self.C
            nu = self.nu
            x = C*x.clone()
            t = (C**2)*t.clone()
            #C=1.                    
            cos, sin = torch.cos(x-t), torch.sin(x-t)            
            cos_cos, sin_sin = cos[:,0]*cos[:,1], sin[:,0]*sin[:,1]
            col1 = torch.stack((sin_sin, cos_cos), dim=1)
            col2 = torch.stack((-cos_cos, -sin_sin), dim=1)
            return (C**2)*torch.stack((col1, col2), dim=-1) * torch.exp(-2.*t*nu)  
        elif self.case == Case.TGV_func2:
            C = (2*np.pi/(r**2))
            #C=1.                    
            cos, sin = torch.cos(C*(x-t)), torch.sin(C*(x-t))            
            cos_cos, sin_sin = cos[:,0]*cos[:,1], sin[:,0]*sin[:,1]
            col1 = torch.stack((-sin_sin, -cos_cos), dim=1)
            col2 = torch.stack((cos_cos, sin_sin), dim=1)
            return (2./3.)*C*torch.stack((col1, col2), dim=-1)
        elif self.case == Case.Beltrami_3D:
            if x.shape[-1] != 3:
                RuntimeError("Error Beltrami_3D solution is valid only for d = 3")
            a = self.a
            d = self.d            
            nu = self.nu
            x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            dx_u1 = (a*torch.exp(a*x1)*torch.sin(a*x2+d*x3) - 
                     a*torch.exp(a*x3)*torch.sin(a*x1+d*x2))
            dy_u1 = (a*torch.exp(a*x1)*torch.cos(a*x2+d*x3) - 
                     d*torch.exp(a*x3)*torch.sin(a*x1+d*x2))
            dz_u1 = (d*torch.exp(a*x1)*torch.cos(a*x2+d*x3) + 
                     a*torch.exp(a*x3)*torch.cos(a*x1+d*x2))

            dx_u2 = (d*torch.exp(a*x2)*torch.cos(a*x3+d*x1) +
                     a*torch.exp(a*x1)*torch.cos(a*x2+d*x3))            
            dy_u2 = (a*torch.exp(a*x2)*torch.sin(a*x3+d*x1) -
                     a*torch.exp(a*x1)*torch.sin(a*x2+d*x3))
            dz_u2 = (a*torch.exp(a*x2)*torch.cos(a*x3+d*x1) -
                     d*torch.exp(a*x1)*torch.sin(a*x2+d*x3))

            dx_u3= (a*torch.exp(a*x3)*torch.cos(a*x1+d*x2) -
                    d*torch.exp(a*x2)*torch.sin(a*x3+d*x1))
            dy_u3= (d*torch.exp(a*x3)*torch.cos(a*x1+d*x2) +
                    a*torch.exp(a*x2)*torch.cos(a*x3+d*x1))
            dz_u3= (a*torch.exp(a*x3)*torch.sin(a*x1+d*x2) -
                    a*torch.exp(a*x2)*torch.sin(a*x3+d*x1))

            dx_u = torch.stack((dx_u1, dx_u2, dx_u3), dim=-1)
            dy_u = torch.stack((dy_u1, dy_u2, dy_u3), dim=-1)
            dz_u = torch.stack((dz_u1, dz_u2, dz_u3), dim=-1)

            grad_u = -a*torch.stack((dx_u, dy_u, dz_u), dim=-1)*torch.exp(-(d**2)*t*nu)
            return grad_u
            
        elif self.case == None:
            return torch.zeros(x.shape[0],2,2)         
        else:
            raise RuntimeError("Unknown function.")

    def deriv_grad(self, x, t):
        t = t.clone() + self.t_init
        """
        Return the matrix [[dxy_f1, dyy_f1], [dxx_f2, dxy_f2]]
        """
        r = self.radius
        if self.case == Case.sin_func:
            x1, x2 = x[:, 0], x[:, 1]
            C = (2*np.pi/(r**2))
            cos, sin = torch.cos(C*(x1**2 + x2**2)), torch.sin(C*(x1**2 + x2**2))
            dxy_f1 = -2*x1*C*cos + 4*x1*(x2**2)*(C**2)*sin
            dyy_f1 = -6*x2*C*cos + 4*(x2**3)*(C**2)*sin
            dxy_f2 = 2*x2*C*cos - 4*(x1**2)*x2*(C**2)*sin
            dxx_f2 = 6*C*x1*cos - 4*(x1**3)*(C**2)*sin
            col_1 = torch.stack((dxy_f1, dxx_f2), dim=1)
            col_2 = torch.stack((dyy_f1, dxy_f2), dim=1)
            return torch.stack((col_1, col_2), dim=-1)
        elif self.case == Case.TGV_func:
            C = self.C
            nu = self.nu            
            x = C*x.clone()
            t = (C**2)*t.clone()
            #C=1.            
            cos, sin = torch.cos(x-t), torch.sin(x-t)
            cosx_siny, sinx_cosy = cos[:,0]*sin[:,1], sin[:,0]*cos[:,1]
            col1 = torch.stack((sinx_cosy, -sinx_cosy), dim=1)
            col2 = torch.stack((cosx_siny, -cosx_siny), dim=1)
            return (C**3)*torch.stack((col1, col2), dim=-1) * torch.exp(-2.*t*nu)
        elif self.case == None:
            return torch.zeros(x.shape[0], 2, 2) 

    def pressure(self, x, t):
        C = self.C
        if self.case == Case.TGV_func:
            x = C*x.clone()
            t = (C**2)*t.clone()
            t = t.clone() + self.t_init
            x = C*x.clone()
            t = (C**2)*t.clone()         
            x1, x2 = x[:, 0], x[:, 1]
            
            p = -0.25*(C**2)*(torch.cos(2*(x1-t)) + torch.cos(2*(x2-t)))*torch.exp(-4.*t*self.nu)
            p = p.unsqueeze(-1)
            
            return p
        elif self.case == Case.Beltrami_3D:
            if x.shape[-1] != 3:
                RuntimeError("Error Beltrami_3D solution is valid only for d = 3") 
            a = self.a
            d = self.d            
            nu = self.nu
            x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            p = -(a**2/2.)*(torch.exp(2*a*x1) + torch.exp(2*a*x2) + torch.exp(2*a*x3) +
                2*torch.sin(a*x3+d*x1)*torch.cos(a*x2+d*x3) +
                2*torch.sin(a*x2+d*x3)*torch.cos(a*x1+d*x2) +
                2*torch.sin(a*x1+d*x2)*torch.cos(a*x3+d*x1))*torch.exp(-2.*(d**2)*t*self.nu)
            p = p.unsqueeze(-1)
            return p
        else:
            raise RuntimeError("Unknown function.")

    def dn_u_pn(self, x, t, n):
        
        # Calculate (du / dn + pn)
        C = self.C        
        grad = self.grad(x.clone(), t.clone())
        if n.ndim==2:
            n2 = n.unsqueeze(-1)
        else:
            n2 = n
        dn_u = torch.matmul(grad, n2)
                         
        #x1, x2 = x[:, 0], x[:, 1]
        #p = -0.25*(C**2)*(torch.cos(2*(x1-t)) + torch.cos(2*(x2-t)))*torch.exp(-4.*t*self.nu)
        #p = p.unsqueeze(-1)
        p = self.pressure(x, t)
        u = self(x, t)
        u_n = (u*n.squeeze(-1)).sum(dim=-1)
        u_u_n = u * u_n.unsqueeze(-1)

        return self.nu*dn_u.squeeze() - p*n.squeeze() #- u_u_n

    def rot_laplacian(self, x, t):
        t = t.clone() + self.t_init
        if self.case == Case.TGV_func:
            C = self.C
            nu = self.nu
            x = C*x.clone()
            t = (C**2)*t.clone()         
            #C=1.            
            cos, sin = torch.cos(x-t), torch.sin(x-t)
            cosx_cosy =  cos[:,0]*cos[:,1]
 
            return (C**4)*4*cosx_cosy*torch.exp(-2.*t*nu)
        else:
            raise RuntimeError("Lalpacian not defined for ", self.case)

    def rot_dt(self, x, t):
        t = t.clone() + self.t_init
        if self.case == Case.sin_func:
            return torch.zeros(x.shape[0])
        elif self.case == Case.TGV_func:
            C = self.C
            nu = self.nu
            x = C*x.clone()
            t = (C**2)*t.clone()         
            #C=1.
            cos, sin = torch.cos(x-t), torch.sin(x-t)
            cosx_cosy =  cos[:,0]*cos[:,1]            
            cosx_siny, sinx_cosy = cos[:,0]*sin[:,1], sin[:,0]*cos[:,1]
            return (-2.*(C**4)*(sinx_cosy + cosx_siny) + 4*nu*(C**4)*cosx_cosy)* torch.exp(-2.*(C**2)*t*nu) 
        elif self.case == None:
            return torch.zeros(x.shape[0])            
