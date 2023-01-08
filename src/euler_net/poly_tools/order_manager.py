import torch

class OrderManager():
    def __init__(self, order, dt):
        self.order = order
        self.dt = dt
        self.coef_order = []

        for i in range(order):
            self.coef_order.append(self.init_coef_order(i+1))

        self.len_coef = self.init_coef_order(order)[0].shape[0]

    def init_coef_order(self, order):
        if order == 1:            
            coef_un= [1.]
            coef_rhs = 1.
            coef_extrapolation = [1.]
        elif order == 2:            
            # BDF scheme
            coef_un = [4./3., -1./3.]
            coef_rhs = (2./3.)
            coef_extrapolation = [2.,-1.,]
        elif order == 3:
            # BDF scheme            
            coef_un = [18./11., -9./11., 2./11.]
            coef_rhs = (6./11.)
            coef_extrapolation = [3.,-3.,1.,]
        elif order == 4:            
            # BDF scheme
            coef_un = [48/25., -36/25., 16/25., -3/25.]
            coef_rhs = (12./25.)
            coef_extrapolation = [4.,-6.,4.,-1.,]
        else:
            raise RuntimeError("Order ", order, " not implemented.")

        return torch.tensor(coef_un).flip(0), coef_rhs, torch.tensor(coef_extrapolation).flip(0)

    def get_coef(self, t):
        order = self.order
        if ((order == 2 and t - self.dt < 0)
            or (order == 3 and t - 2*self.dt < 0)
            or (order == 4 and t - 3*self.dt < 0)):
            order = int((t[0] // self.dt) +1)
        return self.coef_order[order-1]