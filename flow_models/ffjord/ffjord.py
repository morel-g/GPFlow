# Source: https://github.com/rtqichen/ffjord/
from .layers.container import SequentialFlow
from .layers.cnf import CNF
from .layers.odefunc import ODEfunc, divergence_bf, divergence_approx
from .layers.odefunc import ODEnet
import torch


class FFJORD(torch.nn.Module):
    def __init__(
        self,
        dim,
        n_neurons,
        nblocks=1,
        non_linearity="tanh",
        bruteforce_eval=True,
    ):
        super(FFJORD, self).__init__()
        # nblocks=num_blocks
        def build_cnf():
            diffeq = ODEnet(
                hidden_dims=n_neurons,
                input_shape=(dim,),
                strides=None,
                conv=False,
                layer_type="concatsquash",
                nonlinearity=non_linearity,
            )
            odefunc = ODEfunc(
                diffeq=diffeq,
                divergence_fn="approximate",
                residual=False,
                rademacher=False,
            )
            cnf = CNF(
                odefunc=odefunc,
                T=1.0,
                train_T=False,
                regularization_fns=None,
                solver="dopri5",
            )
            return cnf

        chain = [build_cnf() for _ in range(nblocks)]

        self.model = SequentialFlow(chain)
        self.bruteforce_eval = bruteforce_eval

    def forward(self, x):
        zero = torch.zeros(x.shape[0], 1).type_as(x)
        model_x = self.model(x, zero)
        return model_x[0], -model_x[1].squeeze(-1)  # run model forward

    def backward(self, x):
        return self.model(x, reverse=True)

    def eval_mode(self):
        if self.bruteforce_eval:
            for cnf in self.model.chain:
                cnf.odefunc.divergence_fn = divergence_bf

    def eval(self):
        self.model.eval()
        self.eval_mode()

    def train(self, mode=True):
        self.model.train(mode)
        if mode:
            for cnf in self.model.chain:
                cnf.odefunc.divergence_fn = divergence_approx
        else:
            self.eval_mode()
