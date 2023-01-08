# source https://github.com/CW-Huang/CP-Flow/
from .lib.icnn import ICNN3
from .lib.flows.flows import SequentialFlow, ActNorm
from .lib.flows.cpflows import DeepConvexFlow
import torch


class CPFlow(torch.nn.Module):
    def __init__(
        self,
        dim,
        n_neurons,
        nb_layers,
        nblocks=1,
        bruteforce_eval=False,
        softplus_type="softplus",
    ):
        super(CPFlow, self).__init__()
        depth = nb_layers
        icnns = [
            ICNN3(
                dim,
                n_neurons,
                depth,
                symm_act_first=True,
                softplus_type=softplus_type,
                zero_softplus=True,
            )
            for _ in range(nblocks)
        ]
        if nblocks == 1:
            # for printing the potential only
            layers = [None] * (nblocks + 1)
            # noinspection PyTypeChecker
            layers[0] = ActNorm(dim)
            layers[1:] = [
                DeepConvexFlow(icnn, dim, unbiased=False, bias_w1=-0.0)
                for _, icnn in zip(range(nblocks), icnns)
            ]
        else:
            layers = [None] * (2 * nblocks + 1)
            layers[0::2] = [ActNorm(dim) for _ in range(nblocks + 1)]
            layers[1::2] = [
                DeepConvexFlow(
                    icnn, dim, unbiased=False, bias_w1=-0.0, trainable_w0=False
                )
                for _, icnn in zip(range(nblocks), icnns)
            ]

        self.flow = SequentialFlow(layers)
        self.bruteforce_eval = bruteforce_eval

    def forward(self, x):
        return self.flow.forward_transform(x, context=None)

    def backward(self, x):
        return self.flow.reverse(x)

    def parameters(self):
        return self.flow.parameters()

    def eval(self):
        self.flow.eval()
        if self.bruteforce_eval:
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False

    def train(self, mode):
        if self.bruteforce_eval:
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = mode
        self.flow.train(mode)
