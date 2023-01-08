# source https://github.com/CW-Huang/CP-Flow/
from .lib.icnn import ICNN3
from .lib.flows.flows import SequentialFlow, ActNorm, InvertibleLinear, IAFTemplate, NAFDSF
import torch

class NAF(torch.nn.Module):
    def __init__(self, dim, n_neurons, nb_layers, nblocks=1):
        super(NAF, self).__init__()
        depth = nb_layers
        flows = list()
        flows.extend([ActNorm(dim)])
        for _ in range(nblocks):
            flows.extend([NAFDSF(dim, n_neurons, depth, ndim=16), InvertibleLinear(dim), ActNorm(dim)])
        self.flow = SequentialFlow(flows)

    def forward(self, x):
        return self.flow.forward_transform(x, context=None)

class IAF(torch.nn.Module):
    def __init__(self, dim, n_neurons, depth, nblocks=1):
        super(IAF, self).__init__()

        flows = list()
        flows.extend([ActNorm(dim)])
        for _ in range(nblocks):
            flows.extend([IAFTemplate(dim, n_neurons, depth), InvertibleLinear(dim), ActNorm(dim)])
        self.flow = SequentialFlow(flows)
        

        
    def forward(self, x):
        return self.flow.forward_transform(x, context=None)

    """
    def backward(self, x):
        return self.flow.reverse(x)    

    
    def eval(self):
        self.flow.eval()
        for f in self.flow.flows[1::2]:
            f.no_bruteforce = False

    def train(self, mode):        
        for f in self.flow.flows[1::2]:
            f.no_bruteforce = mode
        self.flow.train(mode)
    """