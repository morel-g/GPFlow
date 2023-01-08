import torch
from .bnaf_module import (
    Tanh,
    BNAFModule,
    MaskedWeight,
    Permutation,
    Sequential,
)


class BNAF(torch.nn.Module):
    def __init__(self, dim, n_neurons, nb_layers, nblocks=1):
        super(BNAF, self).__init__()
        residual = "gated"  # or "normal"
        flows = []
        for f in range(nblocks):
            layers = []
            for _ in range(nb_layers - 1):
                layers.append(MaskedWeight(n_neurons, n_neurons, dim=dim,))
                layers.append(Tanh())

            flows.append(
                BNAFModule(
                    *(
                        [MaskedWeight(dim, n_neurons, dim=dim), Tanh(),]
                        + layers
                        + [MaskedWeight(n_neurons, dim, dim=dim)]
                    ),
                    res=residual if f < nblocks - 1 else None
                )
            )

            if f < nblocks - 1:
                flows.append(Permutation(dim, "flip"))

        self.flow = Sequential(*flows)  # .to(args.device)

    def forward(self, x):
        return self.flow.forward(x)
