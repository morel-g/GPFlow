import torch
import numpy as np


class Compose(torch.nn.Module):
    def __init__(self, layers):
        super(Compose, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.euler_coef_penalization = self.layers[0].euler_coef_penalization

    def __call__(self, z, reverse=False, save_trajectories=False):
        # if id_final_layer != -1 and len(self.layers) != 1:
        #     raise RuntimeError("Can only fix id_final_layer ofr one block.")
        if not reverse:
            return self.forward(
                z, save_trajectories  # , id_final_layer=id_final_layer
            )
        else:
            return self.backward(z, save_trajectories)

    def forward(self, z, save_trajectories=False, id_final_layer=-1):
        for id, layer in enumerate(self.layers):
            z = layer(
                z,
                save_trajectories=save_trajectories,
                id_final_layer=id_final_layer,
            )
        self.compute_lagrangian()
        return z

    def backward(self, z, save_trajectories=False):
        for id, layer in enumerate(reversed(self.layers)):
            z = layer(z, save_trajectories=save_trajectories, reverse=True)
        return z

    def get_trajectories(self):
        x_trajectories, v_trajectories = [], []
        for layer in self.layers:
            x_trajectories.extend(
                [
                    layer.x_trajectories[i].detach().cpu().numpy()
                    for i in range(len(layer.x_trajectories))
                ]
            )
            v_trajectories.extend(
                [
                    layer.v_trajectories[i].detach().cpu().numpy()
                    for i in range(len(layer.v_trajectories))
                ]
            )
        return np.array(x_trajectories), np.array(v_trajectories)

    def regularization_coef(self):
        if (
            len(self.layers) > 1
            and self.layers[0].euler_coef_penalization is not None
        ):
            raise RuntimeError(
                "Euler penalization only implemented for 1 num_blocks"
            )

        return self.layers[0].regularization_value

    def compute_lagrangian(self):
        self.lagrangian = self.layers[0].lagrangian
        for l in self.layers[1:]:
            self.lagrangian += l.lagrangian

        return self.lagrangian
