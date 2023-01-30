import os
import numpy as np
import torch

from src.save_load_obj import load_obj
from src.gp_flows.network import Network
from src.case import Case
from src.extend_vae import ExtendVAE
from libs.disentangling_vae.utils.datasets import get_dataloaders
from libs.disentangling_vae.disvae.utils.modelIO import (
    load_model,
    load_metadata,
)

RES_DIR = "libs/disentangling_vae/results"


def encode(model, x):
    mean, log_var = model.encoder(x)
    return model.reparameterize(mean, log_var)


def save_latent_samples(dir, model, dataset, device, latent_path, name=""):
    x_latent = np.zeros((0, model.latent_dim))
    data_loader = get_dataloaders(dataset, batch_size=128, shuffle=True)
    for x in data_loader:
        z = encode(model, x[0].to(device)).cpu().detach().numpy()
        x_latent = np.concatenate((x_latent, z))
    latent_path = latent_path + "/x_latent_" + name + ".npy"
    print("saving latent variable as ", latent_path)
    np.save(latent_path, x_latent)
    # return samples


def main_vae(
    exp_name,
    net=None,
    device=torch.device("cpu"),
    latent_path="datasets/latent_var",
):
    """Save encoded data of some VAE.

    Args:
        exp_name: The VAE to encode the data.
        device: Device used. Defaults to torch.device("cpu").
    """
    model_dir = os.path.join(RES_DIR, exp_name)
    meta_data = load_metadata(model_dir)
    dataset = meta_data["dataset"]
    vae = load_model(model_dir)
    vae.eval()
    vae.to(device)
    if net is not None:
        model = ExtendVAE(vae, net)
        model.latent_case = Case.default_vae
        print("Saving initial latent points.")
        save_latent_samples(
            model_dir,
            model,
            dataset,
            device,
            latent_path,
            name=exp_name + "_latent",
        )
        model.latent_case = Case.apply_nf
        print("Saving NF points.")
        save_latent_samples(
            model_dir,
            model,
            dataset,
            device,
            latent_path,
            name=exp_name + "apply_nf",
        )
        model.latent_case = Case.apply_gp
        print("Saving GP points.")
        save_latent_samples(
            model_dir,
            model,
            dataset,
            device,
            latent_path,
            name=exp_name + "apply_gp",
        )
    else:
        model = vae
        model.latent_case = Case.latent_case
        save_latent_samples(
            model_dir, model, dataset, device, latent_path, name=exp_name
        )
    return


if __name__ == "__main__":
    apply_map = True
    device = torch.device("cuda:2")
    if not apply_map:
        exp_name = "btcvae_mnist"
        main_vae(exp_name, device=device)
    else:
        ckpt_path = "outputs/flow_saved_model/dsprites/ot_ffjord_euler_2/Checkpoint_epoch=2149-val_loss_gp=5.260.ckpt"
        load_path = ckpt_path[: ckpt_path.index("/Checkpoint")]
        data = load_obj(load_path + "/data.obj")
        data_type = data.data_type
        exp_name = "btcvae_" + data_type
        net = Network.load_from_checkpoint(ckpt_path, data=data)
        net.eval()
        net.map.eval()
        net.to(device)
        main_vae(exp_name, net=net, device=device, latent_path=load_path)
