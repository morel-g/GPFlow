import torch
from src.case import Case
from src.save_load_obj import load_obj, save_obj
from src.gp_flows.network import Network
from src.extend_vae import ExtendVAE
from src.data_module import DataModule
from src.plots.helper_plot_2d import compute_outputs_2d
from data_helpers.data_parser import parse_viz
from data_helpers.data_type import toy_data_type, latent_data_type
from src.plots.outputs import compute_ot_costs

from libs.disentangling_vae.disvae.utils.modelIO import (
    load_model,
    load_metadata,
)
from libs.disentangling_vae.utils.datasets import get_dataloaders
from libs.disentangling_vae.utils.visualize import Visualizer
from libs.disentangling_vae.utils.viz_helpers import get_samples
from libs.disentangling_vae.metrics.dci import compute_dci
from sklearn.model_selection import train_test_split

import os
import numpy as np
import PIL
from PIL import Image, ImageFont
from pathlib import Path
from tabulate import tabulate


def merge_imgs(output_dir, imgs_name, imgs_tilte, id, delete_imgs=not True):
    images = [Image.open(x).resize((464, 662)) for x in imgs_name]
    widths, heights = zip(*(i.size for i in images))
    # print("widths, heights = ", widths, heights)
    padding_height = 40  # 40
    padding_width = 10
    font_size = 24  # 20

    total_width = sum(widths) + len(widths) * (padding_width - 1)
    max_height = max(heights) + padding_height

    new_im = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, padding_height))
        x_offset += im.size[0] + padding_width

    font = ImageFont.truetype("arialbd.ttf", font_size)
    draw = PIL.ImageDraw.Draw(new_im)
    [
        draw.text(
            (  # 80+
                i * padding_width
                + np.array(widths)[:i].sum()
                + widths[i] // 2,
                padding_height // 2,
            ),
            title,
            fill=(0, 0, 0),
            font=font,
            anchor="mm",
        )
        for i, title in enumerate(imgs_tilte)
    ]

    new_im.save(output_dir + "/traverse_comparison_" + str(id) + ".jpg")

    if delete_imgs:
        [os.remove(img) for img in imgs_name]


def compute_metrics(vae, dataset):
    test_loader = get_dataloaders(dataset, batch_size=512, shuffle=False)

    vae.latent_case = Case.default_vae
    print("Computing latent dci...")
    dci_vae = get_dci(test_loader, vae)
    # dci_nf = dci_vae
    # dci_gp = dci_vae
    print("Computing nf dci...")
    vae.latent_case = Case.apply_nf
    dci_nf = get_dci(test_loader, vae)
    print("Computing gp dci...")
    vae.latent_case = Case.apply_gp
    dci_gp = get_dci(test_loader, vae)

    dcis = {"Latent": dci_vae, "NF": dci_nf, "GP": dci_gp}
    headers = [key for key in dci_vae]
    tab_values = []
    for key, dci in dcis.items():
        dci_val = [key]
        # Consider only the average values.
        [dci_val.append(value[-1]) for _, value in dci.items()]
        tab_values.append(dci_val)
    print(tabulate(tab_values, headers=headers))


def get_dci(test_loader, vae):
    latent_var = np.zeros((0, vae.latent_dim))
    # Assuming Dsprites
    latent_labels = np.zeros((0, 6))
    for batch in test_loader:
        x, y = batch
        z = vae.reparameterize(*vae.encoder(x.to(device)))
        latent_var = np.concatenate(
            (latent_var, z.detach().cpu().numpy()), axis=0
        )
        latent_labels = np.concatenate((latent_labels, y), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(
        latent_var,
        latent_labels[:, 1:],  # For dSprites we do not consider shape factor
        test_size=0.2,
        random_state=42,
    )
    x_train, y_train = x_train[:40000], y_train[:40000]
    x_test, y_test = x_test[:10000], y_test[:10000]
    dci = compute_dci(x_train, y_train, x_test, y_test)
    return dci


def compute_viz_outputs(
    viz,
    samples,
    name,
    nb_reconstruct_samples=1,
    nb_cols=7,
    nb_rows=10,
    # is_right_text=True,
):
    with torch.no_grad():
        gif_name = ("_").join(["gif", name, "interp"]) + ".gif"
        viz.gif_traversals(
            samples[:nb_cols, ...],
            n_latents=nb_rows,
            name=gif_name,
        )
        for i in range(nb_reconstruct_samples):
            png_name = ("_").join(["traverse", name, str(i)]) + ".png"
            viz.reconstruct_traverse(
                samples[i:, ...],
                is_posterior=True,
                n_latents=nb_rows,
                n_per_latent=nb_cols,
                is_show_text=False,  # True,
                show_reconstruct=False,
                name=png_name,
            )


def viz_latent(
    data, net, device, load_path, use_vae_dir_for_outputs=True, name=""
):
    data_type = data.data_type
    # res_dir = "disentangling_vae/results"
    vae_dir = "libs/disentangling_vae/results"
    model_dir = os.path.join(vae_dir, "btcvae_" + data_type)

    if use_vae_dir_for_outputs:
        output_dir = model_dir
    else:
        output_dir = load_path + "/figures"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    meta_data = load_metadata(model_dir)
    model = load_model(model_dir)
    dataset = meta_data["dataset"]
    print("- Save results to ", output_dir)

    model.eval()  # don't sample from latent: use mean

    model.to(device)

    vae = ExtendVAE(model, net)

    nb_cols, nb_rows = 7, 10
    num_samples = nb_cols * nb_rows
    nb_reconstruct_samples = 7
    samples = get_samples(dataset, num_samples, idcs=[])
    # data_loader = get_dataloaders(dataset, batch_size=512, shuffle=False)
    viz = Visualizer(
        model=vae,
        model_dir=model_dir,
        output_dir=output_dir,
        dataset=dataset,
        max_traversal=2,
        loss_of_interest="kl_loss_",
        upsample_factor=1,
    )

    if metric and data_type == Case.dsprites:
        compute_metrics(vae, dataset)

    latent_name = ("_").join(filter(None, ["latent", name]))
    nf_name = ("_").join(filter(None, ["nf", name]))
    gp_name = ("_").join(filter(None, ["gp", name]))

    print("Computing latent...")
    vae.latent_case = Case.default_vae
    compute_viz_outputs(
        viz,
        samples,
        latent_name,
        nb_reconstruct_samples,
        nb_cols,
        nb_rows,
    )
    print("Computing nf...")
    vae.latent_case = Case.apply_nf
    compute_viz_outputs(
        viz,
        samples,
        nf_name,
        nb_reconstruct_samples,
        nb_cols,
        nb_rows,
    )

    print("Computing gp...")
    vae.latent_case = Case.apply_gp
    compute_viz_outputs(
        viz,
        samples,
        gp_name,
        nb_reconstruct_samples,
        nb_cols,
        nb_rows,
    )

    euler = " + EULER" if data.euler_dict["use_euler"] else ""
    for i in range(nb_reconstruct_samples):
        merge_imgs(
            output_dir,
            [
                output_dir
                + "/traverse_"
                + latent_name
                + "_"
                + str(i)
                + ".png",
                output_dir + "/traverse_" + nf_name + "_" + str(i) + ".png",
                output_dir + "/traverse_" + gp_name + "_" + str(i) + ".png",
            ],
            ["Initial latent space", "FFJORD", "FFJORD + GP" + euler],
            id=i,
        )


if __name__ == "__main__":
    args = parse_viz()

    # For reproducibility
    # np.random.seed(42)
    # torch.manual_seed(42)
    # random.seed(42)

    use_vae_dir_for_outputs = False
    metric = args.metric
    ckpt_path = args.ckpt_path
    nb_layers_eval = args.nb_layers_eval
    compute_losses = args.losses
    # ckpt_path = "outputs/tensorboard_logs/version_387/Checkpoint_epoch=594-val_loss_gp=3.175.ckpt"
    gpu = args.gpu
    name = args.name
    device = torch.device("cuda:" + str(gpu) if gpu != -1 else "cpu")
    load_path = ckpt_path[: ckpt_path.index("/Checkpoint")]
    data = load_obj(load_path + "/data.obj")
    data.device = None if int(args.gpu) == -1 else [int(args.gpu)]
    data.accelerator = "cpu" if int(args.gpu) == -1 else "gpu"
    # data.load_dict["load_map"] = data.load_dict["load_model"]
    # save_obj(data, load_path + "/data.obj")

    data_type = data.data_type
    net = Network.load_from_checkpoint(ckpt_path, data=data)
    net.eval()
    net.map.eval()
    net.to(device)

    if nb_layers_eval is not None:
        net.flow.set_nb_layers_eval(nb_layers_eval)

    if compute_losses:
        data_module = DataModule(data, load_path)
        x_val = data_module.val_data.x
        output_dir = load_path + "/figures"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        compute_ot_costs(
            net,
            x_val,
            output_dir=output_dir,
            device=device,
            only_ot_costs=False,
            save_latent_points=True,
        )
    else:
        x_val = None

    if data_type in toy_data_type:
        if x_val is None:
            data_module = DataModule(data, load_path)
            x_val = data_module.val_data.x
        compute_outputs_2d(net, x_val, load_path)
    elif data_type in latent_data_type:
        viz_latent(
            data,
            net,
            device,
            load_path,
            use_vae_dir_for_outputs=use_vae_dir_for_outputs,
            name=name,
        )
    else:
        raise RuntimeError("Unknown data type.")
