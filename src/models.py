import torch
import os

from .tool_box import get_device

from .case import Case
from .gp_flows.nf_model_tool_box import (
    str_to_class,
    is_ffjord,
    init_nf_model,
)
from .save_load_obj import load_obj
from .gp_flows.res_net import get_gp_flow
from .precision import torch_float_precision
from .tool_box import train_nf


def get_model(data):
    """Initialize the models used during the simulation.

    Args:
        data: a data object containing all the parameters

    Returns:
        The NF model initialized, the GP flow (optional may be None).
    """

    model = nf_model(data)

    if data.nf_model_dict["name"] == Case.cpflow:
        # Need to pass one dummy batch to cpflow else won't load properly
        x = torch.randn(data.batch_size, data.dim)
        device = get_device(data.accelerator, data.device)
        model(x.to(device))

    gp_flow = get_gp_flow(
        data.dim,
        data.velocity_dict,
        data.nb_layers_gp,
        data.nb_blocks,
        data.T_final,
        data.ode_params,
        data.euler_dict,
        data.euler_spectral_velocity_dict,
    )
    map_only = train_nf(data.train_dict)
    # If map_only the gp_flow has been initialized for
    # reproducibility
    gp_flow = None if map_only else gp_flow

    if (
        not callable(getattr(model, "backward", None))
        and gp_flow is not None
        and data.train_dict["gp_data_case"] == Case.train_gp_on_gaussian_noise
    ):
        data.train_dict["gp_data_case"] = Case.train_gp_on_data
        print(
            "The model has no backward method. GP flow will be trained \
                on data."
        )
    return model, gp_flow


def nf_model(data):
    device = get_device(data.accelerator, data.device)
    if not data.load_dict["load_map"]:
        nf_model = init_nf_model(data)
    else:
        if not os.path.isdir(data.load_dict["model_path"]):
            raise RuntimeError(
                "Directory "
                + data.load_dict["model_path"]
                + " does not exists. Specify a valid path to load the model."
            )
        data_load = load_obj(data.load_dict["model_path"] + "/data.obj")
        data.nf_model_dict = data_load.nf_model_dict

        nf_model = str_to_class(data_load.nf_model_dict["name"])(
            **data_load.nf_model_dict["kwargs"]
        )

        nf_model = torch.load(
            data.load_dict["model_path"] + "/torch_nf_model.pt", device
        )

        if torch_float_precision == torch.double:
            nf_model.double()

        if is_ffjord(nf_model):
            nf_model.bruteforce_eval = data.nf_model_dict["kwargs"][
                "bruteforce_eval"
            ]
    return nf_model
