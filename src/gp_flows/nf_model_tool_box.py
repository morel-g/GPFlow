import sys
import torch

from ..case import Case
from ..precision import torch_float_precision
from ..tool_box import get_device
from flow_models.bnaf.bnaf import BNAF as BNAF
from flow_models.ffjord.ffjord import FFJORD
from flow_models.cp_flows.cp_flow import CPFlow


def str_to_class(name):
    if name == Case.cpflow:
        class_name = "CPFlow"
    elif name == Case.ffjord:
        class_name = "FFJORD"
    elif name == Case.bnaf:
        class_name = "BNAF"
    else:
        class_name = name
    return getattr(sys.modules[__name__], class_name)


def is_ffjord(nf_model):
    return isinstance(nf_model, FFJORD)


def is_cpflow(nf_model):
    return isinstance(nf_model, CPFlow)


def init_nf_model(data):
    nf_model = str_to_class(data.nf_model_dict["name"])(
        **data.nf_model_dict["kwargs"]
    )
    if torch_float_precision == torch.double:
        nf_model.double()

    nf_model.to(get_device(data.accelerator, data.device))
    # if data.nf_model_dict["name"] in [Case.cpflow, "IAF", "NAF"]:
    # with torch.no_grad():
    #     nf_model(x_test[: data.batch_size])
    return nf_model
