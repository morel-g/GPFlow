import torch
from pathlib import Path

from src.precision import torch_float_precision
from src.training import run_sim
from src.data import Data
from data_helpers.data_type import toy_data_type, latent_data_type
from data_helpers.latent_data import (
    get_latent_data,
    EXP_NAME,
    latent_nf_model_dict,
)
from data_helpers.toy_data import get_toy_data, toy_nf_model_dict
from data_helpers.data_parser import parse_main


def parser_to_data(args):
    """Generate Data obj from parser.

    Args:
        args: parser.

    Returns:
        Data obj.
    """
    data = Data()

    data.n_samples = int(args.nb_samples)
    data.data_type = args.data_type
    data.dim = args.dim
    data.epochs = args.epochs
    data.batch_size = args.batch_size
    data.check_val_every_n_epochs = args.check_val_every_n_epochs

    if args.data_type in toy_data_type:
        data.nf_model_dict = toy_nf_model_dict(args.nf_model)
    elif args.data_type in latent_data_type:
        exp_name = EXP_NAME[data.data_type]
        data.load_dict["latent_data_path"] = (
            "datasets/latent_var/x_latent_" + exp_name + ".npy"
        )
        data.nf_model_dict = latent_nf_model_dict(
            args.nf_model, data.data_type, dim=data.dim
        )
    else:
        raise RuntimeError("Unknown training case.")

    data.nb_layers_gp = args.nb_layers_gp
    data.velocity_dict = {
        "nb_neurons": args.nb_neurons_gp,
        "nb_div_blocks": args.nb_div_blocks_gp,
    }
    data.train_dict.update(
        {"gp_opt_type": args.opt_type, "gp_data_case": args.data_case_gp}
    )
    data.reload_dataloaders_every_n_epochs = (
        args.reload_dataloaders_every_n_epochs
    )
    data.euler_dict = {
        "use_euler": args.use_euler,
        "case": args.euler_case,
        "coef_penalization": args.euler_coef,
        "nb_decay_penalization": args.nb_decay_euler_coef,
    }

    lr = args.learning_rate
    restore_training = args.restore_training
    ckpt_path = args.ckpt_path
    nb_decay_lr = args.nb_decay_lr
    model_path = args.model_path
    data.set_params(
        lr=lr,
        nb_decay=nb_decay_lr,
        model_path=model_path,
        ckpt_path=ckpt_path,
        restore_training=restore_training,
    )
    return data


if __name__ == "__main__":
    torch.set_num_threads(6)
    torch.set_default_dtype(torch_float_precision)

    args = parse_main()
    data_type = args.data_type
    nf_model = args.nf_model
    opt_type = args.opt_type
    model_path = args.model_path
    model_path = None if args.default_model_path else model_path

    Path("outputs").mkdir(parents=True, exist_ok=True)
    if args.default_params:
        if data_type in toy_data_type:
            data = get_toy_data(
                data_type,
                nf_model,
                opt_type,
                model_path,
                use_euler=args.use_euler,
                euler_case=args.euler_case,
            )
        elif data_type in latent_data_type:
            data = get_latent_data(
                data_type,
                nf_model,
                opt_type,
                model_path,
                use_euler=args.use_euler,
                euler_case=args.euler_case,
            )
        else:
            raise RuntimeError("Unknown data type.")
    else:
        data = parser_to_data(args)

    data.device = None if int(args.gpu) == -1 else [int(args.gpu)]
    data.accelerator = "cpu" if int(args.gpu) == -1 else "gpu"

    run_sim(data)
