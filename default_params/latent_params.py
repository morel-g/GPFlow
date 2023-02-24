from src.data import Data
from src.case import Case
from src.seed import set_seed_data


BATCH_SIZE = {
    Case.train_nf: 1024,
    Case.train_gp: 1024,
}

EXP_NAME = {
    Case.dsprites: "btcvae_dsprites",
    Case.mnist: "btcvae_mnist",
    Case.chairs: "btcvae_chairs",
}
EPOCHS = {
    Case.ffjord: {
        Case.train_nf: {
            Case.dsprites: 200,
            Case.mnist: 1600,
            Case.chairs: 1500,
        },
        Case.train_gp: {
            Case.dsprites: 1250,
            Case.mnist: 1250,
            Case.chairs: 1250,
        },
    },
    Case.cpflow: {
        Case.train_nf: {
            Case.dsprites: 700,
            Case.mnist: 7000,
            Case.chairs: 10000,
        },
    },
}

LR = {
    Case.train_nf: {
        Case.dsprites: 1e-3,
        Case.mnist: 1e-3,
        Case.chairs: 1e-3,
    },
    Case.train_gp: {
        Case.dsprites: 5e-4,
        Case.mnist: 5e-4,
        Case.chairs: 5e-4,
    },
}
HIDDEN_DIMS = {
    Case.dsprites: [50] * 3,  # [40, 40],
    Case.mnist: [100] * 4,
    Case.chairs: [100] * 4,
}

KWARGS_NF = {
    Case.ffjord: {
        Case.dsprites: {
            "dim": 2,
            "n_neurons": (64,) * 2,  #
            "bruteforce_eval": True,
            "nblocks": 3,  # 3,
        },
        Case.mnist: {
            "dim": 2,
            "n_neurons": (128,) * 3,
            "bruteforce_eval": True,
            "nblocks": 1,  # 3,
        },
        Case.chairs: {
            "dim": 2,
            "n_neurons": (128,) * 3,
            "bruteforce_eval": True,
            "nblocks": 1,  # 3,
        },
    },
    Case.cpflow: {
        Case.dsprites: {
            "dim": 10,
            "n_neurons": 128,
            "nb_layers": 5,
            "bruteforce_eval": True,
            "nblocks": 1,
            "softplus_type": "softplus",
        },
        Case.mnist: {
            "dim": 10,
            "n_neurons": 128,  #
            "nb_layers": 5,
            "bruteforce_eval": True,
            "nblocks": 1,
            "softplus_type": "softplus",
        },
        Case.chairs: {
            "dim": 10,
            "n_neurons": 128,
            "nb_layers": 5,
            "bruteforce_eval": True,
            "nblocks": 1,
            "softplus_type": "softplus",
        },
        # Case.dsprites: {
        #     "dim": 10,
        #     "n_neurons": 64,  #
        #     "nb_layers": 5,
        #     "bruteforce_eval": True,
        #     "nblocks": 3,  # 3,
        # },
        # Case.mnist: {
        #     "dim": 10,
        #     "n_neurons": 64,  #
        #     "nb_layers": 5,
        #     "bruteforce_eval": True,
        #     "nblocks": 3,  # 3,
        # },
        # Case.chairs: {
        #     "dim": 10,
        #     "n_neurons": 64,
        #     "nb_layers": 5,
        #     "bruteforce_eval": True,
        #     "nblocks": 3,
        # },
    },
    Case.bnaf: {
        Case.dsprites: {
            "dim": 10,
            "n_neurons": 60,
            "nb_layers": 3,
            "nblocks": 3,
        },
        Case.mnist: {
            "dim": 10,
            "n_neurons": 60,
            "nb_layers": 3,
            "nblocks": 3,
        },
    },
}


def latent_nf_model_dict(nf_case, latent_case, **kwargs):
    if nf_case == Case.bnaf:
        nf_model_dict = {
            "name": Case.bnaf,
            "kwargs": KWARGS_NF[Case.bnaf][latent_case],
        }
    elif nf_case == Case.ffjord:
        nf_model_dict = {
            "name": Case.ffjord,
            "kwargs": KWARGS_NF[Case.ffjord][latent_case],
        }
    elif nf_case == Case.cpflow:
        nf_model_dict = {
            "name": Case.cpflow,
            "kwargs": KWARGS_NF[Case.cpflow][latent_case],
        }
    else:
        raise RuntimeError("Unknwon nf case " + nf_case)
    for key, value in kwargs.items():
        nf_model_dict[key] = value
    return nf_model_dict


def get_latent_params(
    data_type,
    nf_model,
    opt_type,
    model_path=None,
    use_euler=False,
    euler_case=Case.penalization,
    ckpt_path=None,
):
    """Return data object filled with default params.

    Args:
        data_type: Data type used (e.g. dsprites, mnist...).
        nf_model: Normailizing flows model used (e.g. FFJORD, CPFlows...)
        opt_type: Optimization type: train_nf or train_gp.
        model_path: Path to the NF pre-trained model. Defaults to None.
        use_euler: For using Euler during the simulation. Default to False.
        euler_case: Euler case (penalization or spectral). Default to
        penalization.

    Returns:
        A data object filled with the default parameters.
    """
    data = Data()
    set_seed_data(data, set_seed=False)

    data.n_samples = 300000
    data.data_type = data_type
    ####################################
    # GP params
    ####################################
    data.nb_layers_gp = 15
    data.velocity_dict = {
        "nb_neurons": HIDDEN_DIMS[data_type],
        "nb_div_blocks": 9,
        # "activ_func": Case.relu,
    }
    data.euler_dict = {
        "use_euler": use_euler,
        "case": euler_case,  # penalization,  #
        "coef_penalization": 5e-5,
        "nb_decay_penalization": 10,
        "batch_size": None,
    }
    ####################################
    # GP training params
    ####################################
    data.train_dict.update(
        {
            "gp_opt_type": opt_type,
            "gp_data_case": Case.train_gp_on_gaussian_noise,  # Case.train_gp_on_data,  #      #   # ,  #
        }
    )
    ####################################
    # General training params
    ####################################

    data.dim = 10
    data.epochs = EPOCHS[nf_model][data.train_dict["gp_opt_type"]][
        data.data_type
    ]

    data.check_val_every_n_epoch = 10
    data.reload_dataloaders_every_n_epochs = 3
    data.batch_size = BATCH_SIZE[data.train_dict["gp_opt_type"]]
    lr = LR[data.train_dict["gp_opt_type"]][data.data_type]
    nb_decay_lr = 0
    data.nb_blocks = 1

    ####################################
    # Loading model params
    ####################################
    restore_training = ckpt_path is not None
    exp_name = EXP_NAME[data.data_type]
    data.load_dict["latent_data_path"] = (
        "datasets/latent_var/x_latent_" + exp_name + ".npy"
    )

    ##############
    data.nf_model_dict = {
        "name": nf_model,
        "kwargs": KWARGS_NF[nf_model][data.data_type],
    }
    # Set params into data
    data.set_params(
        lr=lr,
        nb_decay=nb_decay_lr,
        model_path=model_path,
        ckpt_path=ckpt_path,
        restore_training=restore_training,
    )
    return data
