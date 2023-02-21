from src.data import Data
from src.case import Case
from src.seed import set_seed_data

EPOCHS = {Case.eight_gaussians: 2000, Case.moons: 1000}
LR = {
    Case.eight_gaussians: 2e-3,
    Case.moons: 5e-3,
    Case.checkerboard: 5e-3,
    Case.pinwheel: 2e-3,
}
NB_NEURONS = {
    Case.eight_gaussians: [15] * 2,
    Case.moons: [15] * 2,
    Case.checkerboard: [30] * 2,
    Case.pinwheel: [15] * 2,
}


def toy_nf_model_dict(nf_case, **kwargs):
    if nf_case == Case.bnaf:
        nf_model_dict = {
            "name": Case.bnaf,
            "kwargs": {
                "dim": 2,
                "n_neurons": 32,
                "nb_layers": 3,
                "nblocks": 1,
            },
        }
    elif nf_case == Case.ffjord:
        nf_model_dict = {
            "name": Case.ffjord,
            "kwargs": {
                "dim": 2,
                "n_neurons": (32, 32),
                "bruteforce_eval": True,
                "nblocks": 3,
            },
        }
    elif nf_case == Case.cpflow:
        nf_model_dict = {
            "name": Case.cpflow,
            "kwargs": {
                "dim": 2,
                "n_neurons": 32,
                "nb_layers": 3,
                "nblocks": 5,
                "softplus_type": "gaussian_softplus",  # "gaussian_softplus"
            },
        }
    else:
        raise RuntimeError("Unknwon nf case " + nf_case)

    for key, value in kwargs.items():
        nf_model_dict[key] = value
    return nf_model_dict


def get_toy_params(
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
        data_type: Data type used (e.g. eight_gaussians, moons...).
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

    data = Data()
    set_seed_data(data, set_seed=False)

    data.data_type = data_type

    ####################################
    # GP params
    ####################################
    data.nb_layers_gp = 15
    data.velocity_dict = {
        "nb_neurons": NB_NEURONS[data_type]
        if data_type in NB_NEURONS
        else [15] * 2
    }
    data.euler_dict = {
        "use_euler": use_euler,
        "case": euler_case,
        "coef_penalization": 5e-4,
        "nb_decay_penalization": 5,
    }
    ####################################
    # GP training params
    ####################################
    data.train_dict.update(
        {
            "gp_opt_type": opt_type,
            "gp_data_case": Case.train_gp_on_gaussian_noise,
        }
    )
    ####################################
    # General training params
    ####################################
    restore_training = ckpt_path is not None
    data.n_samples = 100000
    data.epochs = EPOCHS[data_type] if data_type in EPOCHS else 1000
    data.check_val_every_n_epoch = 10
    data.batch_size = 1024
    lr = LR[data_type] if data_type in LR else 2e-3
    nb_decay = 0

    ##############
    data.nf_model_dict = toy_nf_model_dict(nf_model)

    # Set params into data
    data.set_params(
        lr=lr,
        nb_decay=nb_decay,
        model_path=model_path,
        ckpt_path=ckpt_path,
        restore_training=restore_training,
    )

    return data
