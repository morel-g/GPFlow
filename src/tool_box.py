import torch
import os
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from .case import Case


def load_pl_model(ckpt_path, model_obj):
    checkpoint = torch.load(ckpt_path)
    model = model_obj.load_from_checkpoint(
        ckpt_path, **checkpoint["hyper_parameters"]
    )
    return model


def get_logger(logger_path, model_name=""):
    logger = TensorBoardLogger(
        logger_path, name=model_name, default_hp_metric=False
    )
    log_dir = logger.log_dir

    isExist = os.path.exists(log_dir)
    if not isExist:
        os.makedirs(log_dir)
    print("log dir: ", log_dir)
    return logger


def weight_decay(w_init, total_epochs, current_epoch, nb_decay):
    coef_decay = (current_epoch * (nb_decay + 1)) // total_epochs
    # coef_decay = nb_decay-1-coef_decay
    return w_init / (2**coef_decay)


def get_device(accelerator, device=[0]):
    if accelerator == "gpu":
        device = "cuda:" + str(device[0])
    else:
        device = "cpu"
    return device


def backward_func(map, use_backward=True):
    return use_backward and callable(getattr(map, "backward", None))


def train_map(train_dict):
    return train_dict["gp_opt_type"] == Case.train_map


def train_gp_only(train_dict):
    return (
        train_dict["gp_opt_type"] == Case.train_gp
        and train_dict["epoch_start_train_gp"] == 0
    )


def apply_fn_batch(fn, x, batch_size, device=None):
    x_batch = torch.split(x, split_size_or_sections=batch_size)
    init = False
    for xi in x_batch:
        if device is not None:
            # xi = xi.to(device)
            out = fn(xi.to(device))
        else:
            out = fn(xi)

        if not init:
            y, init = out, True
        else:
            if isinstance(out, (list, tuple)):
                y = [torch.cat((yi, out_i)) for yi, out_i in zip(y, out)]
            else:
                y = torch.cat((y, out))

    return y


def grad_to_div(d, time_state=Case.continuous_time):
    """
    Matrix to transform a gradient function into a divergence free vector field.
    """

    A = torch.ones(d, d)
    M = torch.triu(A) - torch.tril(A)
    if Case.continuous_time == time_state:
        # We are not interested with the grad with respect to t
        M = torch.cat((M, torch.zeros(1, d)), dim=0)

    return M


def time_dependent_var(x, t, time_state):
    """
    Return the time dependent variable concatenated (x,t) if needed.
    """
    if Case.continuous_time == time_state:
        time = torch.ones(x.shape[0], 1).type_as(x) * t
        x = torch.cat((x, time), 1)

    return x


def id_batch(batch_shape, id_shape):
    Id = torch.eye(id_shape)
    Id = Id.reshape((1, id_shape, id_shape))
    return Id.repeat(batch_shape, 1, 1)


def tanh_deriv(x):
    return torch.ones(x.shape).type_as(x) - (torch.tanh(x) ** 2).type_as(x)


def log_cosh(x):
    return x + torch.log((1 + torch.exp(-2 * x)) / 2.0).type_as(x)
    # return torch.log(torch.cosh(x)).type_as(x)


def log_cosh_deriv(x):
    return torch.tanh(x)


def relu_quadratic(x):
    return (torch.relu(x) ** 2) / 2.0


def relu_deriv(x):
    return (x > 0) * 1


def compute_jacobian(f, x, t=None):
    if t is None:
        Df = torch.autograd.functional.jacobian(f, x)
        Df = torch.transpose(
            torch.transpose(
                torch.diagonal(Df, dim1=2, dim2=0), dim0=1, dim1=2
            ),
            dim0=0,
            dim1=1,
        )
    else:
        Df = torch.autograd.functional.jacobian(f, (x, t))
        Df0 = torch.transpose(
            torch.transpose(
                torch.diagonal(Df[0], dim1=2, dim2=0), dim0=1, dim1=2
            ),
            dim0=0,
            dim1=1,
        )
        Df = torch.cat((Df0, Df[1]), dim=-1)

    return Df


def compute_div(f, x):
    Df = compute_jacobian(f, x)
    return torch.diagonal(Df, dim1=-2, dim2=-1).sum(-1)


def compute_det_jac(
    f,
    x,
):
    jac = compute_jacobian(f, x)
    return torch.det(jac)
