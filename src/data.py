from .case import Case
import os
import errno
import sys
import json

from data_helpers.data_type import latent_data_type

class NoBracketJsonEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = 8

    def encode(self, o):
        return super().encode(o).replace("{", "").replace("}", "")


def dict_to_str(dict):
    return json.dumps(dict, cls=NoBracketJsonEncoder).rstrip()


class Data:
    def __init__(self):
        """General data object to store parameters."""
        self.n_samples = 2000
        self.dim = 2

        self.data_type = Case.moons
        self.save_output = True
        self.batch_size = 300
        self.epochs = 1
        self.nb_layers_gp = 20
        self.nb_blocks = 1
        self.opt_dict = {"lr_map": 5e-3, "nb_decay_map": 0}
        self.accelerator = "cpu"
        self.device = 0

        self.velocity_dict = {}
        self.euler_dict = {
            "use_euler": False,
            "case": Case.penalization,
            "coef_penalization": 1.0,
            "nb_decay_penalization": 0,
        }
        self.euler_spectral_velocity_dict = {
            "coef_scheme": {"alpha": 0.5, "beta": 0.0, "gamma": 0.0},
            "order_scheme": 1,
            "poly_case": Case.incompressible_poly,
            "order_poly": 9,
            "boundary_dict": {
                "case": Case.rectangle,
                "bounds": ([-1.0, 1.0], [-1.0, 1.0]),
            },
            "euler_only": False,
        }
        self.load_dict = {
            "load_map": False,
            "model_path": "",
            "restore_training": False,
            "training_ckpt_path": "",
        }
        self.early_stop_dict = {"early_stop": False, "stopping_threshold": 0.0}
        self.train_dict = {
            "gp_opt_type": Case.train_map,
            "epoch_start_train_gp": 0,
        }
        self.nf_model_dict = {"name": None, "kwargs": None}
        #Default ODE params.
        self.ode_params = {
            "method": Case.RK4,
            # Theta for Euler (0 = explicit, 0.5 = mid point, 1= implicit)
            "theta": 0.0,
            # The rest of the parameters is used only if the scheme is not
            # fully explicit (if the method is not RK4 or not Euler with theta=0.)
            "solver": Case.fix_point,
            "tol": 1e-5,
            "nb_it_max": int(1e2),
            # Only standard available for now
            "solve_method": Case.standard,
        }

        self.T_final = 1.0
        self.paths = {}
        self.logger_path = "outputs/tensorboard_logs"

        self.use_backward = True
        self.print_opt = True
        self.weight_decay = 1e-4
        self.check_val_every_n_epoch = 1
        self.reload_dataloaders_every_n_epochs = 0
        self.set_seed = False
        self.seed = 123
        self.model_name = ""

    def to_string(self):
        """Print parameters.

        Returns:
            A string of the parameters to be printed.
        """
        s = ""
        s += "-" * 20 + "  \n  "
        s += "Data values" + "  \n  "
        s += "-" * 20 + "  \n  "
        s += (
            "- n_samples                           = "
            + str(self.n_samples)
            + "  \n  "
        )
        s += (
            "- data_type                           = "
            + str(self.data_type)
            + "  \n  "
        )
        if not self.train_dict["gp_opt_type"] == Case.train_map:
            s += (
                "- velocity_dict                       = "
                + dict_to_str(self.velocity_dict)
                + "  \n  "
            )
        if self.euler_dict["use_euler"]:
            s += (
                "- euler_dict                          = "
                + dict_to_str(self.euler_dict)
                + "  \n  "
            )
            if self.euler_dict["case"] == Case.spectral_method:
                s += (
                    "- euler_spectral_velocity_dict                          ="
                    + " "
                    + dict_to_str(self.euler_spectral_velocity_dict)
                    + "  \n  "
                )
        else:
            s += (
                "- euler_dict                          = "
                + dict_to_str({"use_euler": self.euler_dict["use_euler"]})
                + "  \n  "
            )
        s += (
            "- train_dict                          = "
            + dict_to_str(self.train_dict)
            + "  \n  "
        )
        s += (
            "- nf_model_dict                       = "
            + dict_to_str(self.nf_model_dict)
            + "  \n  "
        )

        load_dict_print = self.load_dict.copy()
        if not load_dict_print["load_map"]:
            load_dict_print.pop("model_path")
        if not load_dict_print["restore_training"]:
            load_dict_print.pop("training_ckpt_path")
        if self.data_type not in latent_data_type:
            if "latent_data_path" in load_dict_print:
                load_dict_print.pop("latent_data_path")
        s += (
            "- load_dict                           = "
            + dict_to_str(load_dict_print)
            + "  \n  "
        )
        s += (
            "- early_stop_dict                     = "
            + dict_to_str(self.early_stop_dict)
            + "  \n  "
        )
        if self.ode_params["method"] == Case.RK4:
            s += (
                "- ode_params                          = "
                + dict_to_str({"method": Case.RK4})
                + "  \n  "
            )
        else:
            s += (
                "- ode_params                          = "
                + dict_to_str(self.ode_params)
                + "  \n  "
            )
        opt_dict_print = self.opt_dict.copy()
        if self.train_dict["gp_opt_type"] == Case.train_map:
            opt_dict_print.pop("lr_gp_flow")
            opt_dict_print.pop("nb_decay_gp")
        else:
            opt_dict_print.pop("lr_map")
            opt_dict_print.pop("nb_decay_map")
        s += (
            "- opt_dict                            = "
            + dict_to_str(opt_dict_print)
            + "  \n  "
        )
        if self.train_dict["gp_opt_type"] == Case.train_gp:
            s += (
                "- nb_layers_gp                        = "
                + str(self.nb_layers_gp)
                + "  \n  "
            )
            s += (
                "- nb_blocks                           = "
                + str(self.nb_blocks)
                + "  \n  "
            )
        s += (
            "- epochs                              = "
            + str(self.epochs)
            + "  \n  "
        )
        s += (
            "- batch_size                          = "
            + str(self.batch_size)
            + "  \n  "
        )
        s += (
            "- use_backward                        = "
            + str(self.use_backward)
            + "  \n  "
        )
        s += (
            "- T_final                             = "
            + str(self.T_final)
            + "  \n  "
        )
        s += (
            "- accelerator                         = "
            + str(self.accelerator)
            + "  \n  "
        )
        if self.accelerator == "gpu":
            s += (
                "- device                              = "
                + str(self.device)
                + "  \n  "
            )
        s += (
            "- weight_decay                        = "
            + str(self.weight_decay)
            + "  \n  "
        )
        s += (
            "- set_seed                            = "
            + str(self.set_seed)
            + "  \n  "
        )
        s += (
            "- seed                                = "
            + str(self.seed)
            + "  \n  "
        )
        s += (
            "- check_val_every_n_epoch             = "
            + str(self.check_val_every_n_epoch)
            + "  \n  "
        )
        s += (
            "- reload_dataloaders_every_n_epochs   = "
            + str(self.reload_dataloaders_every_n_epochs)
            + "  \n  "
        )

        s += "-" * 20 + "  \n  "
        s += "-" * 20 + "  \n  "

        return s

    def print(self, f=sys.stdout):
        """Print the data.

        Args:
            f: Where to print data. Defaults to sys.stdout.
        """
        print(self.to_string(), file=f)

    def write(self, name, print=True):
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        f = open(name, "w")
        self.print(f)
        if print:
            self.print(None)
        f.close()

    def set_params(
        self,
        lr=None,
        nb_decay=None,
        load_map=None,
        model_path=None,
        ckpt_path=None,
        restore_training=None,
    ):
        """A function to set easily the data parameters

        Args:
            lr: The learning rate. Defaults to None.
            nb_decay: Number of decay of the learning rate. Defaults to None.
            load_map: True if some model is lodaed. Defaults to None.
            model_path: Path to the model. Defaults to None.
            ckpt_path: Path to some checkpoint (only used if 'restore_training'
                is set to True). Defaults to None.
            restore_training: If the training is restored from a previous
                simulation. Defaults to None.
        """

        if lr is not None:
            self.opt_dict.update({"lr_map": lr, "lr_gp_flow": lr})
        if nb_decay is not None:
            self.opt_dict.update(
                {
                    "nb_decay_map": nb_decay,
                    "nb_decay_gp": nb_decay,
                }
            )
        if load_map is not None:
            self.load_dict["load_map"] = load_map
        else:
            self.load_dict["load_map"] = (
                self.train_dict["gp_opt_type"] == Case.train_gp
            )
        if model_path is not None:
            self.load_dict["model_path"] = model_path
        if ckpt_path is not None:
            self.load_dict["training_ckpt_path"] = ckpt_path
        if restore_training is not None:
            self.load_dict["restore_training"] = restore_training
