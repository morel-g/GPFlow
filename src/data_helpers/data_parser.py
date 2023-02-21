import argparse
from src.case import Case
from .data_type import latent_data_type

toy_data_type = [
    Case.eight_gaussians,
    Case.moons,
    Case.pinwheel,
    Case.checkerboard,
]


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=80)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join(action.option_strings) + " " + args_string


def parse_main(case=Case.toy):
    """Parses arguments."""
    fmt = lambda prog: CustomHelpFormatter(prog)
    parser = argparse.ArgumentParser(
        formatter_class=fmt,
        description="Train data with a normalizing flow or GP flow.",
    )

    # General parser
    general = parser.add_argument_group("General options")
    general.add_argument(
        "-dat",
        "--data_type",
        type=str,
        default=Case.eight_gaussians,
        help="Data type",
        choices=toy_data_type + latent_data_type,
    )
    general.add_argument(
        "-o",
        "--opt_type",
        type=str,
        default=Case.train_nf,
        help="Either train a normalizing flow or GP flow on a pre-trained model.",
        choices=[Case.train_nf, Case.train_gp],
    )

    general.add_argument(
        "-ns",
        "--nb_samples",
        type=int,
        default=3e5,
        metavar="",
        help="How many samples in 2d or when training GP on gaussian noise.",
    )

    general.add_argument(
        "--default_params",
        action="store_true",
        default=False,
        help="Restore training from previous checkpoint.",
    )
    general.add_argument(
        "-d",
        "--dim",
        type=int,
        default=2 if case == Case.toy else 10,
        metavar="",
        help="Dimension.",
    )
    general.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1000,
        metavar="",
        help="Number of epochs.",
    )
    general.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1024,
        metavar="",
        help="Batch size.",
    )
    general.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        metavar="",
        help="Learning rate.",
    )
    general.add_argument(
        "--nb_decay_lr",
        type=int,
        default=0,
        metavar="",
        help=(
            "Number of time where the learning rate is periodically divided"
            + "by 2 during training."
        ),
    )
    general.add_argument(
        "--check_val_every_n_epochs",
        type=int,
        default=5,
        metavar="",
        help="Check validation every n epochs.",
    )
    general.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        default="0",
        metavar="",
        help="GPU id used, -1 for CPU.",
    )
    general.add_argument(
        "--restore_training",
        action="store_true",
        default=False,
        help="Restore training from previous checkpoint.",
    )
    general.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Chekpoint path.",
    )
    general.add_argument(
        "--nf_model",
        type=str,
        default=Case.ffjord,
        help="The NF model used.",
        choices=[Case.ffjord, Case.bnaf, Case.cpflow],
    )

    # GP flow parser.
    gp_flow = parser.add_argument_group("GP flow")
    gp_flow.add_argument(
        "--default_model_path",
        action="store_true",
        default=False,
        help="Use default model path outputs/flow_saved_model/data_type/nf_model.",
    )
    gp_flow.add_argument(
        "--model_path",
        type=str,
        default=None,
        metavar="",
        # required="--default_model_path" in sys.argv,
        help="Path to the pre-trained model.",
    )
    gp_flow.add_argument(
        "-nl_gp",
        "--nb_layers_gp",
        type=int,
        default=15,
        metavar="",
        help="How many time steps. Default to 15.",
    )
    gp_flow.add_argument(
        "-nn_gp",
        "--nb_neurons_gp",
        nargs="+",
        type=int,
        default=[15, 15],
        metavar="",
        help="Number of neurons for each hidden layer in the velocity.",
    )
    gp_flow.add_argument(
        "-ndb_gp",
        "--nb_div_blocks_gp",
        type=int,
        default=1,
        metavar="",
        help="Number of div blocks. Should be d-1 to have all the divergence free functions.",
    )
    gp_flow.add_argument(
        "--data_case_gp",
        type=str,
        default=Case.train_gp_on_gaussian_noise,
        help="To train GP flow on data or gaussian nois (only if backward function is available for the model).",
        choices=[Case.train_gp_on_gaussian_noise, Case.train_gp_on_data],
    )
    gp_flow.add_argument(
        "--reload_dataloaders_every_n_epochs",
        type=int,
        default=5,
        metavar="",
        help="Reload data every n rpochs. Only when training GP flow with gaussian noise.",
    )
    gp_flow.add_argument(
        "--use_euler",
        action="store_true",
        default=False,
        help="Use euler.",
    )
    gp_flow.add_argument(
        "--euler_case",
        type=str,
        choices=[Case.penalization, Case.spectral_method],
        default=Case.penalization,
        help="Method to solve Euler's equations (spectral method works only in 2d).",
    )
    gp_flow.add_argument(
        "--euler_coef",
        type=int,
        default=5e-4,
        metavar="",
        help="Coefficient for Euler's penalization.",
    )
    gp_flow.add_argument(
        "--nb_decay_euler_coef",
        type=int,
        default=0,
        metavar="",
        help="Number of times where Euler's coefficient is divided periodically by 2 during the training.",
    )

    return parser.parse_args()


def parse_viz():
    """Parses arguments."""
    fmt = lambda prog: CustomHelpFormatter(prog)
    parser = argparse.ArgumentParser(
        formatter_class=fmt,
        description="Train data with a normalizing flow or GP flow.",
    )

    # General parser
    general = parser.add_argument_group("General options")
    general.add_argument(
        "-c",
        "--ckpt_path",
        type=str,
        default="",
        help="Chekpoint path.",
    )
    general.add_argument(
        "-m",
        "--metric",
        action="store_true",
        default=False,
        help="Compute the disantengled metric (only for dsprites).",
    )
    general.add_argument(
        "--nb_layers_eval",
        type=int,
        default=None,
        metavar="",
        help="Number of layers used for evaluation. Same as during training by default.",
    )
    general.add_argument(
        "-l",
        "--losses",
        action="store_true",
        default=False,
        help="Compute losses.",
    )
    general.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        default=-1,
        metavar="",
        help="GPU id used, -1 for CPU.",
    )
    general.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="Additional name for the plots.",
    )

    return parser.parse_args()
