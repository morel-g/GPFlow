import torch
from datetime import datetime
import pytorch_lightning as pl

from src.case import Case
from src.plots.helper_plot_2d import compute_outputs_2d
from src.plots.outputs import compute_ot_costs
from src.gp_flows.network import Network
from src.save_load_obj import save_obj
from src.data_module import DataModule
from src.tool_box import (
    get_logger,
    get_device,
    train_gp_only,
)
from .callbacks import get_callbacks, get_checkpoint_callback
from .gp_flows.nf_model_tool_box import (
    is_ffjord,
    is_cpflow,
)
from pathlib import Path


def set_dim_key(data):
    """Set dimension in data.nf_model_dict.

    Args:
        data: Data object.
    """
    if "dim" in data.nf_model_dict.get("kwargs", []):
        data.nf_model_dict["kwargs"]["dim"] = data.dim
    if "dims" in data.nf_model_dict.get("kwargs", []):
        data.nf_model_dict["kwargs"]["dims"] = (data.dim,)


def save_outputs(data, net, x_val, logger, device):
    """Save outputs after training the network.

    Args:
        data: Data object of the simulation.
        net: The trained network.
        x_val: Validation set.
        logger: Logger of the simulation.
        device: The device used.
    """
    nf_model, gp_flow = net.map, net.flow
    net = net.to(device)
    save_obj(data, logger.log_dir + "/data.obj")
    # save_obj(net.map.cpu(), logger.log_dir + "/nf_model.obj")
    torch.save(net.map.cpu(), logger.log_dir + "/torch_nf_model.pt")
    torch.save(
        net.map.cpu().state_dict(), logger.log_dir + "/nf_model_state.pt"
    )
    if net.flow is not None:
        torch.save(net.flow.cpu().state_dict(), logger.log_dir + "/gp_flow.pt")
    nf_model.to(device)
    if gp_flow is not None:
        gp_flow.to(device)

    if is_ffjord(nf_model) or is_cpflow(nf_model):
        print("Setting bruteforce eval. Testing time may be quiet long.")
        net.map.bruteforce_eval = True
    net.map.eval()
    net.eval()

    if data.save_output:
        startTime = datetime.now()

        compute_ot_costs(
            net, x_val, logger, device=device, only_ot_costs=False
        )
        print("Computation time OT costs = ", datetime.now() - startTime)
        x_val = x_val.to(device)
        if data.dim == 2:
            compute_outputs_2d(net, x_val, logger.log_dir)


def train_nf_model(
    data,
    data_module,
    logger=None,
):
    """Main training function.

    Args:
        data: Data object
        data_module: Data module.
        logger: Logger used to save the results. Defaults to None.

    Returns:
        The trained network.
    """

    Path(data.logger_path).mkdir(parents=True, exist_ok=True)
    if logger is None:
        logger = get_logger(data.logger_path)
    log_dir = logger.log_dir
    set_dim_key(data)
    device = get_device(data.accelerator, data.device)

    startTime = datetime.now()
    if not data.load_dict["restore_training"]:
        net = Network(data)
    else:
        net = Network.load_from_checkpoint(
            data.load_dict["training_ckpt_path"], data=data
        )
    data_module.set_map(net.map)

    if data.print_opt:
        print("Writing data to " + log_dir + "/data.txt")
        data.write(name=log_dir + "/data.txt")

    # Get callbacks
    gp_only, early_stop, stopping_threshold = (
        train_gp_only(data.train_dict),
        data.early_stop_dict.get("early_stop", False),
        data.early_stop_dict.get("stopping_threshold", 0.0),
    )
    callbacks = get_callbacks(
        logger.log_dir, gp_only, early_stop, stopping_threshold
    )
    checkpoint_callback = get_checkpoint_callback(callbacks)
    data_device = data.device if data.accelerator == "gpu" else "auto"

    trainer = pl.Trainer(
        max_epochs=data.epochs,
        accelerator=data.accelerator,
        devices=data_device,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=data.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=(
            data.reload_dataloaders_every_n_epochs
        ),
    )

    if not data.load_dict["restore_training"]:
        trainer.fit(
            net,
            datamodule=data_module,
        )
    elif data.epochs > 0:
        trainer.fit(
            net,
            ckpt_path=data.load_dict["training_ckpt_path"],
            datamodule=data_module,
        )
    else:
        data_module.setup()

    if checkpoint_callback.best_model_path != "" and not (
        data.nf_model_dict["name"] is Case.cpflow
        or (
            (
                train_gp_only(data.train_dict)
                and (
                    data.euler_dict["use_euler"]
                    and data.euler_dict["case"] == Case.spectral_method
                )
            )
        )
    ):
        print("Loading checkpoint " + checkpoint_callback.best_model_path)
        net = Network.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            data=data,
        )

    if data.print_opt:
        print("Execution time = ", datetime.now() - startTime)

    x_val = data_module.val_data.x
    save_outputs(data, net, x_val, logger, device)

    return net


def run_sim(data):
    """Run the simulation.

    Args:
        data: Data object.
    """
    logger = get_logger(data.logger_path)
    data_module = DataModule(data, logger.log_dir)
    train_nf_model(data, data_module, logger)
