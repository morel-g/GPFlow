from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.gp_flows.gp_flow_callback import GPFlowCallback


def get_checkpoint_callback(callbacks):
    """Dummy function to get the checkpoint callback.

    Args:
        callbacks: A list of callbacks generate with 'get_callbacks()'.

    Returns:
        The checkpoint callback.
    """
    return callbacks[0]


def get_callbacks(log_dir, gp_only, early_stop=False, stopping_threshold=0.0):
    """Generate a list of the callbacks needed for the simulation.

    Args:
        log_dir: Directory of the logger (i.e. where outputs will be saved).
        gp_only: True if only GP flow is trained during the simulation.
        early_stop: If the model should stop training when the validation loss
            reach a certain value. Defaults to False.
        stopping_threshold: The stopping threshold for early_stop.
            Defaults to 0.0.

    Returns:
        A list of callbacks.
    """
    if not gp_only:
        monitor = "val_loss"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=log_dir,
            save_top_k=2,
            mode="min",
            filename="Checkpoint_{epoch}-{val_loss:.3f}",
            every_n_epochs=1,
        )
    else:
        monitor = "val_loss_gp"
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=log_dir,
            save_top_k=2,
            mode="min",
            filename="Checkpoint_{epoch}-{val_loss_gp:.3f}",
            every_n_epochs=1,
        )
    callbacks = [checkpoint_callback, GPFlowCallback()]

    if early_stop:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            stopping_threshold=stopping_threshold,
            patience=1000,
        )
        callbacks.append(early_stop_callback)

    return callbacks
