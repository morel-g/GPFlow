from pytorch_lightning.callbacks import Callback
from ..tool_box import weight_decay
from ..save_load_obj import save_obj


class GPFlowCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        pass

    def on_fit_end(self, trainer, pl_module):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        data = pl_module.data
        epoch = pl_module.current_epoch
        if (
            pl_module.flow is not None
            and pl_module.flow.euler_coef_penalization is not None
        ):
            nb_decay = data.euler_dict["nb_decay_penalization"]
            pl_module.flow.euler_coef_penalization = weight_decay(
                data.euler_dict["coef_penalization"],
                data.epochs,
                epoch,
                nb_decay,
            )

    def on_train_start(self, trainer, pl_module):
        pl_module.logger.experiment.add_text(
            "data", pl_module.data.to_string()
        )
        save_obj(pl_module.data, pl_module.logger.log_dir + "/data.obj")

    def on_train_end(self, trainer, pl_module):
        pass
