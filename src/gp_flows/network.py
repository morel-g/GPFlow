import torch
import pytorch_lightning as pl
from contextlib import ExitStack

from .probability_distribution import ProbabilityDistribution
from ..tool_box import apply_fn_batch, train_map, get_device
from ..models import get_model

import numpy as np
from ..case import Case
from ..precision import eps_precision


class Network(pl.LightningModule):
    def __init__(
        self,
        data,
    ):
        """A general network which can be used to train a NF model or a GP flow.

        Args:
            data: Data object.
        """
        super(Network, self).__init__()
        self.save_hyperparameters()

        self.map, self.flow = get_model(data)
        self.data = data
        self.batch_size = data.batch_size

        self.train_map = self.flow is None
        self.train_gp = not self.train_map

        self.train_gp_on_data = (
            data.train_dict["gp_data_case"] == Case.train_gp_on_data
        )
        self.train_map = train_map(data.train_dict)
        self.eps = eps_precision
        # Control wether the gaussian is the source or the target distribution
        # of map. If map is a normalizing flow then the gaussian
        # is the target distribution
        self.target_gaussian = True
        self.probability_distribution = ProbabilityDistribution(data.dim)

        if self.flow is not None:
            self.flow_params = self.flow.parameters()
        else:
            # Initialize a dumy ResNet for reproducibility.
            self.flow_params = [torch.nn.Parameter(torch.zeros(1))]

        self.sigma = 1
        self.automatic_optimization = False

    def neg_log_likelihood_map(self, x):
        """Compute the negative log likelihood of the map.

        Args:
            x: Positions to compute the neg log likelihood from.

        Returns:
            The negative log likelihood evaluate at x.
        """
        map_x, log_jacobian = self.map(x)

        return (
            -self.probability_distribution.log_prob(map_x).mean()
            - (log_jacobian).mean()
        )

    def loss_gp_flow(self, x):
        """Compute the loss associated with a GP flow.

        Args:
            x: Positions to compute the loss from.

        Returns:
            The OT cost associated with the GP flow.
        """
        with torch.no_grad():
            x_map = self.map(x)[0]
        gp_flow, _ = self.gp_flow(x_map)
        loss = ((x - gp_flow) ** 2).sum(-1).mean()

        return loss

    def training_step(self, batch, batch_idx):
        def closure(opt, loss):
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        x = batch

        map_opt, gp_flow_opt = self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        if self.train_map:
            nll_map = self.neg_log_likelihood_map(x)
            self.log("loss", nll_map, prog_bar=True, rank_zero_only=True)
            closure(map_opt, nll_map)  # loss_map)
            if self.trainer.is_last_batch:
                """
                for param_group in map_opt.param_groups:
                    print("map lr = ", param_group['lr'])
                """
                sch1.step()

        if self.train_gp:
            loss_gp = self.loss_gp_flow(x)
            self.log("loss_gp", loss_gp, prog_bar=True, rank_zero_only=True)

            reg = self.flow.regularization_coef()
            self.log("reg", reg, prog_bar=True, rank_zero_only=True)
            loss_gp = loss_gp + reg
            closure(gp_flow_opt, loss_gp)

            if self.trainer.is_last_batch:
                """
                for param_group in gp_flow_opt.param_groups:
                    print("gp_flow = ", param_group["lr"])
                """

                sch2.step()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        x = batch
        self.log(
            "val_OT_cost ",
            ((x - self.map(x)[0]) ** 2).sum(-1).mean(),
            prog_bar=True,
        )

        if self.train_map:
            nll_map = self.neg_log_likelihood_map(x)
            self.log("val_loss", nll_map, prog_bar=True)

        if self.train_gp:
            loss_gp = self.loss_gp_flow(x)
            self.log("val_loss_gp", loss_gp, prog_bar=True)
            self.log("lagrangian", self.flow.lagrangian)

            if torch.isnan(loss_gp):
                raise RuntimeError(
                    "GP loss is NaN. Stopping the training but the last"
                    + " valid checkpoint is available in the output directory."
                )

    def predict_map(self, x, reverse=False):
        """Apply the map.

        Args:
            x: The positions.
            reverse: Apply the map backward. Defaults to False.

        Returns:
            The values of the map at x.
        """
        if not reverse:
            x, _ = self.map(x)
            return x
        else:
            return self.map.backward(x)

    def gp_flow(self, x, reverse=False, save_trajectories=False):
        """Apply the GP flow.

        Args:
            x: The positions.
            reverse: Apply the GP flow backward. Defaults to False.
            save_trajectories: Save the trajectories (to visualize 2d
            outputs). Defaults to False.

        Returns:
            The values of the GP flow at x.
        """
        if self.flow is None:
            if not reverse:
                return x, torch.zeros_like(x).sum(1)
            else:
                return x
        eps = self.eps
        gp_flow_x = torch.erf(x / (np.sqrt(2.0) * self.sigma))
        gp_flow_x = self.flow(
            gp_flow_x, reverse=reverse, save_trajectories=save_trajectories
        )
        gp_flow_x = (
            np.sqrt(2)
            * self.sigma
            * torch.erfinv(torch.clip(gp_flow_x, -1 + eps, 1 - eps))
        )
        log_abs_det = ((gp_flow_x**2).sum(1) - (x**2).sum(1)) / (
            2.0 * self.sigma**2
        )
        if not reverse:
            return gp_flow_x, log_abs_det
        else:
            return gp_flow_x

    def ot_map(
        self,
        x,
        reverse=False,
        no_grad_map=False,
        no_grad_gp_flow=False,
        return_log_det=False,
    ):
        """Apply the map then the GP flow.

        Args:
            x: The positions.
            reverse: Apply GP flow backward then the map backward. Defaults to False.
            no_grad_map: Do not store gradients associated to the map.
            Defaults to False.
            no_grad_gp_flow: Do not store gradients associated to the GP flow.
            Defaults to False.
            return_log_det: Return also the log determinent of the
            transformation. If not backward tuple is Defaults to True.
        """

        def map_fun(x, reverse):
            if not reverse:
                return self.map(x)
            else:
                return self.map.backward(x), None

        def gp_flow_fun(x, reverse):
            if not reverse:
                return self.gp_flow(x)
            else:
                return self.gp_flow(x, reverse=True), None

        target_gaussian = (
            not self.target_gaussian if reverse else self.target_gaussian
        )
        with ExitStack() as stack:
            if target_gaussian:
                if no_grad_map:
                    stack.enter_context(torch.no_grad())
                map_x, log_map = map_fun(x, reverse=reverse)
                stack.close()
                if no_grad_gp_flow:
                    stack.enter_context(torch.no_grad())
                gp_flow, log_gp = gp_flow_fun(map_x, reverse=reverse)
                stack.close()
            else:
                if no_grad_gp_flow:
                    stack.enter_context(torch.no_grad())
                gp_flow, log_gp = gp_flow_fun(x, reverse=reverse)
                stack.close()
                if no_grad_map:
                    stack.enter_context(torch.no_grad())
                gp_flow, log_map = map_fun(gp_flow, reverse=reverse)
                stack.close()

        if return_log_det:
            log_map += log_gp
            if not reverse:
                return gp_flow, log_map
            else:
                return gp_flow, log_map
        return gp_flow

    def get_flow_trajectories(self):
        return self.flow.get_trajectories()

    def sample(self, nb_samples):
        return self.probability_distribution.sample(nb_samples)

    def reinit_cuda(self, x):
        self.flow.reinit_cuda()

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        def compute_nb_epochs(data):
            if data.train_dict["gp_opt_type"] == Case.train_gp:
                nb_epochs_gp = (
                    data.epochs - data.train_dict["epoch_start_train_gp"]
                )
                nb_epochs_map = data.epochs - nb_epochs_gp
            else:
                nb_epochs_gp = data.epochs
                nb_epochs_map = data.epochs
            return nb_epochs_map, nb_epochs_gp

        def get_step_size(epochs, lr_nb_decay):
            step_size = (
                epochs // (lr_nb_decay + 1)
                if epochs >= (lr_nb_decay + 1)
                else epochs
            )
            return step_size

        nb_epochs_map, nb_epochs_gp = compute_nb_epochs(self.data)
        step_size = get_step_size(
            nb_epochs_map, self.data.opt_dict["nb_decay_map"]
        )

        step_size_gp = get_step_size(
            nb_epochs_gp, self.data.opt_dict["nb_decay_gp"]
        )

        map_opt = torch.optim.Adam(
            self.map.parameters(),
            lr=self.data.opt_dict["lr_map"],
            betas=(0.9, 0.999),
            weight_decay=self.data.weight_decay,
        )
        map_scheduler = torch.optim.lr_scheduler.StepLR(
            map_opt, step_size=step_size, gamma=0.5
        )

        weight_decay_gp = (
            self.data.opt_dict["weight_decay_gp"]
            if ("weight_decay_gp" in self.data.opt_dict)
            else self.data.weight_decay
        )
        gp_flow_opt = torch.optim.Adam(
            self.flow_params,
            lr=self.data.opt_dict["lr_gp_flow"],
            betas=(0.9, 0.999),
            weight_decay=weight_decay_gp,
        )  # self.flow.v.v.denses.parameters(),
        gp_flow_scheduler = torch.optim.lr_scheduler.StepLR(
            gp_flow_opt, step_size=step_size_gp, gamma=0.5
        )

        return (
            {"optimizer": map_opt, "lr_scheduler": map_scheduler},
            {"optimizer": gp_flow_opt, "lr_scheduler": gp_flow_scheduler},
        )
