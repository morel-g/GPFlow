import pytorch_lightning as pl
from src.datasets.prepare_dataset import prepare_dataset
import torch
from .case import Case
from torch.utils.data import DataLoader
from src.dataset import Dataset
from .tool_box import get_device, apply_fn_batch, train_nf
from .gp_flows.probability_distribution import ProbabilityDistribution
from src.tool_box import get_logger


class DataModule(pl.LightningDataModule):
    def __init__(self, data, log_dir=None):
        """Initialize data module.

        Args:
            data: A data object containing all the parameters of the simulation.
            logger: A logger if available to store the outputs. Defaults to None.
        """
        super().__init__()
        self.data = data
        self.device = get_device(data.accelerator, data.device)
        self.batch_size = data.batch_size
        self.pin_memory = False if (self.device == "cpu") else True
        self.train_nf = train_nf(data.train_dict)
        self.train_gp_on_data = (
            data.train_dict["gp_data_case"] == Case.train_gp_on_data
        )
        self.map = None
        self.train_img_data, self.val_img_data = None, None
        self.probability_distribution = ProbabilityDistribution(data.dim)

        (
            self.train_data,
            self.val_data,
        ) = prepare_dataset(self.data, log_dir)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def encode_data(self, auto_encoder):
        """Encode training and validation sets.

        Args:
            auto_encoder: The autoencoder to use for the encoding.

        Returns:
            The encoded training and validation sets.
        """
        self.train_img_data, self.val_img_data = self.train_data, self.val_data

        self.train_labels = self.train_data.dataset.targets[
            self.train_data.indices
        ]
        self.val_labels = self.val_data.dataset.targets[self.val_data.indices]
        eps = 1e-6
        latent_dim = auto_encoder.latent_dim
        device = self.device

        def encode_current_data(images, encode, device):
            x0 = images
            x0 = auto_encoder.encode(x0.to(device))  # encode
            if (
                auto_encoder.center
                and hasattr(auto_encoder, "mu")
                and hasattr(auto_encoder, "std")
            ):
                x0 = (x0 - auto_encoder.mu) / (
                    auto_encoder.std + eps
                )  # normalize
            return torch.cat((encode, x0.cpu()))

        with torch.no_grad():
            auto_encoder.to(device)
            train_encode = torch.zeros(0, latent_dim)
            val_encode = torch.zeros(0, latent_dim)
            train_label, val_label = torch.tensor([]), torch.tensor([])
            train_loader = self.train_dataloader()
            val_loader = self.val_dataloader()
            for datas in train_loader:
                images, l_train = datas
                train_encode = encode_current_data(
                    images, train_encode, device
                )
                train_label = torch.cat((train_label, l_train))
            for datas in val_loader:
                images, l_test = datas
                val_encode = encode_current_data(images, val_encode, device)
                val_label = torch.cat((val_label, l_test))
            self.train_data, self.val_data = (
                Dataset(train_encode),
                Dataset(val_encode),
            )

    def train_dataloader(self):
        """Return the dataloader of the encoded training set.

        Returns:
            Dataloader of the encoded training set.
        """
        train_data = self.get_dataset(train=True)

        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            # shuffle=True,
            num_workers=2,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return the dataloader of the encoded validation set.

        Returns:
            Dataloader of the encoded validation set.
        """
        val_data = self.get_dataset(train=False)
        return DataLoader(
            val_data,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=self.pin_memory,
        )

    def val_img_dataloader(self):
        """Return the dataloader of the amge validation set.

        Returns:
            Dataloader of the image validation set.
        """
        if self.val_img_data is not None:
            return DataLoader(
                self.val_img_data,
                batch_size=self.batch_size,
                num_workers=2,
                pin_memory=self.pin_memory,
            )
        return None

    def test_dataloader(self):
        return self.val_dataloader()

    def get_dataset(self, train=True):
        """Compute and return the dataset used for the simulation

        Args:
            train: True for the training set else validation set. Defaults to True.

        Returns:
            The dataset used for the simulation.
        """
        data = self.data
        if not (
            data.train_dict["gp_opt_type"] not in [Case.train_nf, None]
            and data.train_dict["gp_data_case"] != Case.train_gp_on_data
        ):
            return self.train_data if train else self.val_data
        else:
            # If GP flow is trained on Gaussian points.
            if self.map is None or (
                not callable(getattr(self.map, "backward", None))
            ):
                raise RuntimeError(
                    "If train_gp_on_data is set to False the map should have a"
                    + " function backward defined."
                )
            # Create the Gaussian data points and apply the map.backward
            # function to it.
            with torch.no_grad():
                len_train = int(
                    0.8 * data.n_samples if train else 0.2 * data.n_samples
                )
                prob_dist = self.probability_distribution.sample([len_train])
                self.map.to(self.device)

                prob_dist = apply_fn_batch(
                    self.map.backward,
                    prob_dist.to(self.device),
                    min(len_train // 10, 2000),
                )
                dataset = Dataset(prob_dist.cpu())

            return dataset

    def set_map(self, map):
        self.map = map
