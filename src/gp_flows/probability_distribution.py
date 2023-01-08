import torch
import torch.distributions as D


class ProbabilityDistribution:
    def __init__(self, dim, cov=1):
        self.dim = dim
        self.cov = cov
        self.distrib = D.multivariate_normal.MultivariateNormal(
            torch.zeros(dim), cov * torch.eye(dim)
        )
        self.distrib_1d = D.normal.Normal(0.0, cov)
        self.init_device = False

    def init_mixture_device(self, x):
        dim = self.dim
        cov = self.cov
        self.distrib = D.multivariate_normal.MultivariateNormal(
            torch.zeros(dim).type_as(x), cov * torch.eye(dim).type_as(x)
        )
        """
        self.distrib.loc = self.distrib.loc.type_as(x)
        self.distrib.covariance_matrix
        = self.distrib.covariance_matrix.type_as(x)
        self.distrib.scale_tril = self.distrib.scale_tril.type_as(x)
        """

    def log_prob(self, x):
        if not self.init_device:
            self.init_mixture_device(x)
            self.init_device = True
        return self.distrib.log_prob(x)

    def log_prob_1d(self, x):
        return self.distrib_1d.log_prob(x)

    def sample(self, shape):
        return self.distrib.sample(shape)
