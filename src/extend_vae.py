import torch
from .case import Case


def slerp(a, b, t):
    a_norm = a / torch.norm(a, dim=-1, keepdim=True)
    b_norm = b / torch.norm(b, dim=-1, keepdim=True)
    omega = torch.acos((a_norm * b_norm).sum(-1))
    tol = 1e-4

    sin_omega = torch.sin(omega)
    if abs(sin_omega) < tol:
        print("Waring sin angle to small use lerp instead.")
        return torch.lerp(a, b, t)
    res = (torch.sin((1.0 - t) * omega) / sin_omega) * a + (
        torch.sin(t * omega) / sin_omega
    ) * b
    return res


class ExtendVAE(torch.nn.Module):
    def __init__(self, vae, network):
        super(ExtendVAE, self).__init__()
        self.vae = vae
        self.network = network
        self.latent_case = Case.default_vae
        self.parameters = vae.parameters
        self.decoder = vae.decoder
        self.reparameterize = vae.reparameterize
        self.training = vae.training
        self.latent_dim = vae.latent_dim
        self.img_size = vae.img_size
        self.training = self.vae.training

    def apply_nf(self, x):
        if self.latent_case == Case.default_vae:
            return x
        if self.latent_case == Case.apply_nf:
            return self.network.predict_map(x)
        if self.latent_case == Case.apply_gp:
            return self.network.ot_map(x)

    def interp(self, samples, idx):
        x1, x2 = samples[0], samples[-1]
        n_interp = samples.shape[0]
        device = x1.device
        y1, y2 = self.apply_nf(x1.unsqueeze(0)).squeeze(0), self.apply_nf(
            x2.unsqueeze(0)
        ).squeeze(0)
        t = torch.linspace(0, 1, n_interp).unsqueeze(1).to(device)
        interp_fun = (
            torch.lerp  # if self.latent_case == Case.default_vae else slerp
        )

        return interp_fun(y1.unsqueeze(0), y2.unsqueeze(0), t).to(device)

    def encoder(self, x):
        mu, logvar = self.vae.encoder(x)
        if self.latent_case == Case.default_vae:
            return mu, logvar
        x = self.reparameterize(mu, logvar)
        if self.latent_case == Case.apply_nf:
            x = self.network.predict_map(x)
        elif self.latent_case == Case.apply_gp:
            x = self.network.ot_map(x)
        return x, torch.zeros_like(x)

    def decoder(self, x):
        if self.latent_case == Case.apply_gp:
            x = self.network.ot_map(x, reverse=True)
        if self.latent_case == Case.apply_nf:
            x = self.network.predict_map(x, reverse=True)
        return self.vae.decoder(x)

    def reparameterize(self, mu, logvar):
        if self.latent_case == Case.default_vae:
            return self.vae.reparameterize(mu, logvar)
        return mu

    def forward(self, x):
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample
