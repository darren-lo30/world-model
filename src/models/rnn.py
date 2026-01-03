import torch
import torch.nn as nn
import lightning as L
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        in_dim = input_size + hidden_size

        self.forget_gate = nn.Linear(in_dim, hidden_size)
        self.input_gate = nn.Linear(in_dim, hidden_size)
        self.output_gate = nn.Linear(in_dim, hidden_size)
        self.cell_gate = nn.Linear(in_dim, hidden_size)

    def forward(self, x, h=None, c=None):
        B, L, D = x.shape

        if h is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)

        if c is None:
            c = torch.zeros(B, self.hidden_size, device=x.device)

        outputs = []

        for t in range(L):
            x_t = x[:, t, :]                   # [B, input]
            xh = torch.cat([x_t, h], dim=-1)   # [B, input+hidden]

            f = torch.sigmoid(self.forget_gate(xh))
            i = torch.sigmoid(self.input_gate(xh))
            o = torch.sigmoid(self.output_gate(xh))
            g = torch.tanh(self.cell_gate(xh))

            c = f * c + i * g
            h = o * torch.tanh(c)

            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1), (h, c)

def sample_gmm(pi, mu, sigma, n_samples):
    mix = Categorical(probs=pi)
    comp = Normal(loc=mu, scale=sigma)
    component_dist = Normal(mu, sigma)
    component_dist = Independent(component_dist, 1)
    gmm = MixtureSameFamily(mix, component_dist)

    samples = gmm.sample((n_samples,))
    return samples

def visualize_predictions(dist, batch, vae):
    pi, mu, sigma = dist
    x, _, _, _ = batch

    fig, ax = plt.subplots()
    n_samples = 5
    def denormalize(img):
        img = (img + 1) * 127.5
        img = img.permute(0, 2, 3, 1) # Reshape to B, H, W, C
        img = img.clamp(0, 255).to(dtype=torch.uint8)
        return img
    
    print(x.shape)
    x = x[:, 0, ...]
    x = denormalize(x).cpu().numpy()

    print(pi.shape, mu.shape, sigma.shape)
    samples = sample_gmm(pi, mu, sigma, n_samples)
    print(samples.shape)
    samples = samples[:, 0, ...]
    print(samples.shape)
    samples = vae.decode(samples)
    samples = denormalize(samples)


    fig, axes = plt.subplots(1, n_samples + 1, figsize=(8, 4))
    for i in range(0, len(x), len(x) // 50):
        axes[0].imshow(x[i])
        axes[0].set_title("Original")
        axes[0].axis("off")

        for j in range(n_samples):
            axes[j + 1].imshow(samples[j])
            axes[j + 1].set_title(f"Prediction {j}")
            axes[j + 1].axis("off")

        model.logger.experiment.log({f"val/prediction_epoch{epoch}": wandb.Image(fig)})
        plt.close(fig)

class WorldModelRNN(L.LightningModule):
    def __init__(self, input_size, hidden_size, latent_size, num_gaussians, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.rnn = LSTM(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_gaussians = num_gaussians

        # MDN heads
        self.fc_pi = nn.Linear(hidden_size, num_gaussians)
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_size)
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_size)

        self.lr = lr

    def forward(self, x, h=None, c=None):
        y, (h, c) = self.rnn(x, h, c)

        B, L, D = y.shape

        pi = torch.softmax(self.fc_pi(y), dim=-1)
        mu = self.fc_mu(y).view(B, L, self.num_gaussians, self.latent_size)
        sigma = torch.exp(self.fc_sigma(y)).view(B, L, self.num_gaussians, self.latent_size)

        return pi, mu, sigma, (h, c)

    def get_init_state(self, batch_size):
        h = torch.zeros((batch_size, self.hidden_size))
        c = torch.zeros((batch_size, self.hidden_size))

        return h, c

    def mdn_loss(self, z_target, pi, mu, sigma):
        # z_target: [B,L,latent]
        z = z_target.unsqueeze(2)  # [B,L,1,latent]

        m = torch.distributions.Normal(mu, sigma)
        log_probs = m.log_prob(z).sum(-1)  # [B,L,components]

        weighted = torch.log(pi + 1e-8) + log_probs
        log_sum = torch.logsumexp(weighted, dim=-1)

        return -log_sum.mean()

    def compute_loss(self, batch):
        _, actions, mean, logvar = batch
        latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        x = torch.cat([latents[:, :-1], actions[:, :-1]], dim=-1)
        target_latents = latents[:, 1:]

        pi, mu, sigma, _ = self.forward(x)

        return self.mdn_loss(target_latents, pi, mu, sigma), (pi, mu, sigma)

    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dist = self.compute_loss(batch)
        self.log("val_loss", loss)

        visualize_predictions(dist, batch, self.trainer.datamodule.vae)
        return loss

    def configure_optimizers(self):
        super().configure_optimizers()
