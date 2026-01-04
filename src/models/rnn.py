import torch
import torch.nn as nn
import lightning as L
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent
import matplotlib.pyplot as plt
import wandb

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
    component_dist = Normal(mu, sigma)
    component_dist = Independent(component_dist, 1)
    gmm = MixtureSameFamily(mix, component_dist)

    samples = gmm.sample((n_samples,))
    return samples

def visualize_predictions(model, dist, batch, vae, epoch):
    pi, mu, sigma = dist
    x, _, _, _ = batch 
    vae = vae.to(x.device)

    fig, ax = plt.subplots()
    n_samples = 5
    def denormalize(img):
        img = (img + 1) * 127.5
        img = img.permute(0, 2, 3, 1) # Reshape to B, H, W, C
        img = img.clamp(0, 255).to(dtype=torch.uint8)
        return img

    x_next = x[:, -1, ...]
    x = x[:, -2, ...]
    x = denormalize(x).cpu().numpy()
    x_next = denormalize(x_next).cpu().numpy()

    samples = sample_gmm(pi, mu, sigma, n_samples) # [n_samples, batch, time, latent_dim]
    # Get rid of time dimension
    samples = samples[:, :, -1, :]
    samples_shape = samples.shape
    samples = samples.view((samples_shape[0] * samples_shape[1], -1))
    samples = vae.decode(samples)
    samples = denormalize(samples)
    samples = samples.reshape(n_samples, -1, *samples.shape[1:])

    fig, axes = plt.subplots(1, n_samples + 2, figsize=(12, 4))
    for i in range(0, len(x), len(x) // 16):
        axes[0].imshow(x[i])
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(x_next[i])
        axes[1].set_title("Target")
        axes[1].axis("off")


        for j in range(n_samples):
            axes[j + 2].imshow(samples[j][i].cpu().numpy())
            axes[j + 2].set_title(f"Prediction {j}")
            axes[j + 2].axis("off")

        model.logger.experiment.log({f"val/prediction_epoch{epoch}": wandb.Image(fig)})
        plt.close(fig)

class WorldModelRNN(L.LightningModule):
    def __init__(self, input_size, hidden_size, latent_size, num_gaussians, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.num_layers = 1
        # self.rnn = LSTM(input_size, hidden_size)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_gaussians = num_gaussians

        # MDN heads
        self.fc_pi = nn.Linear(hidden_size, num_gaussians)
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_size)
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_size)

        self.lr = lr


    def forward(self, x, h=None, c=None):
        B, L, _ = x.shape

        if h is None or c is None:
            h0, c0 = self.get_init_state(B)
        else:
            h0, c0 = h, c

        y, (h, c) = self.rnn(x, (h0, c0))


        B, L, D = y.shape

        pi = torch.softmax(self.fc_pi(y), dim=-1)
        mu = self.fc_mu(y).view(B, L, self.num_gaussians, self.latent_size)
        sigma = torch.exp(self.fc_sigma(y)).view(B, L, self.num_gaussians, self.latent_size) + 1e-4

        return pi, mu, sigma, (h, c)

    def get_init_state(self, batch_size):
        device = self.device
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)

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

        visualize_predictions(self, dist, batch, self.trainer.datamodule.vae, self.current_epoch)
        return loss

    def configure_optimizers(self):
        super().configure_optimizers()
