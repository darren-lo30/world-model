from torch import Tensor, nn, optim
from dataclasses import dataclass
import torch.nn.functional as F
from einops import rearrange
import torch
import lightning as L
import matplotlib.pyplot as plt
import wandb

class DiagonalGaussian(nn.Module):
    def __init__(self, chunk_dim: int = 1):
        super().__init__()
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor, sample: bool) -> tuple[Tensor, Tensor, Tensor]:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if sample:
            std = torch.exp(0.5 * logvar)
            z_sample = mean + std * torch.randn_like(mean)
        else:
            z_sample = mean
        return z_sample, mean, logvar

def visualize_predictions(model, x, x_hat, epoch):
    fig, ax = plt.subplots()
    def denormalize(img):
        img = (img + 1) * 127.5
        img = img.permute(0, 2, 3, 1) # Reshape to B, H, W, C
        img = img.clamp(0, 255).to(dtype=torch.uint8)
        return img
    
    x = denormalize(x).cpu().numpy()
    x_hat = denormalize(x_hat).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i in range(len(x)):
        axes[0].imshow(x[i])
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(x_hat[i])
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        model.logger.experiment.log({f"val/prediction_epoch{epoch}": wandb.Image(fig)})
        plt.close(fig)


class Decoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc = nn.Linear(latent_size, 1024)
        self.encoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 6, stride=2),
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        return F.tanh(self.encoder_conv(x))

class Encoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.fc_mu_logvar = nn.Linear(256 * 4, latent_size * 2)

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        
        return self.fc_mu_logvar(x)


class VAE(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            in_channels,
            latent_channels
        )
        self.decoder = Decoder(
            in_channels, 
            latent_channels
        )
        self.reg = DiagonalGaussian()

    def encode(self, x: Tensor, sample: bool) -> tuple[Tensor, Tensor, Tensor]:
        z_sample, mean, logvar = self.reg(self.encoder(x), sample)
        return z_sample, mean, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def kl_divergence_loss(self, mean: Tensor, logvar: Tensor) -> Tensor:
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl_loss

    def forward(self, x: Tensor, sample: bool) -> tuple[Tensor, Tensor, Tensor]:
        z_sample, mean, logvar = self.encode(x, sample)
        x_recon = self.decode(z_sample)
        return x_recon, mean, logvar

    def compute_loss(self, x_hat, x, mean, logvar):
        # Reconstruction loss
        recon_loss = torch.sum((x_hat - x)**2) / x.shape[0]
        
        # KL divergence loss
        kl_loss = self.kl_divergence_loss(mean, logvar) / x.shape[0]
        
        # Total loss
        loss = recon_loss + kl_loss

        return recon_loss, kl_loss, loss

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        x_hat, mean, logvar = self.forward(x, sample=True)
        
        # Reconstruction loss
        recon_loss, kl_loss, loss = self.compute_loss(x_hat, x, mean, logvar)
        
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        x_hat, mean, logvar = self.forward(x, sample=False)
        
        # Reconstruction loss
        recon_loss, kl_loss, loss = self.compute_loss(x_hat, x, mean, logvar) 
        
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)

        visualize_predictions(self, x, x_hat, self.current_epoch)

        return loss


    def configure_optimizers(self):
        super().configure_optimizers()
