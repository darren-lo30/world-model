from torch import Tensor, nn, optim
from dataclasses import dataclass
import torch.nn.functional as F
from einops import rearrange
import torch
import lightning as L
import matplotlib.pyplot as plt
import wandb
# This architecture is taken from FLUX including some of the Encoder/Decoder code


def swish(x: Tensor) -> Tensor:
    return x * F.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, affine=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=in_channels, affine=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x) -> Tensor:
        y = x
        y = self.conv1(swish(self.norm1(y)))
        y = self.conv2(swish(self.norm2(y)))

        if self.in_channels != self.out_channels:
            x = self.skip(x)

        return x + y


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.proj_qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        q, k, v = self.proj_qkv(x).chunk(3, dim = 1)

        # We don't add positional encodings here (for some reason)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        x = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(x, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))
    

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        ch_mult: list[int],
        num_res_blocks: int,
        latent_channels: int,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = self.hidden_channels
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = hidden_channels * (ch_mult[i_level - 1] if i_level > 0 else 1)
            block_out = hidden_channels * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)

            # Do not downsample the last resolution
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        latent_channels: int,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = hidden_channels * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = hidden_channels * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean

def visualize_predictions(model, x, x_hat):
    fig, ax = plt.subplots()
    def denormalize(img):
        img = (img + 1) * 127.5
        img = img.permute(0, 2, 3, 1) # Reshape to B, H, W, C
        img = img.clamp(0, 255).to(dtype=torch.uint8)
        return img
    
    x = denormalize(x).cpu().numpy()
    x_hat = denormalize(x_hat).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x[0])
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(x_hat[0])
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    model.logger.experiment.log({"val/prediction": wandb.Image(fig)})
    plt.close(fig)

class VAE(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 32,
        hidden_channels: int = 32,
        ch_mult: list[int] = [2, 2, 2],
        sample_latent: bool = False,
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )
        self.reg = DiagonalGaussian(sample=sample_latent)

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        return z

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
    
    def training_step(self, batch, batch_idx):
        x = batch
        z = self.reg(self.encoder(x))
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        val_loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

        visualize_predictions(self, x, x_hat)

        return val_loss


    def configure_optimizers(self):
        return super().configure_optimizers()
