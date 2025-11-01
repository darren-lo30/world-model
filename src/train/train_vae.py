from lightning.pytorch.cli import LightningCLI
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from src.data.dataset import GymDataModule
from src.models.vae import VAE

def get_checkpoint(checkpoint_dir):
    from lightning.pytorch.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=10,
        filename="{epoch:02d}",
    )

def train_vae(checkpoint_dir, resume_from_checkpoint=None):
    logger = WandbLogger(log_model="all")

    cli = LightningCLI(VAE, GymDataModule, run=False)

    autoencoder = cli.model
    dm = cli.datamodule

    checkpoint_callback = get_checkpoint(checkpoint_dir)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=20, logger=logger, callbacks=[checkpoint_callback])
    
    if resume_from_checkpoint is not None:
        trainer.fit(
            model=autoencoder,
            datamodule=dm,
            ckpt_path=resume_from_checkpoint,
        )
    else:
        trainer.fit(autoencoder, datamodule=dm)
