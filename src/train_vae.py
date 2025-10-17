from typing import Optional
from lightning.pytorch.cli import LightningCLI
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

from src.data.dataset import GymDataModule, GymObservationDataset
from src.models.vae import VAE
import modal
from pathlib import Path

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


volume = modal.Volume.from_name("example-long-training", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "lightning~=2.4.0", "torch~=2.4.0", "torchvision==0.19.0"
)
app = modal.App("example-long-training", image=image)

volume_path = Path("/experiments")
CHECKPOINTS_PATH = volume_path / "checkpoints"

volumes = {volume_path: volume}

def train(experiment):
    experiment_dir = CHECKPOINTS_PATH / experiment
    last_checkpoint = experiment_dir / "last.ckpt"

    if last_checkpoint.exists():
        print(f"Resuming training from the latest checkpoint: {last_checkpoint}")
        train_vae(
            experiment_dir,
            resume_from_checkpoint=last_checkpoint,
        )
    else:
        train_vae(experiment_dir)

retries = modal.Retries(initial_delay=0.0, max_retries=1)
timeout = 30  


@app.function(
    volumes=volumes, gpu="a10g", timeout=timeout, retries=retries, max_inputs=1
)
def train_interruptible(*args, **kwargs):
    train(*args, **kwargs)

@app.local_entrypoint()
def main(experiment: Optional[str] = None):
    print("HI")
    if experiment is None:
        from uuid import uuid4

        experiment = uuid4().hex[:8]
    # train_interruptible.spawn(experiment).get()
    train_interruptible.local(experiment)

train_vae('./results')