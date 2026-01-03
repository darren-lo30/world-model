from lightning.pytorch.cli import LightningCLI
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from src.data.dataset import GymDataModule
from src.models.rnn import WorldModelRNN
from jsonargparse import lazy_instance
import os
from datetime import datetime
from src.models.vae import VAE
 
def get_checkpoint(checkpoint_dir):
    from lightning.pytorch.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=10,
        filename="{epoch:02d}",
    )

def train_rnn(results_dir, resume_from_checkpoint=None):
    experiment_dir = os.path.join(results_dir, 'rnn', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    checkpoint_callback = get_checkpoint(experiment_dir)

    cli = LightningCLI(WorldModelRNN, GymDataModule, run=False, 
        trainer_defaults={
            "logger": lazy_instance(WandbLogger, log_model="all", save_dir=experiment_dir),
            "callbacks": [checkpoint_callback],
            "default_root_dir": experiment_dir,
        },
    )

    rnn = cli.model
    dm = cli.datamodule
    trainer = cli.trainer
    
    if resume_from_checkpoint is not None:
        trainer.fit(
            model=rnn,
            datamodule=dm,
            ckpt_path=resume_from_checkpoint,
        )
    else:
        trainer.fit(rnn, datamodule=dm)

train_rnn('./results')