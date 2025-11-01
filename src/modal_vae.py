import modal
from typing import Optional
from pathlib import Path

from src.train.train_vae import train_vae


volume = modal.Volume.from_name("vae-train", create_if_missing=True)
image = modal.Image.debian_slim().uv_sync()
app = modal.App("vae-train", image=image)

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
    if experiment is None:
        from uuid import uuid4

        experiment = uuid4().hex[:8]
    train_interruptible.spawn(experiment).get()
