from git import Optional
import gymnasium as gym
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import lightning as L
from torch.utils.data.dataloader import DataLoader
import os
from torchvision import transforms
from PIL import Image
from src.models.vae import VAE
import torch.nn.functional as F

# Collects observations for the VAE to encoder
# This uses random rollout to collect observations
# Hopefully this is general enough for a lot of envs
class GymObservationDataset(Dataset):
    def __init__(self, env_name: str, n_samples: int = 500, cache_path: Optional[str] = None):
        self.episode_data = []
        if cache_path is not None and os.path.exists(cache_path):
            print(f"Loading cached dataset: {cache_path}")
            self.episode_data = torch.load(cache_path, weights_only=False)
        else:
            self.episode_data = self._gen_data(env_name, n_samples)

            if cache_path is not None:
                print(f"Saving dataset to {cache_path}")
                torch.save(self.episode_data, cache_path)

        # flatten obs_data 
        self.obs_data = [state for ep in self.episode_data for state in ep][:n_samples]


    def _gen_data(self, env_name, n_samples):
        obs_data = []
        self.env = gym.make(env_name)
        max_steps = 100000

        pbar = tqdm(total=n_samples)
        total_samples = 0
        print("Generating examples")
        while total_samples < n_samples:
            obs, _ = self.env.reset()
            episode = []
            for step in range(max_steps):
                obs = torch.tensor(obs)
                # Remove black bar at bottom
                bottom_removed = 12
                obs = obs[:-bottom_removed, bottom_removed//2:-bottom_removed//2, :]
                obs = F.interpolate(obs.permute(2, 0, 1).unsqueeze(0), size=(64, 64)).squeeze(0)

                action = self.env.action_space.sample()

                pbar.update(1)
                prev_obs = obs
                obs, reward, done, info, _ = self.env.step(action)
                episode.append((prev_obs, action, done))
                if done:
                    break
            
            total_samples += len(episode)
            print(total_samples)
            obs_data.append(episode)
        pbar.close()
        return obs_data
    
    def __len__(self):
        return len(self.obs_data)
    
    def __getitem__(self, idx):
        # Normalize
        img, action, done = self.obs_data[idx]
        img = (img - 127.5) / 127.5

        return img, action, done

class GymDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, train_size = 1024, val_size = 256, cache_dir = None, model='vae', vae_ckpt_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.cache_dir = cache_dir
        self.model = model
        self.vae_ckpt_path = vae_ckpt_path


    def setup(self, stage=None):
        train_cache = None
        val_cache = None
        
        if self.cache_dir is not None:
            print('Caching data')
            train_cache = self.cache_dir + '/train.pt'
            val_cache = self.cache_dir + '/val.pt'

        self.train_dataset = GymObservationDataset("CarRacing-v3", self.train_size, cache_path=train_cache)
        self.val_dataset = GymObservationDataset("CarRacing-v3", self.val_size, cache_path=val_cache)
        if self.model == 'rnn':
            self.vae = VAE.load_from_checkpoint(self.vae_ckpt_path)
            self.vae.eval()

            self.train_dataset = RNNDataset(self.train_dataset, self.vae)
            self.val_dataset = RNNDataset(self.val_dataset, self.vae)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class RNNDataset(Dataset):
    def __init__(self, observation_dataset, vae, chunk_size=25):
        self.chunk_size = chunk_size
        self.sequences = []

        for episode in observation_dataset.episode_data:
            for i in range(0, len(episode), chunk_size):
                if len(episode[i:i+chunk_size]) < chunk_size:
                    continue

                # Compute latents
                imgs = [(img - 127.5) / 127.5 for img, _, _ in episode[i:i + chunk_size]]
                imgs = torch.stack(imgs, dim=0)

                acts = torch.stack([torch.from_numpy(act) for _, act, _ in episode[i:i + chunk_size]])
                with torch.no_grad():
                    _, mean, logvar = vae.encode(imgs.to(device=vae.device), sample=False)

                self.sequences.append((imgs.cpu(), acts.cpu(), mean.cpu(), logvar.cpu()))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

