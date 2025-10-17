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

# Collects observations for the VAE to encoder
# This uses random rollout to collect observations
# Hopefully this is general enough for a lot of envs
class GymObservationDataset(Dataset):
    def __init__(self, env_name: str, n_samples: int = 500, cache_path: Optional[str] = None):
        self.obs_data = []
        if cache_path is not None and len(os.listdir(cache_path)) > 0:
            if len(os.listdir(cache_path)) != n_samples:
                raise Exception(f"Delete the old cache and regenerate it. Expected {n_samples} samples, found {len(os.listdir(cache_path))} ")
            image_files = [f for f in os.listdir(cache_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            to_tensor = transforms.ToTensor()

            for file in image_files:
                img_path = os.path.join(cache_path, file)
                img = Image.open(img_path).convert("RGB")
                tensor = to_tensor(img) * 255
                tensor = tensor.permute(1, 2, 0)
                tensor = tensor.type(torch.uint8)
                self.obs_data.append(tensor)
        else:
            self.obs_data = self._gen_data(env_name, n_samples)

            for i, img in enumerate(self.obs_data):
                img = img.cpu().numpy()
                pil_img = Image.fromarray(img)
                
                save_path = os.path.join(cache_path, f"image_{i:04d}.png")
                pil_img.save(save_path)


    def _gen_data(self, env_name, n_samples):
        obs_data = []
        self.env = gym.make(env_name)
        max_steps = 1000

        pbar = tqdm(total=n_samples)

        print("Generating examples")
        while len(obs_data) < n_samples:
            obs, _ = self.env.reset()
            for step in range(max_steps):
                obs = torch.tensor(obs)
                obs_data.append(obs)

                action = self.env.action_space.sample()

                pbar.update(1)
                obs, reward, done, info, _ = self.env.step(action)
                if done:
                    break
        pbar.close()
        return obs_data[:n_samples]
    
    def __len__(self):
        return len(self.obs_data)
    
    def __getitem__(self, idx):
        # Normalize
        data = self.obs_data[idx].permute(2, 0, 1) # C H W
        data = (data - 127.5) / 127.5

        return data


class GymDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, train_size = 1024, val_size = 256, cache_dir = None):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.cache_dir = cache_dir

    def setup(self, stage=None):
        train_cache = None
        val_cache = None
        
        if self.cache_dir is not None:
            print('Caching data')
            train_cache = self.cache_dir + '/train'
            val_cache = self.cache_dir + '/val'
            os.makedirs(train_cache, exist_ok=True)
            os.makedirs(val_cache, exist_ok=True)

        self.train_dataset = GymObservationDataset("CarRacing-v3", self.train_size, cache_path=train_cache)
        self.val_dataset = GymObservationDataset("CarRacing-v3", self.val_size, cache_path=val_cache)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
