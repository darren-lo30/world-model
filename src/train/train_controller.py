import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import cma
import numpy as np
from torch import nn
import click
import gymnasium as gym
from src.models.vae import VAE
from src.models.rnn import WorldModelRNN
import torch.nn.functional as F
from tqdm import tqdm
import random
import os
from multiprocessing import Pool
import multiprocessing as mp
import warnings
import wandb
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate_model_rollout_job(controller_params, env_name, vae_path, rnn_path, num_rollouts = 16):
    vae = VAE.load_from_checkpoint(vae_path).to(device).eval()
    rnn = WorldModelRNN.load_from_checkpoint(rnn_path).to(device).eval()
    controller = Controller(vae.decoder.latent_size, rnn.hidden_size, 3).to(device=device)
    controller.set_params(controller_params)

    return evaluate_model_rollout(controller, env_name, vae, rnn, num_rollouts)

@torch.no_grad()
def evaluate_model_rollout(controller, env_name, vae, rnn, num_rollouts = 16):
    avg_reward = 0.0
    max_step = 1000
    pbar = tqdm(range(num_rollouts))
    for i in pbar:
        if i > 0:
            pbar.set_postfix(avg_reward = avg_reward / i)

        step = 0
        total_reward = 0.0

        env = gym.make(env_name)
        obs, _ = env.reset(seed=random.randint(0, 1_000_000))
        done = False
        h, c = rnn.get_init_state(1)
        h = h.to(device=device)
        c = c.to(device=device)

        while not done and step < max_step:
            obs = torch.tensor(obs)
            # Remove black bar at bottom
            bottom_removed = 12
            obs = obs[:-bottom_removed, bottom_removed//2:-bottom_removed//2, :]
            obs = F.interpolate(obs.permute(2, 0, 1).unsqueeze(0), size=(64, 64)).squeeze(0)
            obs = (obs - 127.5) / 127.5
            obs = obs.to(device=device).unsqueeze(0)
            encoded, _, _ = vae.encode(obs, sample=False)
            action = controller(encoded, h)           
            _, _, _, (h, c) = rnn(torch.cat([encoded.unsqueeze(0), action.view(1, 1, -1)], dim=-1), h, c)

            obs, reward, done, info, _ = env.step(action.squeeze(0).cpu().numpy())

            total_reward += reward
            step += 1
        env.close()
        avg_reward += total_reward
    return avg_reward / num_rollouts

@torch.no_grad()
def evaluate_model(params, env_name, vae_path, rnn_path, num_rollouts=16):
    # Prepare args for workers
    args_list = [
        (param, env_name, vae_path, rnn_path, num_rollouts)
        for param in params
    ]

    # Use all CPUs or a reasonable number
    num_workers = os.cpu_count()

    with Pool(processes=num_workers) as pool:
        rewards = pool.starmap(evaluate_model_rollout_job, args_list)
    print(rewards)
    return rewards
    

class Controller(nn.Module):
    def __init__(self, latent_size, hidden_size, action_size):
        super().__init__()
        self.fc = nn.Linear(latent_size + hidden_size, action_size, bias=False)

    def forward(self, z, h):
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))

    def get_param_dim(self):
        return self.fc.weight.flatten().shape[0]

    def set_params(self, params):
        weights_shape = self.fc.weight.flatten().shape[0]

        weights = params.reshape(self.fc.weight.shape)

        self.fc.weight.data = torch.from_numpy(weights).to(device=device, dtype=torch.float32)


class CMAES():
    def __init__(self, env_name, vae_path, rnn_path, controller, save_dir, init_sigma=0.1, pop_size=16, num_rollouts=4, generations=2000, save_freq=200):
        super().__init__()

        self.init_params = np.zeros(controller.get_param_dim())
        self.init_sigma = init_sigma

        opts = cma.CMAOptions()
        opts.set("popsize", pop_size)
        opts.set("maxiter", generations)
        opts.set("verbose", -9)  # silence
        self.generations = generations

        self.opts = opts

        self.vae_path = vae_path
        self.env_name = env_name
        self.rnn_path = rnn_path
        self.num_rollouts = num_rollouts
        self.save_freq = save_freq
        self.controller = controller
        self.save_dir = save_dir


    def run(self):
        es = cma.CMAEvolutionStrategy(self.init_params, self.init_sigma, self.opts)
        generation = 0
        with tqdm(total=self.generations, desc="Generations") as pbar:
            while not es.stop():
                solutions = es.ask()
                rewards = evaluate_model(solutions, self.env_name, self.vae_path, self.rnn_path, self.num_rollouts)
                
                best_reward = max(rewards)
                worst_reward = min(rewards)
                avg_reward = np.mean(rewards)

                fitnesses = [-f for f in rewards]
                es.tell(solutions, fitnesses)

                if generation % self.save_freq == 0:
                    best_reward_idx = np.argmax(rewards)
                    print(f'Saving model with reward {rewards[best_reward_idx]}')
                    self.controller.set_params(solutions[best_reward_idx])
                    torch.save(self.controller.state_dict(), f'{self.save_dir}/controller_{generation}.pt')
                    

                best_reward = -es.result.fbest
                wandb.log({
                    "best_reward": best_reward,
                    "worst_reward": worst_reward,
                    "avg_reward": avg_reward,
                    "generation": generation
                })
                generation += 1
                pbar.update(1)

        best_params = es.result.xbest

        controller.set_params(best_params)

        return controller

@click.command()
@click.option('--vae_path')
@click.option('--rnn_path')
@click.option('--pop_size', default=16)
@click.option('--num_rollouts', default=4)
@click.option('--generations', default=2000)
@click.option('--save_freq', default=200)
@click.option('--save_dir', default='./results/controller')
@click.option('--env_name', default='CarRacing-v3')
def train(vae_path, rnn_path, env_name, pop_size, num_rollouts, generations, save_freq, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project='train-controller', config={
        "env_name": env_name,
        "vae_path": vae_path,
        "rnn_path": rnn_path,
        "algorithm": "CMA-ES"
    })

    vae = VAE.load_from_checkpoint(vae_path)
    rnn = WorldModelRNN.load_from_checkpoint(rnn_path)
    controller = Controller(vae.decoder.latent_size, rnn.hidden_size, 3)

    controller = CMAES(env_name, vae_path, rnn_path, controller, pop_size=pop_size, num_rollouts=num_rollouts, generations=generations, save_freq=save_freq, save_dir=save_dir).run()
    torch.save(controller.state_dict(), f'{save_dir}/controller_final.pt')
    wandb.finish()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    train()

    