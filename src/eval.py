import click
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from src.models.vae import VAE
from src.models.rnn import WorldModelRNN
from src.train.train_controller import Controller, evaluate_model_rollout
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@click.command()
@click.option('--vae_path', required=True)
@click.option('--rnn_path', required=True)
@click.option('--controller_path', required=True)
@click.option('--num_rollouts', default=16)
@click.option('--env_name', default='CarRacing-v3')
@click.option('--video_dir', default='./videos', help='Directory to save evaluation video')
def eval(vae_path, rnn_path, controller_path, env_name, num_rollouts, video_dir):
    vae = VAE.load_from_checkpoint(vae_path)
    rnn = WorldModelRNN.load_from_checkpoint(rnn_path)
    controller = Controller(vae.decoder.latent_size, rnn.hidden_size, 3)
    controller.load_state_dict(torch.load(controller_path, weights_only=True))

    controller.to(device)
    vae.to(device)
    rnn.to(device)

    vae.eval()
    rnn.eval()
    controller.eval()

    # avg_reward = evaluate_model_rollout(controller, env_name, vae, rnn, num_rollouts = num_rollouts)
    # print(f"Average Reward is {avg_reward}")

    os.makedirs(video_dir, exist_ok=True)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix='run',
        episode_trigger=lambda x: True,
        disable_logger=True
    )

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    total_reward = 0.0

    h, c = rnn.get_init_state(1)
    h = h.to(device=device)
    c = c.to(device=device)

    while not (done or truncated):
        # Preprocess observation (same as your logic)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        bottom_removed = 12
        obs_cropped = obs_tensor[:-bottom_removed, bottom_removed//2:-bottom_removed//2, :]
        obs_resized = F.interpolate(
            obs_cropped.permute(2, 0, 1).unsqueeze(0),
            size=(64, 64),
            mode='bilinear'
        ).squeeze(0)
        obs_norm = (obs_resized - 127.5) / 127.5
        obs_batch = obs_norm.to(device=device).unsqueeze(0)

        with torch.no_grad():
            encoded, _, _ = vae.encode(obs_batch, sample=False)
            action = controller(encoded, h)
            _, _, _, (h, c) = rnn(
                torch.cat([encoded.unsqueeze(0), action.view(1, 1, -1)], dim=-1),
                h, c
            )

        obs, reward, done, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        total_reward += reward
        step += 1

    env.close()
    print(f"Video saved to {video_dir}. Final reward: {total_reward:.2f}")

if __name__ == "__main__":
    eval()