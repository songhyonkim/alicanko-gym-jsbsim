import gym
import numpy as np
import argparse
from datetime import datetime
import gym_jsbsim
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("timesteps", type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = DummyVecEnv([lambda: env])

    # the noise object for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise)
    model.learn(total_timesteps=args.timesteps)
    model.save(datetime.now().strftime("%d-%m_%H-%M"))


if __name__ == "__main__":
    main()
