import numpy as np
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines3.td3.policies import MlpPolicy as TD3_MlpPolicy
from stable_baselines3.sac.policies import MlpPolicy as SAC_MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


class Agents:
    @staticmethod
    def create_model(env, algorithm, save_path):
        # the noise object
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                    sigma=float(0.2) * np.ones(n_actions),
                                                    theta=0.15)
        if algorithm == "ddpg":
            return DDPG(DDPG_MlpPolicy,
                        env,
                        learning_rate=0.001,
                        buffer_size=1000000,
                        batch_size=64,
                        tau=0.001,
                        gamma=0.99,
                        train_freq=(10, "step"),
                        action_noise=action_noise,
                        policy_kwargs=dict(optimizer_class=th.optim.AdamW),
                        tensorboard_log=save_path)
        elif algorithm == "td3":
            return TD3(TD3_MlpPolicy,
                       env,
                       action_noise=action_noise,
                       tensorboard_log=save_path)
        elif algorithm == "sac":
            return SAC(SAC_MlpPolicy,
                       env,
                       action_noise=action_noise,
                       tensorboard_log=save_path)
        else:
            raise Exception("--> Alican's LOG: Unknown agent type!")

    @staticmethod
    def load_model(env, algorithm, filename):
        if algorithm == "ddpg":
            return DDPG.load(filename, env=env)
        elif algorithm == "td3":
            return TD3.load(filename, env=env)
        elif algorithm == "sac":
            return SAC.load(filename, env=env)
        else:
            raise Exception("--> Alican's LOG: Unknown agent type!")
