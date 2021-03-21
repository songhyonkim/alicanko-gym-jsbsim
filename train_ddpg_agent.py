import gym
import numpy as np
import argparse
import gym_jsbsim
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3 import DDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("timesteps", type=int)
    parser.add_argument("loop_count", type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = DummyVecEnv([lambda: Monitor(env)])

    # the noise object for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model_string = "F16FDM-"
    timesteps = args.timesteps
    first_run = False
    for x in range(args.loop_count):
        if not first_run:
            model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, tensorboard_log="./tensor_logs/")
            first_run = True
        else:
            model = DDPG.load(model_string + str(timesteps), env=env, action_noise=action_noise,
                              tensorboard_log="./tensor_logs/")
            model.load_replay_buffer("ddpg_replay_buffer")
            model.set_env(env)
            timesteps = timesteps + args.timesteps
        model.learn(total_timesteps=args.timesteps, log_interval=20,
                    tb_log_name=str(timesteps - args.timesteps) + "-" + str(timesteps))
        model.save(model_string + str(timesteps))
        if args.loop_count > 1:
            model.save_replay_buffer("ddpg_replay_buffer")


if __name__ == "__main__":
    main()
