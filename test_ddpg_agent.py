import gym
import argparse
import gym_jsbsim
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = DummyVecEnv([lambda: env])

    model = DDPG.load("28-02_22-06", env=env)
    model.set_env(env)

    done = False
    obs = env.reset()
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode="human")


if __name__ == "__main__":
    main()
