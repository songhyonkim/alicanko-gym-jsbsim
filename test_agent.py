import gym
import argparse
import gym_jsbsim
from gym_jsbsim.agents import Agents
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("algorithm", type=str)
    parser.add_argument("model", type=str)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

    model = Agents.load_model(env, args.algorithm, args.model)
    model.set_env(env)

    done = False
    obs = env.reset()
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode="human")


if __name__ == "__main__":
    main()
