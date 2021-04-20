import gym
import argparse
import gym_jsbsim
from gym_jsbsim.agents import Agents
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("algorithm", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("timesteps", type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = DummyVecEnv([lambda: Monitor(env, info_keywords=('action',), filename="./reward_logs/train_monitor.csv")])
    env = VecNormalize(env)

    eval_callback = EvalCallback(env, best_model_save_path='./models/', eval_freq=args.timesteps/100, deterministic=True)

    if args.model == "none":
        print("--> Alican's LOG: A new model will be created for training")
        model = Agents.get_model(env, args.algorithm)
    else:
        print("--> Alican's LOG: An already existed model will be used for training")
        model = Agents.load_model(env, args.algorithm, args.model)

    model.learn(total_timesteps=args.timesteps, callback=eval_callback, log_interval=20,
                tb_log_name=str(args.timesteps))


if __name__ == "__main__":
    main()
