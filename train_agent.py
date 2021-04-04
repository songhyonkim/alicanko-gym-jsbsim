import gym
import argparse
import gym_jsbsim
from gym_jsbsim.agents import Agents
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("algorithm", type=str)
    parser.add_argument("timesteps", type=int)
    parser.add_argument("loop_count", type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    env = DummyVecEnv([lambda: Monitor(env, filename="./reward_logs/train_monitor.csv")])

    model_string = "F16FDM-"
    timesteps = args.timesteps
    first_run = True
    for x in range(args.loop_count):
        if first_run:
            model = Agents.get_model(env, args.algorithm)
            first_run = False
        else:
            model = Agents.load_model(env, args.algorithm, model_string + str(timesteps))
            model.load_replay_buffer("replay_buffer")
            model.set_env(env)
            timesteps = timesteps + args.timesteps
        model.learn(total_timesteps=args.timesteps, log_interval=20,
                    tb_log_name=str(timesteps - args.timesteps) + "-" + str(timesteps))
        model.save(model_string + str(timesteps))
        if args.loop_count > 1:
            model.save_replay_buffer("replay_buffer")


if __name__ == "__main__":
    main()
