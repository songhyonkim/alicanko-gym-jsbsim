import os
import pickle
import yaml
import gym_jsbsim
from datetime import datetime
from gym_jsbsim.agents import Agents
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import pyttsx3  # sound library for pure fun


def main():
    # get init time and use it for save path
    now = datetime.now()
    save_path = './trained/' + now.strftime("%B %d, %Y - %H.%M")
    os.mkdir(save_path)
    # using sound library for pure fun
    engine = pyttsx3.init()  # object creation
    engine.setProperty('rate', 150)     # setting up new voice rate

    with open('config.yml') as file:
        configurations = yaml.safe_load(file)
    configurations['general']['flightgear'] = 'false'
    configurations['general']['agent_interaction_freq'] = 5
    with open('config.yml', 'w') as file:
        yaml.dump(configurations, file)

    env_make = make_vec_env(configurations['general']['env'], n_envs=1, seed=0)
    env = VecNormalize(env_make, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, best_model_save_path=save_path, eval_freq=configurations['train']['timesteps']/100, deterministic=True)
    with open(save_path + '/env.pkl', "wb") as file_handler:
        pickle.dump(env, file_handler, pickle.HIGHEST_PROTOCOL)

    if configurations['train']['model'] == "none":
        print("--> Alican's LOG: A new model will be created for training")
        model = Agents.create_model(env, configurations['general']['algorithm'], save_path)
    else:
        print("--> Alican's LOG: An already existed model will be used for training")
        model = Agents.load_model(env, configurations['general']['algorithm'], configurations['train']['model'] + '/best_model')

    model.learn(total_timesteps=configurations['train']['timesteps'], callback=eval_callback, log_interval=20)

    engine.say("Training is finished!")
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    main()
