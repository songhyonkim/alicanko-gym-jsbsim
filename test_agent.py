import pickle
import yaml
import gym_jsbsim
from gym_jsbsim.agents import Agents
from stable_baselines3.common.env_util import make_vec_env


def main():
    with open('config.yml') as file:
        configurations = yaml.safe_load(file)
    configurations['general']['flightgear'] = 'true'
    configurations['general']['agent_interaction_freq'] = 5
    with open('config.yml', 'w') as file:
        yaml.dump(configurations, file)

    env_make = make_vec_env(configurations['general']['env'], n_envs=1, seed=0)
    with open(configurations['test']['model'] + '/env.pkl', "rb") as file_handler:
        env = pickle.load(file_handler)
    env.set_venv(env_make)

    model = Agents.load_model(env, configurations['general']['algorithm'], configurations['test']['model'] + '/best_model')

    obs = env.reset()

    mode = configurations['test']['mode']
    ''' 임시
    if mode == 'csv':
        now = datetime.now()
        save_path = './logs/' + now.strftime("%B %d, %Y - %H.%M")
        os.makedirs(save_path, exist_ok=True)
    '''

    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode=mode)


if __name__ == "__main__":
    main()
