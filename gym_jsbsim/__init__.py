import gym.envs.registration
import enum
from gym_jsbsim import utils

"""
This script registers all combinations of task, and aircraft
 etc. with OpenAI Gym so that they can be instantiated with a gym.make(id)
 command.

The gym_jsbsim.Envs enum stores all registered environments as members with
 their gym id string as value. This allows convenient autocompletion and value
 safety. To use do:
       env = gym.make(gym_jsbsim.Envs.desired_environment.value)
"""

for env_id, (task, plane) in utils.get_env_id_kwargs_map().items():
    entry_point = 'gym_jsbsim.environment:JsbSimEnv'
    kwargs = dict(task_type=task,
                  aircraft=plane)
    gym.envs.registration.register(id=env_id,
                                   entry_point=entry_point,
                                   kwargs=kwargs)

# make an Enum storing every Gym-JSBSim environment ID for convenience and value safety
Envs = enum.Enum.__call__('Envs', [(utils.AttributeFormatter.translate(env_id), env_id)
                                   for env_id in utils.get_env_id_kwargs_map().keys()])
