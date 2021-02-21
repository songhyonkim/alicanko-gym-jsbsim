import gym
import enum
import numpy as np
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.properties import Property
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple


class Task(ABC):
    """
    Interface for Tasks, modules implementing specific environments in JSBSim.

    A task defines its own state space, action space, termination conditions and agent_reward function.
    """

    @abstractmethod
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Calculates new state, reward and termination.

        :param sim: a Simulation, the simulation from which to extract state
        :param action: sequence of floats, the agent's last action
        :param sim_steps: number of JSBSim integration steps to perform following action
            prior to making observation
        :return: tuple of (observation, reward, done, info) where,
            observation: array, agent's observation of the environment state
            reward: float, the reward for that step
            done: bool, True if the episode is over else False
            info: dict, optional, containing diagnostic info for debugging etc.
        """

    ...

    @abstractmethod
    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        """
        Initialise any state/controls and get first state observation from reset sim.

        :param sim: Simulation, the environment simulation
        :return: np array, the first state observation of the episode
        """
        ...

    @abstractmethod
    def get_initial_conditions(self) -> Optional[Dict[Property, float]]:
        """
        Returns dictionary mapping initial episode conditions to values.

        Episode initial conditions (ICs) are defined by specifying values for
        JSBSim properties, represented by their name (string) in JSBSim.

        JSBSim uses a distinct set of properties for ICs, beginning with 'ic/'
        which differ from property names during the simulation, e.g. "ic/u-fps"
        instead of "velocities/u-fps". See https://jsbsim-team.github.io/jsbsim/

        :return: dict mapping string for each initial condition property to
            initial value, a float, or None to use Env defaults
        """
        ...

    @abstractmethod
    def get_state_space(self) -> gym.Space:
        """ Get the task's state Space object """
        ...

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """ Get the task's action Space object """
        ...

class Shaping(enum.Enum):
    STANDARD = 'STANDARD'
    EXTRA = 'EXTRA'
    EXTRA_SEQUENTIAL = 'EXTRA_SEQUENTIAL'
