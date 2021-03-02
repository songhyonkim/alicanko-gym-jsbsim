import types
import math
import warnings
import gym
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple, NamedTuple, Type
import gym_jsbsim.properties as prp
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.properties import BoundedProperty, Property
from gym_jsbsim import assessors, rewards
from gym_jsbsim.task import Task

from gym_jsbsim.rewards import RewardStub


class FlightTask(Task, ABC):
    """
    Abstract superclass for flight tasks.

    Concrete subclasses should implement the following:
        state_variables attribute: tuple of Propertys, the task's state representation
        action_variables attribute: tuple of Propertys, the task's actions
        get_initial_conditions(): returns dict mapping InitialPropertys to initial values
        _is_terminal(): determines episode termination
        (optional) _new_episode_init(): performs any control input/initialisation on episode reset
        (optional) _update_custom_properties: updates any custom properties in the sim
    """
    INITIAL_ALTITUDE_FT = 5000
    base_state_variables = (prp.altitude_sl_ft, prp.pitch_rad, prp.roll_rad,
                            prp.u_fps, prp.v_fps, prp.w_fps,
                            prp.p_radps, prp.q_radps, prp.r_radps,
                            prp.aileron_left, prp.aileron_right, prp.elevator,
                            prp.rudder)
    base_initial_conditions = types.MappingProxyType(  # MappingProxyType makes dict immutable
        {prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
         prp.initial_terrain_altitude_ft: 0.00000001,
         prp.initial_longitude_geoc_deg: -2.3273,
         prp.initial_latitude_geod_deg: 51.3781  # corresponds to UoBath
         }
    )
    last_agent_reward = Property('reward/last_agent_reward', 'agent reward from step; includes'
                                                             'potential-based shaping reward')
    last_assessment_reward = Property('reward/last_assess_reward', 'assessment reward from step;'
                                                                   'excludes shaping')
    state_variables: Tuple[BoundedProperty, ...]
    action_variables: Tuple[BoundedProperty, ...]
    assessor: assessors.Assessor
    State: Type[NamedTuple]

    def __init__(self, assessor: assessors.Assessor, debug: bool = False) -> None:
        self.last_state = None
        self.assessor = assessor
        self._make_state_class()
        self.debug = debug

    def _make_state_class(self) -> None:
        """ Creates a namedtuple for readable State data """
        # get list of state property names, containing legal chars only
        legal_attribute_names = [prop.get_legal_name() for prop in
                                 self.state_variables]
        self.State = namedtuple('State', legal_attribute_names)

    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[NamedTuple, float, bool, Dict]:
        # input actions
        for prop, command in zip(self.action_variables, action):
            sim[prop] = command

        # run simulation
        for _ in range(sim_steps):
            sim.run()

        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        done = self._is_terminal(sim)
        reward = self.assessor.assess(state, self.last_state, done)
        if done:
            reward = self._reward_terminal_override(reward, sim)
        if self.debug:
            self._validate_state(state, done, action, reward)
        self._store_reward(reward, sim)
        self.last_state = state
        info = {'reward': reward}

        return state, reward.agent_reward(), done, info

    def _validate_state(self, state, done, action, reward):
        if any(math.isnan(el) for el in state):  # float('nan') in state doesn't work!
            msg = (f'Invalid state encountered!\n'
                   f'State: {state}\n'
                   f'Prev. State: {self.last_state}\n'
                   f'Action: {action}\n'
                   f'Terminal: {done}\n'
                   f'Reward: {reward}')
            warnings.warn(msg, RuntimeWarning)

    def _store_reward(self, reward: rewards.Reward, sim: Simulation):
        sim[self.last_agent_reward] = reward.agent_reward()
        sim[self.last_assessment_reward] = reward.assessment_reward()

    def _update_custom_properties(self, sim: Simulation) -> None:
        """ Calculates any custom properties which change every timestep. """
        pass

    @abstractmethod
    def _is_terminal(self, sim: Simulation) -> bool:
        """ Determines whether the current episode should terminate.

        :param sim: the current simulation
        :return: True if the episode should terminate else False
        """
        ...

    @abstractmethod
    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation) -> bool:
        """
        Determines whether a custom reward is needed, e.g. because
        a terminal condition is met.
        """
        ...

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._new_episode_init(sim)
        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        self.last_state = state
        return state

    def _new_episode_init(self, sim: Simulation) -> None:
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        By default it simply starts the aircraft engines.
        """
        sim.start_engines()
        sim.raise_landing_gear()
        self._store_reward(RewardStub(1.0, 1.0), sim)

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        ...

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float32')

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float32')
