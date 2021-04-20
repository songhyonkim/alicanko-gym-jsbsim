import gym
import numpy as np
import math
import warnings
import gym_jsbsim.properties as prp
from gym_jsbsim import utils
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.properties import BoundedProperty, Property
from gym_jsbsim.aircraft import Aircraft
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type


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
    last_agent_reward = Property('reward/last_agent_reward', 'agent reward from step; includes'
                                                             'potential-based shaping reward')
    state_variables: Tuple[BoundedProperty, ...]
    action_variables: Tuple[BoundedProperty, ...]
    State: Type[NamedTuple]

    def __init__(self, debug: bool = False) -> None:
        self.last_state = None
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
        reward = self._get_reward(state, sim)
        if self.debug:
            self._validate_state(state, done, action, reward)
        self._store_reward(reward, sim)
        self.last_state = state
        info = {'reward': reward, 'action': action}

        return state, reward, done, info

    def _validate_state(self, state, done, action, reward):
        if any(math.isnan(el) for el in state):  # float('nan') in state doesn't work!
            msg = (f'Invalid state encountered!\n'
                   f'State: {state}\n'
                   f'Prev. State: {self.last_state}\n'
                   f'Action: {action}\n'
                   f'Terminal: {done}\n'
                   f'Reward: {reward}')
            warnings.warn(msg, RuntimeWarning)

    def _store_reward(self, reward, sim: Simulation):
        sim[self.last_agent_reward] = reward

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
    def _get_reward(self, state, sim: Simulation) -> float:
        """ Compute reward for the task.

        :param state: the current state
        :param sim: the current simulation
        :return: Float calculated reward
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
        """
        self._store_reward(1.0, sim)

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        ...

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype=float)

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float32')


class Heading2ControlTask(FlightTask):
    """
    A task in which the agent must perform steady, level flight maintaining its
    initial heading.
    """
    INITIAL_ALTITUDE_FT = 5000
    INITIAL_HEADING_DEG = 270
    DEFAULT_EPISODE_TIME_S = 60.
    """New variables"""
    target_track_deg = BoundedProperty('target/track-deg', 'desired heading [deg]',
                                       prp.heading_deg.min, prp.heading_deg.max)
    track_error_deg = BoundedProperty('error/track-error-deg',
                                      'error to desired track [deg]', -180, 180)
    altitude_error_ft = BoundedProperty('error/altitude-error-ft',
                                        'error to desired altitude [ft]',
                                        prp.altitude_sl_ft.min,
                                        prp.altitude_sl_ft.max)

    def __init__(self, step_frequency_hz: float, aircraft: Aircraft, episode_time_s: float = DEFAULT_EPISODE_TIME_S):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty('info/steps_left', 'steps remaining in episode', 0, episode_steps)
        self.aircraft = aircraft
        self.state_variables = (self.altitude_error_ft,
                                self.track_error_deg,
                                self.steps_left,
                                prp.pitch_rad,
                                prp.roll_rad,
                                prp.v_down_fps,
                                prp.vc_fps,
                                prp.p_radps,
                                prp.q_radps,
                                prp.r_radps)
        self.action_variables = (prp.aileron_cmd, prp.elevator_cmd)
        super().__init__()

    def get_initial_conditions(self) -> Dict[Property, float]:
        init_conditions = {prp.initial_altitude_ft: self.INITIAL_ALTITUDE_FT,
                           prp.initial_terrain_altitude_ft: 0.00000001,
                           prp.initial_longitude_geoc_deg: 32.565556,
                           prp.initial_latitude_geod_deg: 40.078889,  # corresponds to Akinci
                           prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
                           prp.initial_v_fps: 0,
                           prp.initial_w_fps: 0,
                           prp.initial_p_radps: 0,
                           prp.initial_q_radps: 0,
                           prp.initial_r_radps: 0,
                           prp.initial_roc_fpm: 0,
                           prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
                           prp.gear: 0,
                           prp.gear_all_cmd: 0,
                           prp.rudder_cmd: 0,
                           prp.throttle_cmd: 0.8,
                           prp.mixture_cmd: 1,
                           }
        return init_conditions

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_track_error(sim)
        self._update_altitude_error(sim)
        self._decrement_steps_left(sim)

    def _update_track_error(self, sim: Simulation):
        track_deg = sim[prp.heading_deg]
        target_track_deg = sim[self.target_track_deg]
        error_deg = utils.reduce_reflex_angle_deg(target_track_deg - track_deg)
        sim[self.track_error_deg] = error_deg

    def _update_altitude_error(self, sim: Simulation):
        altitude_ft = sim[prp.altitude_sl_ft]
        target_altitude_ft = self._get_target_altitude()
        error_ft = altitude_ft - target_altitude_ft
        sim[self.altitude_error_ft] = error_ft

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _get_reward(self, state, sim: Simulation) -> float:
        """
        Compute reward for task
        """
        # Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable
        heading_error_scale = 5.0  # degrees
        heading_r = math.exp(-((sim[self.track_error_deg] / heading_error_scale) ** 2))

        alt_error_scale = 50.0  # feet
        alt_r = math.exp(-((sim[self.altitude_error_ft] / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((sim[prp.roll_rad] / roll_error_scale) ** 2))

        """speed_error_scale = 10  # fps (~5%)
        speed_r = math.exp(-(((sim[prp.u_fps] - self.aircraft.get_cruise_speed_fps()) / speed_error_scale) ** 2))

        # accel scale in "g"s
        accel_error_scale_x = 0.1
        accel_error_scale_y = 0.1
        accel_error_scale_z = 0.5
        try:
            accel_r = math.exp(
                -(
                    (sim[prp.accelerations_n_pilot_x_norm] / accel_error_scale_x) ** 2
                    + (sim[prp.accelerations_n_pilot_y_norm] / accel_error_scale_y) ** 2
                    + ((sim[prp.accelerations_n_pilot_z_norm] + 1) / accel_error_scale_z) ** 2
                )  # normal value for z component is -1 g
            ) ** (
                1 / 3
            )  # geometric mean
        except OverflowError:
            accel_r = 0"""

        reward = (heading_r * alt_r * roll_r) ** (1 / 3)
        return reward

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        is_heading_out_of_bounds = abs(sim[self.track_error_deg]) > 10
        is_altitude_out_of_bounds = abs(sim[self.altitude_error_ft]) > 250
        terminal_step = sim[self.steps_left] <= 0
        return terminal_step or is_altitude_out_of_bounds or is_heading_out_of_bounds

    def _new_episode_init(self, sim: Simulation) -> None:
        super()._new_episode_init(sim)
        sim[self.steps_left] = self.steps_left.max
        sim[self.target_track_deg] = self._get_target_track()

    def _get_target_track(self) -> float:
        # use the same, initial heading every episode
        return self.INITIAL_HEADING_DEG

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (prp.u_fps, prp.altitude_sl_ft, self.altitude_error_ft, self.target_track_deg,
                self.track_error_deg, prp.roll_rad, prp.sideslip_deg, self.last_agent_reward,
                self.steps_left)
