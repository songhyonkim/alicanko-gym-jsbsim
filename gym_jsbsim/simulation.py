import jsbsim
import os
import time
import yaml
from typing import Dict, Union
import gym_jsbsim.properties as prp
from gym_jsbsim.aircraft import Aircraft, f16


class Simulation(object):
    """
    A class which wraps an instance of JSBSim and manages communication with it.
    """
    encoding = 'utf-8'  # encoding of bytes returned by JSBSim Cython funcs
    JSBSIM_ROOT_DIR = os.path.abspath("jsbsim_conf")

    def __init__(self,
                 sim_frequency_hz: float = 60.0,
                 aircraft: Aircraft = f16,
                 init_conditions: Dict[prp.Property, float] = None):
        """
        Constructor. Creates an instance of JSBSim and sets initial conditions.

        :param sim_frequency_hz: the JSBSim integration frequency in Hz.
        :param aircraft: name of aircraft to be loaded.
            JSBSim looks for file \\model_name\\model_name.xml from root dir.
        :param init_conditions: dict mapping properties to their initial values.
            Defaults to None, causing a default set of initial props to be used.
        """
        self.jsbsim = jsbsim.FGFDMExec(root_dir=self.JSBSIM_ROOT_DIR)
        self.jsbsim.set_debug_level(0)
        with open('config.yml', 'r') as file:
            configurations = yaml.safe_load(file)
        if configurations['general']['flightgear'] == 'true':
            self.jsbsim.enable_output()
        else:
            self.jsbsim.disable_output()
        self.sim_dt = 1.0 / sim_frequency_hz
        self.aircraft = aircraft
        self.initialise(self.sim_dt, self.aircraft.jsbsim_id, init_conditions)
        self.wall_clock_dt = None

    def __getitem__(self, prop: Union[prp.BoundedProperty, prp.Property]) -> float:
        """
        Retrieves specified simulation property.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        :param prop: BoundedProperty, the property to be retrieved
        :return: float
        """
        return self.jsbsim[prop.name]

    def __setitem__(self, prop: Union[prp.BoundedProperty, prp.Property], value) -> None:
        """
        Sets simulation property to specified value.

        Properties are identified by strings. A list can be found in the JSBSim
        reference manual, launching JSBSim with '--catalog' command line arg or
        calling FGFDMExec.get_property_catalog().

        Warning: JSBSim will create new properties if the specified one exists.
        If the property you are setting is read-only in JSBSim the operation
        will silently fail.

        :param prop: BoundedProperty, the property to be retrieved
        :param value: object?, the value to be set
        """
        self.jsbsim[prop.name] = value

    def load_model(self, model_name: str) -> None:
        """
        Loads the specified aircraft config into the simulation.

        The root JSBSim directory aircraft folder is searched for the aircraft
        XML config file.

        :param model_name: string, the aircraft name
        """
        load_success = self.jsbsim.load_model(model_name)

        if not load_success:
            raise RuntimeError('JSBSim could not find specified model_name: '
                               + model_name)

    def get_aircraft(self) -> Aircraft:
        """
        Gets the Aircraft this sim was initialised with.
        """
        return self.aircraft

    def get_loaded_model_name(self) -> str:
        """
        Gets the name of the aircraft model currently loaded in JSBSim.

        :return: string, the name of the aircraft model if one is loaded, or
            None if no model is loaded.
        """
        name: str = self.jsbsim.get_model_name().decode(self.encoding)
        if name:
            return name
        else:
            # name is empty string, no model is loaded
            return None

    def get_sim_time(self) -> float:
        """ Gets the simulation time from JSBSim, a float. """
        return self.jsbsim['simulation/sim-time-sec']

    def initialise(self, dt: float, model_name: str,
                   init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Loads an aircraft and initialises simulation conditions.

        :param dt: float, the JSBSim integration timestep in seconds
        :param model_name: string, name of aircraft to be loaded
        :param init_conditions: dict mapping properties to their initial values
        """
        self.load_model(model_name)
        self.jsbsim.set_dt(dt)
        self.set_initial_conditions(init_conditions)
        success = self.jsbsim.run_ic()
        self.propulsion_init_running(-1)
        if not success:
            raise RuntimeError('JSBSim failed to init simulation conditions.')

    def set_initial_conditions(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        if init_conditions is not None:
            for prop, value in init_conditions.items():
                self[prop] = value

    def reinitialise(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Resets JSBSim to initial conditions.

        The same aircraft and other settings are kept loaded in JSBSim. If a
        dict of ICs is provided, JSBSim is initialised using these, else the
        last specified ICs are used.

        :param init_conditions: dict mapping properties to their initial values
        """
        self.set_initial_conditions(init_conditions=init_conditions)
        no_output_reset_mode = 0
        self.jsbsim.reset_to_initial_conditions(no_output_reset_mode)

    def run(self) -> bool:
        """
        Runs a single timestep in the JSBSim simulation.

        JSBSim monitors the simulation and detects whether it thinks it should
        end, e.g. because a simulation time was specified. False is returned
        if JSBSim termination criteria are met.

        :return: bool, False if sim has met JSBSim termination criteria else True.
        """
        result = self.jsbsim.run()
        if self.wall_clock_dt is not None:
            time.sleep(self.wall_clock_dt)
        return result

    def close(self):
        """ Closes the simulation and any plots. """
        if self.jsbsim:
            self.jsbsim = None

    def set_simulation_time_factor(self, time_factor):
        """
        Specifies a factor, relative to realtime, for simulation to run at.

        The simulation runs at realtime for time_factor = 1. It runs at double
        speed for time_factor=2, and half speed for 0.5.

        :param time_factor: int or float, nonzero, sim speed relative to realtime
            if None, the simulation is run at maximum computational speed
        """
        if time_factor is None:
            self.wall_clock_dt = None
        elif time_factor <= 0:
            raise ValueError('time factor must be positive and non-zero')
        else:
            self.wall_clock_dt = self.sim_dt / time_factor

    def propulsion_init_running(self, i):
        propulsion = self.jsbsim.get_propulsion()
        n = propulsion.get_num_engines()
        if i >= 0:
            if i >= n:
                raise IndexError("Tried to initialize a non-existent engine!")
            propulsion.get_engine(i).init_running()
            propulsion.get_steady_state()
        else:
            for j in range(n):
                propulsion.get_engine(j).init_running()
            propulsion.get_steady_state()
