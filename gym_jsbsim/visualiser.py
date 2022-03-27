import os
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
import gym_jsbsim.properties as prp
from gym_jsbsim.simulation import Simulation
from typing import NamedTuple, Tuple


class AxesTuple(NamedTuple):
    """ Holds references to figure subplots (axes) """
    axes_state: plt.Axes
    axes_stick: plt.Axes
    axes_throttle: plt.Axes
    axes_rudder: plt.Axes


class FigureVisualiser(object):
    """ Class for manging a matplotlib Figure displaying agent state and actions """
    PLOT_PAUSE_SECONDS = 0.0001
    LABEL_TEXT_KWARGS = dict(fontsize=14,
                             horizontalalignment='right',
                             verticalalignment='baseline')
    VALUE_TEXT_KWARGS = dict(fontsize=14,
                             horizontalalignment='left',
                             verticalalignment='baseline')
    TEXT_X_POSN_LABEL = 0.8
    TEXT_X_POSN_VALUE = 0.9
    TEXT_Y_POSN_INITIAL = 1.0
    TEXT_Y_INCREMENT = -0.1

    def __init__(self, _: Simulation, print_props: Tuple[prp.Property]):
        """
        Constructor.

        Sets here is ft_per_deg_lon, which depends dynamically on aircraft's
        longitude (because of the conversion between geographic and Euclidean
        coordinate systems). We retrieve longitude from the simulation and
        assume it is constant thereafter.

        :param _: (unused) Simulation that will be plotted
        :param print_props: Propertys which will have their values printed to Figure.
            Must be retrievable from the plotted Simulation.
        """
        self.print_props = print_props
        self.figure: plt.Figure = None
        self.csv_file = None
        self.axes: AxesTuple = None
        self.value_texts: Tuple[plt.Text] = None

    def csv(self, sim: Simulation) -> None:
        """
        Save the episode to csv file.

        :param sim: Simulation that will be saved as csv file.
        """
        if not self.csv_file:
            now = datetime.now()
            save_path = './csv_logs'
            save_file = save_path + '/F-16 (' + now.strftime("%Y%m%d-%H%M%S") + ') [Blue].csv'
            os.makedirs(save_path, exist_ok=True)
            self.csv_file = open(save_file, 'w')
            self.csv_file.write('Time,Longitude,Latitude,Altitude,Roll (deg),Pitch (deg),Yaw (deg)\n')
            print(f'CSV file opened: {save_file}')
            time.sleep(1)

        """ save """
        Time        = sim[prp.sim_time_s]
        Longitude   = sim[prp.lng_geoc_deg]
        Latitude    = sim[prp.lat_geod_deg] # position/lat-geod-deg vs. position/lat-gc-deg 확인 필요
        Altitude    = sim[prp.altitude_sl_ft]
        Roll_deg    = math.degrees(sim[prp.roll_rad])
        Pitch_deg   =  math.degrees(sim[prp.pitch_rad])
        Yaw_deg     = sim[prp.heading_deg]

        self.csv_file.write(f'{Time},{Longitude},{Latitude},{Altitude},{Roll_deg},{Pitch_deg},{Yaw_deg}\n')


    def plot(self, sim: Simulation) -> None:
        """
        Creates or updates a 3D plot of the episode.

        :param sim: Simulation that will be plotted
        """
        if not self.figure:
            self.figure, self.axes = self._plot_configure()

        # delete old control surface data points
        for subplot in self.axes[1:]:
            # pop and translate all data points
            while subplot.lines:
                data = subplot.lines.pop()
                del data

        self._print_state(sim)
        self._plot_control_states(sim, self.axes)
        self._plot_control_commands(sim, self.axes)
        plt.pause(self.PLOT_PAUSE_SECONDS)  # voodoo pause needed for figure to update

    def close(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.axes = None
        if self.csv_file:
            self.csv_file.close()

    def _plot_configure(self):
        """
        Creates a figure with subplots for states and actions.

        :return: (figure, axes) where:
            figure: a matplotlib Figure with subplots for state and controls
            axes: an AxesTuple object with references to all figure subplot axes
        """
        plt.ion()  # interactive mode allows dynamic updating of plot
        figure = plt.figure(figsize=(4, 7))

        spec = plt.GridSpec(nrows=3,
                            ncols=2,
                            width_ratios=[5, 1],  # second column very thin
                            height_ratios=[6, 5, 1],  # bottom row very short
                            wspace=0.3)

        # create subplots
        axes_state = figure.add_subplot(spec[0, 0:])
        axes_stick = figure.add_subplot(spec[1, 0])
        axes_throttle = figure.add_subplot(spec[1, 1])
        axes_rudder = figure.add_subplot(spec[2, 0])

        # hide state subplot axes - text will be printed to it
        axes_state.axis('off')
        self._prepare_state_printing(axes_state)

        # config subplot for stick (aileron and elevator control in x/y axes)
        axes_stick.set_xlabel('ailerons [-]', )
        axes_stick.set_ylabel('elevator [-]')
        axes_stick.set_xlim(left=-1, right=1)
        axes_stick.set_ylim(bottom=-1, top=1)
        axes_stick.xaxis.set_label_coords(0.5, 1.08)
        axes_stick.yaxis.set_label_coords(-0.05, 0.5)
        # make axes cross at origin
        axes_stick.spines['left'].set_position('zero')
        axes_stick.spines['bottom'].set_position('zero')
        # only show ticks at extremes of range
        axes_stick.set_xticks([-1, 1])
        axes_stick.xaxis.set_ticks_position('bottom')
        axes_stick.set_yticks([-1, 1])
        axes_stick.yaxis.set_ticks_position('left')
        axes_stick.tick_params(which='both', direction='inout')
        # show minor ticks throughout
        minor_locator = plt.MultipleLocator(0.2)
        axes_stick.xaxis.set_minor_locator(minor_locator)
        axes_stick.yaxis.set_minor_locator(minor_locator)
        # hide unneeded spines
        axes_stick.spines['right'].set_visible(False)
        axes_stick.spines['top'].set_visible(False)

        # config subplot for throttle: a 1D vertical plot
        axes_throttle.set_ylabel('throttle [-]')
        axes_throttle.set_ylim(bottom=0, top=1)
        axes_throttle.set_xlim(left=0, right=1)
        axes_throttle.spines['left'].set_position('zero')
        axes_throttle.yaxis.set_label_coords(0.5, 0.5)
        axes_throttle.set_yticks([0, 0.5, 1])
        axes_throttle.yaxis.set_minor_locator(minor_locator)
        axes_throttle.tick_params(axis='y', which='both', direction='inout')
        # hide horizontal x-axis and related spines
        axes_throttle.xaxis.set_visible(False)
        for spine in ['right', 'bottom', 'top']:
            axes_throttle.spines[spine].set_visible(False)

        # config rudder subplot: 1D horizontal plot
        axes_rudder.set_xlabel('rudder [-]')
        axes_rudder.set_xlim(left=-1, right=1)
        axes_rudder.set_ylim(bottom=0, top=1)
        axes_rudder.xaxis.set_label_coords(0.5, -0.5)
        axes_stick.spines['bottom'].set_position('zero')
        axes_rudder.set_xticks([-1, 0, 1])
        axes_rudder.xaxis.set_minor_locator(minor_locator)
        axes_rudder.tick_params(axis='x', which='both', direction='inout')
        axes_rudder.get_yaxis().set_visible(False)  # only want a 1D subplot
        for spine in ['left', 'right', 'top']:
            axes_rudder.spines[spine].set_visible(False)

        all_axes = AxesTuple(axes_state=axes_state,
                             axes_stick=axes_stick,
                             axes_throttle=axes_throttle,
                             axes_rudder=axes_rudder)

        # create figure-wide legend
        cmd_entry = (
            plt.Line2D([], [], color='b', marker='o', ms=10, linestyle='', fillstyle='none'),
            'Commanded Position, normalised')
        pos_entry = (plt.Line2D([], [], color='r', marker='+', ms=10, linestyle=''),
                     'Current Position, normalised')
        figure.legend((cmd_entry[0], pos_entry[0]),
                      (cmd_entry[1], pos_entry[1]),
                      loc='lower center')

        plt.show()
        plt.pause(self.PLOT_PAUSE_SECONDS)  # voodoo pause needed for figure to appear

        return figure, all_axes

    def _prepare_state_printing(self, ax: plt.Axes):
        ys = [self.TEXT_Y_POSN_INITIAL + i * self.TEXT_Y_INCREMENT
              for i in range(len(self.print_props))]

        for prop, y in zip(self.print_props, ys):
            label = str(prop.name)
            ax.text(self.TEXT_X_POSN_LABEL, y, label, transform=ax.transAxes, **(self.LABEL_TEXT_KWARGS))

        # print and store empty Text objects which we will rewrite each plot call
        value_texts = []
        dummy_msg = ''
        for y in ys:
            text = ax.text(self.TEXT_X_POSN_VALUE, y, dummy_msg, transform=ax.transAxes,
                           **(self.VALUE_TEXT_KWARGS))
            value_texts.append(text)
        self.value_texts = tuple(value_texts)

    def _print_state(self, sim: Simulation):
        # update each Text object with latest value
        for prop, text in zip(self.print_props, self.value_texts):
            text.set_text(f'{sim[prop]:.4g}')

    def _plot_control_states(self, sim: Simulation, all_axes: AxesTuple):
        control_surfaces = [prp.aileron_left, prp.elevator, prp.throttle, prp.rudder]
        ail, ele, thr, rud = [sim[control] for control in control_surfaces]
        # plot aircraft control surface positions
        all_axes.axes_stick.plot([ail], [ele], 'r+', mfc='none', markersize=10, clip_on=False)
        all_axes.axes_throttle.plot([0], [thr], 'r+', mfc='none', markersize=10, clip_on=False)
        all_axes.axes_rudder.plot([rud], [0], 'r+', mfc='none', markersize=10, clip_on=False)

    def _plot_control_commands(self, sim: Simulation, all_axes: AxesTuple):
        """
        Plots agent-commanded actions on the environment figure.

        :param sim: Simulation to plot control commands from
        :param all_axes: AxesTuple, collection of axes of subplots to plot on
        """
        ail_cmd = sim[prp.aileron_cmd]
        ele_cmd = sim[prp.elevator_cmd]
        thr_cmd = sim[prp.throttle_cmd]
        rud_cmd = sim[prp.rudder_cmd]

        all_axes.axes_stick.plot([ail_cmd], [ele_cmd], 'bo', mfc='none', markersize=10,
                                 clip_on=False)
        all_axes.axes_throttle.plot([0], [thr_cmd], 'bo', mfc='none', markersize=10, clip_on=False)
        all_axes.axes_rudder.plot([rud_cmd], [0], 'bo', mfc='none', markersize=10, clip_on=False)
