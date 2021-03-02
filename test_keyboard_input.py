import gym
import os
import numpy as np
import gym_jsbsim
import os.path
from pynput import keyboard
import threading

aileron = 0.0
elevator = 0.0
env = gym.make('JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-v0')


def on_release(key):
    global aileron
    global elevator
    if key.char == 'a':
        aileron = aileron - 0.01
    elif key.char == 'd':
        aileron = aileron + 0.01
    elif key.char == 'w':
        elevator = elevator + 0.01
    elif key.char == 's':
        elevator = elevator - 0.01
    elif key == keyboard.Key.esc:
        # Stop listener
        return False


def work():
    global aileron
    global env
    threading.Timer(0.02, work).start()
    action = np.array([aileron, elevator, 0.0])
    state, reward, done, _ = env.step(action)

    print("action =", action, " ---> State =", state, " : Reward =", reward)


def random_agent():
    global env
    listener = keyboard.Listener(
        on_release=on_release)
    listener.start()

    env.reset()
    done = False
    work()


if __name__ == "__main__":
    random_agent()
