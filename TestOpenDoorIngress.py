from tabnanny import verbose
import gym
import gym_ingress_mc
import gym_opendoor_mc
import numpy as np
import sys
import threading


myenv=gym.make('OpenDoor-v4',visualization=True, verbose=True)
myenv.reset()
action=np.zeros((8,))*0.01
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)