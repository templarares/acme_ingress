import gym
import gym_ingress_mc
import numpy as np
import sys
import threading


myenv=gym.make('Ingress-v1',visualization=True)
myenv.reset()
action=np.ones((12,))*0.01
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