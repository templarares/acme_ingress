import gym
import gym_ingress_mc
import numpy as np
import sys
import threading


myenv=gym.make('Ingress-v0')
myenv.reset()
action=np.ones((12,))*0.01
myenv.step(action)
myenv.reset() 
myenv.step(action)
myenv.step(action)
myenv.reset()
myenv.step(action)
myenv.reset()
myenv.step(action)
myenv.reset()
myenv.step(action)
myenv.reset()
myenv.step(action)
myenv.reset()
myenv.step(action)
myenv.reset()
myenv.step(action)
myenv.reset()
myenv.step(action)