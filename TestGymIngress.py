from tabnanny import verbose
import gym
import gym_ingress_mc
import numpy as np
import sys
import threading


myenv=gym.make('Ingress-v2',visualization=False, verbose=False)
# myenv.reset()
action=np.zeros((8,))*0.01
actions=[0,0,0,0,0,0,0,0,0,0,0]
actions[0]=np.array([ 0.99995923, 0.9338989,   0.99977946,  0.99999285, -0.9070116,  -0.2856775, -0.9696906,  -0.9970446 ])
actions[1]=np.array([ 0.9992894,   0.93387866,  0.9603152,   0.998677,   -0.5209597,  -0.74616534,  0.46032286,  0.17958915])
actions[2]=np.array([ 0.9993305,0.9651247,   0.8924601,   0.9972912,  -0.20317942, -0.7491334,  0.7389246,   0.88815534])
actions[3]=np.array([ 0.9972887,   0.9324932,   0.7619629,   0.9878526,  -0.11220986, -0.6450212,  0.85745406,  0.95566857])
actions[4]=np.array([ 0.94795585, 0.7312763,   0.3731073,   0.738276,   -0.23591942, -0.40059793,  0.9652891,   0.9665034 ])
actions[5]=np.array([ 0.29161644,  0.30192745, -0.39103943, -0.89629465, -0.40922785,  0.13132942,  0.9924003,   0.9825382 ])
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.step(action)
myenv.reset()
myenv.step(action)
myenv.step(action)
myenv.step(action)
# myenv.step(action)
# myenv.step(action)