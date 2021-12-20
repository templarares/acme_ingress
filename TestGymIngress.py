import gym
import gym_ingress_mc
import numpy as np
import sys
import threading


myenv=gym.make('Ingress-v1',visualization=True)
myenv.reset()
action=np.zeros((8,))*0.01
actions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
actions[0]=np.array([ 0.9999881, -1.,         0.9672146,  1.,        -0.9999963, -1.,  1.,        -1.       ])
actions[1]=np.array([ 0.88788724, -0.99999374,  0.99999547,  0.9865413,   0.86814463, -1.,  1.,         -0.9999921 ])
actions[2]=np.array([ 0.8646394,  -0.9995093,   0.99999297,  0.7440603,   0.95202136, -0.99999374,  0.9995363,  -0.9999802 ])
actions[3]=np.array([ 0.22796643, -0.4667759,   0.9974493,  -0.99855936,  0.95768666,  0.9927311, -0.9999314,  -0.9636852 ])
actions[4]=np.array([ 0.7931547,  -0.9987682,  -0.7157223,   0.98805237,  0.5544684,   0.5111309, -1.,          0.9846978 ])
actions[5]=np.array([ 0.46054387, -0.51824796,  0.65468884, -0.40243268,  0.9993745,   0.9930835, -0.99999064, -0.66709644])
actions[6]=np.array([-0.9977446,  -0.99938774, -0.99196714,  0.12371182,  0.9984739,   0.94080627, -0.9982417,  0.87828875])
myenv.step(actions[0])
myenv.step(actions[1])
myenv.step(actions[2])
myenv.step(actions[3])
myenv.step(actions[4])
myenv.step(actions[5])
myenv.step(actions[6])
myenv.step(action)
myenv.step(action)
myenv.step(action)