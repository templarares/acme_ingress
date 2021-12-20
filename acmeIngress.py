
import gym
import gym_ingress_mc
import numpy as np
environment=gym.make('Ingress-v1',visualization=True)
environment.reset()
foo=np.zeros((8,))
environment.step(foo)
environment.reset()

import IPython
import matplotlib
import matplotlib.pyplot as plt
from acme import environment_loop
from acme import specs
from acme import wrappers
from dm_env import TimeStep
environment = wrappers.GymWrapper(environment)
environment=wrappers.SinglePrecisionWrapper(environment)
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import sonnet as snt


# Grab the spec of the environment.
environment_spec = specs.make_environment_spec(environment)

#@title Build agent networks

# Get total number of action dimensions from action spec.
num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

# Create the shared observation network; here simply a state-less operation.
observation_network = tf2_utils.batch_concat

# Create the deterministic policy network.
policy_network = snt.Sequential([
    networks.LayerNormMLP((256*2, 256*2, 256*2), activate_final=True),
    networks.NearZeroInitializedLinear(num_dimensions),
    networks.TanhToSpec(environment_spec.actions),
])

# Create the distributional critic network.
critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP((512*2, 512*2, 256*2), activate_final=True),
    networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
])
# Create a logger for the agent and environment loop.
def myprint(content):
    print(content)
agent_logger = loggers.TerminalLogger(label='agent', print_fn=myprint,time_delta=0)
env_loop_logger = loggers.TerminalLogger(label='env_loop', print_fn=myprint,time_delta=0)

# Create the D4PG agent.
agent = d4pg.D4PG(
    environment_spec=environment_spec,
    policy_network=policy_network,
    critic_network=critic_network,
    observation_network=observation_network,
    sigma=0.0,
    n_step=5,
    checkpoint=True
)



# # Create an loop connecting this agent to the environment created above.
# env_loop = environment_loop.EnvironmentLoop(
#     environment, agent, logger=env_loop_logger)

# Run a `num_episodes` training episodes.
# Rerun this cell until the agent has learned the given task.clear
# numEpi=100
# env_loop.run(num_episodes=numEpi)
# fig=plt.figure()
# ax=plt.axes()
# x=range(numEpi)
# ax.scatter(x,env_loop.rewardHistory())
# plt.savefig('foo.png')
# plt.show()

#learning completed. Now play the result
actions=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
actions[0]=np.array([ 0.9999881, -1.,         0.9672146,  1.,        -0.9999963, -1.,  1.,        -1.       ])
actions[1]=np.array([ 0.88788724, -0.99999374,  0.99999547,  0.9865413,   0.86814463, -1.,  1.,         -0.9999921 ])
actions[2]=np.array([ 0.8646394,  -0.9995093,   0.99999297,  0.7440603,   0.95202136, -0.99999374,  0.9995363,  -0.9999802 ])
actions[3]=np.array([ 0.22796643, -0.4667759,   0.9974493,  -0.99855936,  0.95768666,  0.9927311, -0.9999314,  -0.9636852 ])
actions[4]=np.array([ 0.7931547,  -0.9987682,  -0.7157223,   0.98805237,  0.5544684,   0.5111309, -1.,          0.9846978 ])
actions[5]=np.array([ 0.46054387, -0.51824796,  0.65468884, -0.40243268,  0.9993745,   0.9930835, -0.99999064, -0.66709644])
actions[6]=np.array([-0.9977446,  -0.99938774, -0.99196714,  0.12371182,  0.9984739,   0.94080627, -0.9982417,  0.87828875])
for i in range(20):
    timestep = environment.reset()
    reward=0
    j=0
    while not timestep.last():
    # Simple environment loop.
        #action = agent.select_action(timestep.observation)
        action=actions[j]

        timestep = environment.step(action)
        j+=1

        #   print("reward is: ",timestep.reward)
        #   print("action is: ", action)
        reward+=timestep.reward
    print("final reward is",reward)
