import launchpad as lp
from dm_env import TimeStep

import numpy as np
import IPython
import matplotlib
import matplotlib.pyplot as plt
from acme import environment_loop
from acme import specs
from acme import wrappers


from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import sonnet as snt
# import pyvirtualdisplay
# import imageio
# import base64
# matplotlib.use('TkAgg')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def make_environment(x):
    import gym
    import gym_ingress_mc
    """Visulisation still not working, as visulized environment has to be created before acme.tf imports
    when visulization works, let the argument visulization=x instead of always False"""
    environment=gym.make('Ingress-v0',visualization=False)
    environment = wrappers.GymWrapper(environment)
    environment=wrappers.SinglePrecisionWrapper(environment)
    environment.reset()
    return environment

def make_networks(action_spec: specs.BoundedArray):
    num_dimensions = np.prod(action_spec.shape, dtype=int)
    # Create the deterministic policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP((256*2, 256*2, 256*2), activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    # Create the distributional critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP((512*2, 512*2, 256*2), activate_final=True),
        networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
    ])
    return {'policy': policy_network,
        'critic': critic_network,
        'observation': tf2_utils.batch_concat,}




# Create a logger for the agent and environment loop.
def myprint(content):
    print(content)
agent_logger = loggers.TerminalLogger(label='agent', print_fn=myprint,time_delta=0)
env_loop_logger = loggers.TerminalLogger(label='env_loop', print_fn=myprint,time_delta=0)

# Create the D4PG agent.
agent = d4pg.DistributedD4PG(
    environment_factory=lambda x: make_environment(x),
    network_factory=make_networks,
    num_actors=8,
    batch_size=128,
    min_replay_size=100,
    max_replay_size=1000,
)

program = agent.build()
lp.launch(program, launch_type='local_mp', terminal='current_terminal')

# # Create an loop connecting this agent to the environment created above.
# env_loop = environment_loop.EnvironmentLoop(
#     environment, agent, logger=env_loop_logger)

# # Run a `num_episodes` training episodes.
# # Rerun this cell until the agent has learned the given task.clear
# numEpi=100
# env_loop.run(num_episodes=numEpi)
# fig=plt.figure()
# ax=plt.axes()
# x=range(numEpi)
# ax.scatter(x,env_loop.rewardHistory())
# plt.savefig('foo.png')
# plt.show()

# # learning completed. Now play the result
# timestep = environment.reset()
# reward=0
# while not timestep.last():
#   Simple environment loop.
#   action = agent.select_action(timestep.observation)
#   timestep = environment.step(action)
#   print("reward is: ",timestep.reward)
#   print("action is: ", action)
#   reward+=timestep.reward
# print("final reward is",reward)
