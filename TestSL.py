from dm_env import TimeStep
import gym
import gym_ingress_mc
import numpy as np
import IPython
import matplotlib
import matplotlib.pyplot as plt
from acme import environment_loop
from acme import specs
from acme import wrappers

environment=gym.make('Ingress-v1',visualization=True)
environment = wrappers.GymWrapper(environment)
environment=wrappers.SinglePrecisionWrapper(environment)
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.tf import savers as tf2_savers
from acme.utils import loggers
import sonnet as snt
import pyvirtualdisplay
import imageio
import base64
matplotlib.use('TkAgg')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Grab the spec of the environment.
environment_spec = specs.make_environment_spec(environment)

#@title Build agent networks

# Get total number of action dimensions from action spec.
num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

# Create the shared observation network; here simply a state-less operation.
observation_network = tf2_utils.batch_concat

# Create the deterministic policy network.
policy_network = snt.Sequential([
    networks.LayerNormMLP((256, 256, 256), activate_final=True),
    networks.NearZeroInitializedLinear(num_dimensions),
    networks.TanhToSpec(environment_spec.actions),
])

#copy policy network from a checkpoint
checkpointer = tf2_savers.Checkpointer(
    subdirectory='d4pg_learner',
    objects_to_save={
        'policy': policy_network,
    })
" this will restore from a previous checkpoint; replace names accordingly"
ckpt = '/home/templarares/acme/f997802c-5d99-11ec-903a-7085c2d3e4e3/checkpoints/d4pg_learner/ckpt-8'
checkpointer._checkpoint.restore(ckpt).expect_partial()

# Create the distributional critic network.
critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP((512, 512, 256), activate_final=True),
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
    sigma=0.7,
    checkpoint=False
)


#learning completed. Now play the result
timestep = environment.reset()
reward=0
while not timestep.last():
  # Simple environment loop.
  action = agent.select_action(timestep.observation)
  timestep = environment.step(action)
  print("reward is: ",timestep.reward)
  print("action is: ", action)
  reward+=timestep.reward
print("final reward is",reward)
