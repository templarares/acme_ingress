# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Acme: Quickstart
# ## Guide to installing Acme and training your first D4PG agent.
# # <a href="https://colab.research.google.com/github/deepmind/acme/blob/master/examples/quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 
# 
# %% [markdown]
# ## Select your environment library
# Note: `dm_control` requires a valid Mujoco license.

# %%
environment_library = 'gym'  # @param ['dm_control', 'gym']


# %% [markdown]
# ## Installation
# %% [markdown]
# ### Install Acme

# %% [markdown]
# ### Install the environment library
# 
# Without a valid license you won't be able to use the `dm_control` environments but can still follow this colab using the `gym` environments.
# 
# If you have a personal Mujoco license (_not_ an institutional one), you may 
# need to follow the instructions at https://research.google.com/colaboratory/local-runtimes.html to run a Jupyter kernel on your local machine.
# This will allow you to install `dm_control` by following instructions in
# https://github.com/deepmind/dm_control and using a personal MuJoCo license.
# 



# %% [markdown]
# ### Install visualization packages

import launchpad as lp
import IPython

from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import sonnet as snt

# Import the selected environment lib
if environment_library == 'dm_control':
  from dm_control import suite
elif environment_library == 'gym':
  import gym
# %% [markdown]
# ## Load an environment
# 
# We can now load an environment. In what follows we'll create an environment and grab the environment's specifications.
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def make_environment(x):  
    environment = gym.make('MountainCarContinuous-v0')
    environment = wrappers.GymWrapper(environment)  # To dm_env interface.
    environment = wrappers.SinglePrecisionWrapper(environment)
    environment.reset()
    return environment



def make_networks(action_spec: specs.BoundedArray):
    num_dimensions = np.prod(action_spec.shape, dtype=int)
    # Create the deterministic policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP((256, 256, 256), activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    # Create the distributional critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP((512, 512, 256), activate_final=True),
        networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
    ])
    return {'policy': policy_network,
        'critic': critic_network,
        'observation': tf2_utils.batch_concat,}




# %%
# Create a logger for the agent and environment loop.
agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

# Create the D4PG agent.
agent = d4pg.DistributedD4PG(
    environment_factory=lambda x: make_environment(x),
    network_factory=make_networks,
    num_actors=5,
    batch_size=128,
    min_replay_size=100,
    max_replay_size=1000,
)


program = agent.build()
lp.launch(program, launch_type='local_mp', terminal='current_terminal')

# %% [markdown]
# ## Run a training loop

# %%
