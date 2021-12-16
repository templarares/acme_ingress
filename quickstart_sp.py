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
from acme.tf import savers as tf2_savers

# Import the selected environment lib
if environment_library == 'dm_control':
  from dm_control import suite
elif environment_library == 'gym':
  import gym

# Imports required for visualization
import pyvirtualdisplay
import imageio
import base64

# Set up a virtual display for rendering.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# %% [markdown]
# ## Load an environment
# 
# We can now load an environment. In what follows we'll create an environment and grab the environment's specifications.

# %%
if environment_library == 'dm_control':
  environment = suite.load('cartpole', 'balance')
  
elif environment_library == 'gym':
  environment = gym.make('MountainCarContinuous-v0')
  environment = wrappers.GymWrapper(environment)  # To dm_env interface.

else:
  raise ValueError(
      "Unknown environment library: {};".format(environment_library) +
      "choose among ['dm_control', 'gym'].")


# Make sure the environment outputs single-precision floats.
environment = wrappers.SinglePrecisionWrapper(environment)

# Grab the spec of the environment.
environment_spec = specs.make_environment_spec(environment)

# %% [markdown]
#  ## Create a D4PG agent

# %%
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

# Create the distributional critic network.
critic_network = snt.Sequential([
    # The multiplexer concatenates the observations/actions.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP((512, 512, 256), activate_final=True),
    networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
])
#copy policy network from a checkpoint
checkpointer = tf2_savers.Checkpointer(
    subdirectory='d4pg_learner',
    objects_to_save={
        'policy': policy_network,
    })
" this will restore from a previous checkpoint; replace names accordingly"
ckpt = '/home/templarares/acme/50b4386c-5e23-11ec-9abf-7085c2d3e4e3/checkpoints/d4pg_learner/ckpt-2'
checkpointer._checkpoint.restore(ckpt).expect_partial()


# %%
# Create a logger for the agent and environment loop.
agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

# Create the D4PG agent.
agent = d4pg.D4PG(
    environment_spec=environment_spec,
    policy_network=policy_network,
    critic_network=critic_network,
    observation_network=observation_network,
    sigma=1.0,
    logger=agent_logger,
    checkpoint=False
)

# # Create an loop connecting this agent to the environment created above.
# env_loop = environment_loop.EnvironmentLoop(
#     environment, agent, logger=env_loop_logger)

# # %% [markdown]
# # ## Run a training loop

# # %%
# # Run a `num_episodes` training episodes.
# # Rerun this cell until the agent has learned the given task.
# env_loop.run(num_episodes=100)

# %% [markdown]
# ## Visualize an evaluation loop
# 
# %% [markdown]
# ### Helper functions for rendering and vizualization

# %%
# Create a simple helper function to render a frame from the current state of
# the environment.
if environment_library == 'dm_control':
  def render(env):
    return env.physics.render(camera_id=0)
elif environment_library == 'gym':
  def render(env):
    return env.environment.render(mode='rgb_array')
else:
  raise ValueError(
      "Unknown environment library: {};".format(environment_library) +
      "choose among ['dm_control', 'gym'].")

def display_video(frames, filename='temp.mp4'):
  """Save and display video."""

  # Write video
  with imageio.get_writer(filename, fps=60) as video:
    for frame in frames:
      video.append_data(frame)

  # Read video and display the video
  video = open(filename, 'rb').read()
  b64_video = base64.b64encode(video)
  video_tag = ('<video  width="320" height="240" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())

  return IPython.display.HTML(video_tag)

# %% [markdown]
# ### Run and visualize the agent in the environment for an episode

# %%
timestep = environment.reset()
frames = [render(environment)]

while not timestep.last():
  # Simple environment loop.
  action = agent.select_action(timestep.observation)
  timestep = environment.step(action)

  # Render the scene and add it to the frame stack.
  frames.append(render(environment))

# Save and display a video of the behaviour.
display_video(np.array(frames))


# %%



