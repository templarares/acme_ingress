import gym
import gym_ingress_mc
import numpy as np
environment=gym.make('Ingress-v2',visualization=True,verbose=True)
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
    n_step=8,
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
totalReward=0
for i in range(50):
    timestep = environment.reset()
    reward=0
    j=0
    while not timestep.last():
    # Simple environment loop.
        action = agent.select_action(timestep.observation)
        timestep = environment.step(action)
        #print("reward is: ",timestep.reward)
        #print("action is: ", action)
        reward+=timestep.reward
    print("final reward is",reward)
    totalReward+=reward
print("total reward is", totalReward)
