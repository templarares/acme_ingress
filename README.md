Use the dm-acme framework to perform RL on the bit-car-in-out-controller.
Requirements:
-bit-car-in-out-controller (branch IngressMujocoRL) and its requirements
-dm-acme: the RL framework including many typical agents
-gym-ingress: a openai gym-based wrapper for the bit-car-in-out-controller
-mc_mujoco: physics engine
-docker and xmanager: for distributed learning (multi-processing)

to restore from the previous run's checkpoint, add the followling line to the corresponding learning agent's learning.py, in the Learner's init method:
ckpt = '/home/templarares/acme/69dfc886-5d76-11ec-8153-7085c2d3e4e3/checkpoints/d4pg_learner/ckpt-2'
self._checkpointer._checkpoint.restore(ckpt)
