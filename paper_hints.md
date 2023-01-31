# Paper hint 
## Abstract
## Introduction
### Reinforcement Learning:
Main goal: given an observation o that provide a partial description of the state s of the environment, the agent must be able to take an action according to a specific policy {\pi} that tries to maximize its cumulative reward over time. The reward is, in general, a number that tells to the agent how good or bad the current state is. 

### Sim-To-Real:
The robotic control policy learning process could be accelerated by performing it in a physics simulator. In fact, this technique allows to perform analysis in a faster, more scalable and lower-cost way compared to physical robots. 
This is more useful in the context of deep reinforcement learning, for which some applications could be impratical for different reasons: random exploration, collecting huge amount of data, damage of physical hardware. 
Ideally, we could learn policies entirely in simulation and successfully run those on physical robots.
The main challenge to overcome to have reasonable results in real environments is to bridge the reality gap, that is the discrepancy between physics simulators and real world. 

### Uniform Domain Randomization:
We have explored the Uniform Domain Randomization, that consists in training not in only one simulated environment, but in a wide range by varying the robot’s physical parameters in the training phase. 
This method is exploited simply by sampling at each training episode a value from different uniform distributions (one for each masses of the robot) and assign it to the masses of robot link. 

### Soft Actor-Critic:
The SAC algorithm aims to optimize a stochastic policy in an off-policy way. 
A central feature of SAC is the entropy regularization. The policy is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy. This has a close connection to the exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate learning later on. It can also prevent the policy from prematurely converging to a bad local optimum.

### Vision-based Reinforcement Learning:
Another method applied rely on passing raw images as state observation to the agent instead of low-dimensional configuration vector. As underlying policy structure, a convolutional neural network has been used to extract the features of the current observation. 

All our experiments has been performed in the Gym Hopper environment. The hopper is a two-dimensional one-legged figure that consist of four main body parts - the torso at the top, the thigh in the middle, the leg in the bottom, and a single foot on which the entire body rests. Under the hood, we have Mujoco, a physical engine with a Python interface to model the robot.
[OPTIONAL: insert a parameter description of the environment, like action and observation space].