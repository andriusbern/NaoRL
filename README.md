# nao_rl  -  Reinforcement Learning Package For *NAO* Robot.
This python package integrates *V-REP* robot simulation software, base libraries for NAO robot control along with reinforcement learning algorithms for solving custom or any *OpenAI-gym*-based learning environments.

## Features:
1. State-of-the-art policy gradient RL algorithms for training the agent - Proximal Policy Optimization (PPO) and Asynchronous Advantage Actor-Critic (A3C) both of which are parallelized, scale to any number of workers and drastically increase the training speed.
2. Custom OpenAI-gym-based API for controlling *V-REP* that makes it easy to create new learning tasks and environments (50 - 100 LOC)
3. Reinforcement learning can be done both in simulation or using the real robot.
4. Policies learned in simulated environments can be directly tested on the real NAO robot and vice-versa.
5. Grid search scripts for hyperparameter optimization.
6. Custom learning environments for the NAO robot:
   

  **1. Balancing / Learning a bipedal gait**         | **2. Object tracking**
:-------------------------:|:-------------------------:
 <img src="assets/ezgif.com-gif-maker.gif" width="600">   | <img src="assets/untitled.gif" width="500">
 The goal is to keep an upright position without falling or learn how to move forward | The goal is to keep the object within the visual field by moving the head motors. 



# Requirements

### **Base** (for learning in simulated environments with virtual NAO):
1. [VREP v3.4.0](http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_4_0_Linux.tar.gz) - Robot simulation software by Copellia Robotics
2. *Python 2.7* and *python-virtualenv*. 
3. *tensorflow, gym, numpy, opencv-python*
    

### **Optional** (if you have access to the NAO robot):
1. [Choregraphe Suite v2.1.2](https://community.ald.softbankrobotics.com/en/dl/ZmllbGRfY29sbGVjdGlvbl9pdGVtLTQyNS1maWVsZF9zb2Z0X2RsX2V4dGVybmFsX2xpbmstMC01MzY2MjU%3D?width=500&height=auto) - for creating a virtual NAO (**requires registering**) (By default installed in */opt/Aldebaran/Choregraphe Suite 2.1*)

2. [Python NAOqi SDK v2.1.2](https://community.ald.softbankrobotics.com/en/dl/ZmllbGRfY29sbGVjdGlvbl9pdGVtLTQ1My1maWVsZF9zb2Z0X2RsX2V4dGVybmFsX2xpbmstMC1lZWFmOGU%3D?width=500&height=auto) - Libraries provided by Softbank robotics for NAO control (**requires registering**)

# Installation
*(Tested on Ubuntu 18.04)*

**1. Clone the repository**
   
```
git clone https://github.com/andriusbern/nao_rl
cd nao-rl
```
**2. Create and activate the virtual environment**
   
```
virtualenv env
source env/bin/activate
```

**3. Install the package and the required libraries**
```
python setup.py install
```
You will be prompted to enter the path to your V-Rep installation directory

**4. Launch the virtual NAO and VREP.**

# Testing the environments
To try the environments out (V-Rep will be launched with the appropriate scene and agent loaded, actions will be sampled randomly):
```
import nao_rl
env = nao_rl.make('env_name')
env.run()
```
Where 'env_name' corresponds to one of the following available environments:
1. NaoTracking  - tracking an object using the camera information
2. NaoBalancing - keeping upright balance
3. NaoWalking   - learning a bipedal gait

# Training

To train the agents in these environments you can use build-in RL algorithms:

```
python train.py NaoTracking a3c 0
```
<img src="assets/live_plot.gif"> 

Live plotting of training results, sped up 40x (use flag '-p' to enable live plotting).

Positional arguments:
1. Environment name: 
2. Training algorithm: 
   1. *--a3c*    - Asynchronous Advantage Actor-Critic or 
   2. *--ppo*    - Proximal Policy Optimization
3. Rendering mode : 
   1. [0] - Do not render
   2. [1] - Render the first worker
   3. [2] - Render all workers

To find out more about additional command line arguments:
```
python train.py -h
```

The training session can be interrupted at any time and the model is going to be saved and can be loaded later.

# Testing trained models

To test trained models:
```
python test.py trained_models/filename.cpkt 
```
Add *-r* flag to run the trained policy on the real NAO (can be dangerous). It is recommended to set low fps for the environment e.g. (the robot will perform actions slowly):
```
python test.py trained_models/filename.cpkt -r -fps 2
```



