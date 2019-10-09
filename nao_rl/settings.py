"""
Author: Andrius Bernatavicius, 2018

This file contains:
    1. Paths to the required software
    2. Local directories of this package
    3. Default hyperparameters for RL Algorithms
"""

import os, multiprocessing, random

############################
### Directories

# Directory of VREP v3.4.0
VREP_DIR  = '/home/andrius/thesis/software/V-REP_PRO_EDU_V3_4_0_Linux'
# Directory of Choregraphe (optional)
CHORE_DIR = '/home/andrius/thesis/software/choregraphe/bin'

# Local
MAIN_DIR       = os.path.dirname(os.path.realpath(__file__))
SCENES         = MAIN_DIR + '/scenes'
TRAINED_MODELS = MAIN_DIR + '/trained_models'
DATA           = MAIN_DIR + '/data'

# System parameters
CPU_COUNT = multiprocessing.cpu_count() / 2

#############################
### Addresses and ports

SIM_PORT      = random.randint(19890, 19990)              # Simulation port of VREP
LOCAL_IP      = '127.0.0.1'        # Local IP Address of VREP and NAOQI
NAO_PORT      = 5995               # Naoqi port of simulated NAO
REAL_NAO_PORT = 9559               # The port of real NAO (can be checked in choregraphe)
REAL_NAO_IP   = '192.168.1.175'    # IP address of real nao (can be checked in choregraphe)

#############################
# Optimal parameters for RL Algorithms for the custom environments
# Found through extensive grid search
default_parameters = {}

# Proximal Policy Optimization
default_parameters['ppo_NaoTracking']  = {'n_workers'      : CPU_COUNT,       # Number of parallel workers
                                          'max_episodes'   : 6500,            # Max number of episodes before the training stops
                                          'episode_length' : 2000,            # Maximum length of the episode in steps
                                          'batch_size'     : 250,            # Batch size of experiences for each training occurence
                                          'epochs'         : 10,               # Number of epochs for gradient descent/ADAM
                                          'epsilon'        : .2,             # Clipping value of surogate objective
                                          'gamma'          : .99,             # Future reward discount factor
                                          'actor_layers'   : [50, 50],      # Number of layers and nodes in each layer of the actor network
                                          'critic_layers'  : [50],      # Number of layers and nodes in each layer of the critic network
                                          'actor_lr'       : .001,          # Actor learning rate
                                          'critic_lr'      : .005}          # Critic learning rate
    
default_parameters['ppo_NaoBalancing'] = {'n_workers'      : CPU_COUNT,
                                          'max_episodes'   : 10000,
                                          'episode_length' : 2000,
                                          'batch_size'     : 2000,
                                          'epochs'         : 5,
                                          'epsilon'        : .1,
                                          'gamma'          : .99,
                                          'actor_layers'   : [256, 256],
                                          'critic_layers'  : [256, 256],
                                          'actor_lr'       : .000005,
                                          'critic_lr'      : .00001}

default_parameters['ppo_NaoWalking']   = {}

# Asynchronous Advantage Actor Critic
default_parameters['a3c_NaoTracking']  = {'n_workers'      : 4,
                                          'max_episodes'   : 5000,
                                          'episode_length' : 500,
                                          'update_every'   : 20,
                                          'entropy_beta'   : .02, 
                                          'gamma'          : .99,
                                          'actor_layers'   : [50, 50],
                                          'critic_layers'  : [50, 50],
                                          'actor_lr'       : .0005, 
                                          'critic_lr'      : .001}

default_parameters['a3c_NaoBalancing'] = {'n_workers'      : 4,
                                          'max_episodes'   : 5000,
                                          'episode_length' : 500,
                                          'update_every'   : 25,
                                          'entropy_beta'   : .02, 
                                          'gamma'          : .99,
                                          'actor_layers'   : [250,250],
                                          'critic_layers'  : [250],
                                          'actor_lr'       : .0001, 
                                          'critic_lr'      : .0002}

default_parameters['a3c_NaoWalking']   = {'n_workers'      : 2,
                                          'max_episodes'   : 5000,
                                          'episode_length' : 500,
                                          'update_every'   : 8,
                                          'entropy_beta'   : .02, 
                                          'gamma'          : .99,
                                          'actor_layers'   : [50,50],
                                          'critic_layers'  : [50],
                                          'actor_lr'       : .0001, 
                                          'critic_lr'      : .0002}

default_parameters['a3c_NaoReaching']  = {'n_workers'      : 4,
                                          'max_episodes'   : 5000,
                                          'episode_length' : 200,
                                          'update_every'   : 20,
                                          'entropy_beta'   : .02, 
                                          'gamma'          : .999,
                                          'actor_layers'   : [200, 200],
                                          'critic_layers'  : [200, 200],
                                          'actor_lr'       : .0005, 
                                          'critic_lr'      : .001}