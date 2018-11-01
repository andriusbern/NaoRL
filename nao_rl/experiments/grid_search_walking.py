"""
Author: Andrius Bernatavicius, 2018
"""

from nao_rl.learning import models
from nao_rl.environments import NaoWalking
import nao_rl.settings as s

"""
Hyperparameter search for the NaoWalking environment
"""

if __name__ == "__main__":

    scene = s.SCENES + '/nao_walk.ttt'
    env = NaoWalking(s.LOCAL_IP, s.SIM_PORT, s.NAO_PORT, scene)

    # Parameters
    steps  = 10000 # Number of steps per trial
    trials = 5     # Number of trials for the same set of hyperparams

    # Hyperparameters
    ## Hidden layer sizes
    actor_layers  = [[50, 50], [100, 100], [150, 150],
                     [50, 50, 50], [100, 100, 100], 
                     [50, 50, 50, 50], [100, 100, 100, 100]]
    critic_layers = []
    gamma = []
    lrate = []
    

    models.build_ddpg_model(env, actor_layers, critic_layers, gamma, lrate)