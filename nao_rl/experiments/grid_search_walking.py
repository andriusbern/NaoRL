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
    env = NaoWalking(s.LOCAL_IP,
                     s.SIM_PORT, 
                     s.NAO_PORT, 
                     scene)

    # Parameters
    steps  = 20000 # Number of steps per trial
    trials = 3     # Number of trials for the same set of hyperparams

    # Hyperparameters
    ## Hidden layer sizes
    actor_layers  = [[50, 50], [100, 100], [150, 150],
                     [50, 50, 50], [100, 100, 100], 
                     [50, 50, 50, 50], [100, 100, 100, 100]]
    critic_layers = [[50, 50], [100, 100], [150, 150],
                     [50, 50, 50], [100, 100, 100],
                     [50, 50, 50, 50], [100, 100, 100, 100]]
    gamma = []
    lrate = []
    

    model = models.build_ddpg_model(env, actor_layers, critic_layers, gamma, lrate)
    history = model.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=200)

    # Save the weights

    model.save_weights(s.SCENES + 'ddpg_{}_weights.h5f', overwrite=True)
    env.stop_simulation()