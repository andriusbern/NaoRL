import matplotlib.pyplot as plt

# nao_rl
from nao_rl.environments import NaoWalking
import nao_rl.settings as settings
from nao_rl.learning import models

import nao_rl

"""
Example file that uses Deep Deterministic Policy Gradient algorithm to train the agent
Hyperparameters are subject to optimization
"""

if __name__ == "__main__":
    
    ENV_NAME = 'nao_bipedal'
    
    env = nao_rl.make(ENV_NAME, settings.SIM_PORT, headless=False, reinit=True)
    
    env.agent.connect(env, env.active_joints)

    model = models.build_ddpg_model(env,
                                    actor_hidden_layers=[80,80],
                                    critic_hidden_layers=[100,100], 
                                    gamma=0.99,
                                    learning_rate=0.001)


    # Train
    history = model.fit(env, nb_steps=200000, visualize=False, verbose=2, nb_max_episode_steps=200)
    filename = settings.TRAINED_MODELS + '/ddpg_{}_weights.h5f'.format(ENV_NAME)
    #agent.save_weights(filename, overwrite=True)
    env.stop_simulation()

    plt.plot(history.history['episode_reward'])
    plt.show()

    # Evaluate
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
