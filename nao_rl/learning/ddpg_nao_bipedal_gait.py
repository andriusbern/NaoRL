import matplotlib.pyplot as plt

# nao_rl
from nao_rl.environments import NaoWalking
import nao_rl.settings as settings
from nao_rl.learning import models

"""
Example file that uses Deep Deterministic Policy Gradient algorithm to train the agent
Hyperparameters are subject to optimization
"""

if __name__ == "__main__":
    
    ENV_NAME = 'NAO_bipedal_gait'
    env = NaoWalking(settings.LOCAL_IP,
                     settings.SIM_PORT, 
                     settings.NAO_PORT, 
                     settings.SCENES + '/nao_test.ttt')
    env.agent.connect_env(env)

    agent = models.build_ddpg_model(env,
                                    actor_hidden_layers=[80,80],
                                    critic_hidden_layers=[100,100], 
                                    gamma=0.99,
                                    learning_rate=0.001)


    # Train
    history = agent.fit(env, nb_steps=200000, visualize=False, verbose=2, nb_max_episode_steps=200)
    filename = settings.TRAINED_MODELS + '/ddpg_{}_weights.h5f'.format(ENV_NAME)
    #agent.save_weights(filename, overwrite=True)
    env.stop_simulation()

    plt.plot(history.history['episode_reward'])
    plt.show()

    # Evaluate
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
