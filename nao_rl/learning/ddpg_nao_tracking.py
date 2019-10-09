import os
import numpy as np
import gym

# keras-rl
from rl.agents import DDPGAgent

# nao_rl
from nao_rl.environments import NaoTracking
import nao_rl.settings as settings
from nao_rl.learning import models


if __name__ == "__main__":

    ENV_NAME = 'NAO_tracking'
    scene = settings.SCENES + '/nao_ball.ttt'

    # Environment and objects
    env = NaoTracking(settings.LOCAL_IP,
                      settings.SIM_PORT,
                      settings.NAO_PORT,
                      scene)

    env.agent.connect_env(env)
    env.ball.connect_env(env)

    agent = models.build_ddpg_model(env,
                                    actor_hidden_layers=[16,16,16],
                                    critic_hidden_layers=[32,32,32], 
                                    gamma=0.99, 
                                    learning_rate=0.001)

    # Train
    h = agent.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=200)

    # Save the weights

    agent.save_weights(settings.SCENES + 'ddpg_{}_weights.h5f', overwrite=True)
    env.stop_simulation()
    # Evaluate
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
