import os
import numpy as np
import gym

# Keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

# keras-rl
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# nao_rl
from nao_rl.environments import NaoWalking
import nao_rl.settings as settings


"""
Example file that uses Deep Deterministic Policy Gradient algorithm to train the agent
Hyperparameters are subject to optimization
"""

if __name__ == "__main__":
    
    ENV_NAME = 'NAO_bipedal_gait'
    scene = settings.SCENES + '/nao_walk.ttt'

    # Environment and objects
    env = NaoWalking(settings.LOCAL_IP,
                     settings.SIM_PORT, 
                     settings.NAO_PORT, 
                     scene)

    env.agent.connect_env(env)

    # The number of actions.
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Actor model
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(96))
    actor.add(Activation('relu'))
    actor.add(Dense(96))
    actor.add(Activation('relu'))
    actor.add(Dense(96))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    # Critic model
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(192)(x)
    x = Activation('relu')(x)
    x = Dense(192)(x)
    x = Activation('relu')(x)
    x = Dense(192)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())


    # Compile the agent
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.2, mu=0.1, sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                    random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    # Train
    h = agent.fit(env, nb_steps=200, visualize=False, verbose=2, nb_max_episode_steps=200)
    file = settings.TRAINED_MODELS + '/ddpg_{}_weights.h5f'.format(ENV_NAME)
    agent.save_weights(file, overwrite=True)
    env.stop_simulation()
    print h
    # Evaluate
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
