"""
Author: Andrius Bernatavicius, 2018
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

# keras-rl
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# Wrapper functions for building keras models

def build_actor(env, hidden_layer_sizes, activation='relu'):

    actor = Sequential()
    # Input layer
    actor.add(Flatten(input_shape=(1,) + env.observation_space))
    # Create hidden layers
    for size in hidden_layer_sizes:
        actor.add(Dense(size))
        actor.add(Activation(activation))
    # Output layer
    actor.add(Dense(env.action_space.shape[0]))
    actor.add(Activation('linear'))

    return actor

def build_critic(env, hidden_layer_sizes, activation='relu'):
    action_input = Input(shape=(env.action_space.shape[0]), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')

    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])

    # Hidden layers
    for size in hidden_layer_sizes:
        x = Dense(size)(x)
        x = Activation(activation)(x)
    # Output layer
    x = Dense(1)(x)
    x = Activation('linear')(x)

    critic = Model(inputs=[action_input, observation_input], outputs=x)

    return critic, action_input


def build_ddpg_model(env, 
                    actor_hidden_layers,
                    critic_hidden_layers,
                    gamma,
                    learning_rate):
    
    actor = build_actor(env, actor_hidden_layers)
    critic, action_input = build_critic(env, critic_hidden_layers)

    nb_actions = env.action_space.shape[0]
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.2, mu=0.1, sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                    random_process=random_process, gamma=.99, target_model_update=1e-3)

    agent.compile(Adam(lr=learning_rate, clipnorm=1.), metrics=['mae'])

    return agent

