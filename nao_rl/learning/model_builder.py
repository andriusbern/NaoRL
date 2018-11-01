"""
Author: Andrius Bernatavicius, 2018
"""


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

# Wrapper functions for building keras models


def build_actor(observation_space, action_space, hidden_layer_sizes, activation='relu'):

    actor = Sequential()
    # Input layer
    actor.add(Flatten(input_shape=(1,) + observation_space))
    # Create hidden layers
    for size in hidden_layer_sizes:
        actor.add(Dense(size))
        actor.add(Activation(activation))
    # Output layer
    actor.add(Dense(action_space.shape[0]))
    actor.add(Activation('linear'))

    return actor

def build_critic(observation_space, action_space, hidden_layer_sizes, activation='relu'):
    action_input = Input(shape=(action_space.shape[0]), name='action_input')
    observation_input = Input(shape=(1,) + observation_space.shape, name='observation_input')

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

    return critic


