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
from nao_rl.environments import NaoTracking
import nao_rl.settings as settings


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

    # The number of actions.
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Actor model
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    # Critic model
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())


    # Compile the agent
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                    memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                    random_process=random_process, gamma=.9999, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    # Train
    agent.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=200)

    # Save the weights
    parent_dir = os.path.abspath(__file__ + "/../../")
    path = parent_dir + '/trained_models/' + 'ddpg_{}_weights.h5f'.format(ENV_NAME)
    
    agent.save_weights(parent_dir + 'ddpg_{}_weights.h5f', overwrite=True)
    env.stop_simulation()
    # Evaluate
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
