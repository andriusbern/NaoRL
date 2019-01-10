"""
Script for running trained models

The custom nao_rl trained models can be run either on virtual NAO in V-REP or 
on the real Robot

Positional arguments:
    1. Tensorflow weights file
    2. []
"""

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import nao_rl
import gym


class RestoredModel(object):
    """
    Contains functions for getting actions from loaded policies
    """

    def __init__(self, name):

        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(name + '.meta')
        saver.restore(self.sess,tf.train.latest_checkpoint(nao_rl.settings.TRAINED_MODELS))

        # Variables
        graph = tf.get_default_graph()
        self.state_input = graph.get_tensor_by_name("state_input:0")
        self.choose_action = graph.get_tensor_by_name("choose_action:0")

    def action(self, state):
        """
        Choose an action based on the state using the loaded policy
        """
        state = state[np.newaxis, :]

        return self.sess.run(self.choose_action, {self.state_input: state})[0]


if __name__ == "__main__":

    name = 'NaoBalancing_ppo_2019-01-07_00:33:26.cpkt'
    env = name.split('_')[0]
    name = nao_rl.settings.TRAINED_MODELS + '/' + name


    model = RestoredModel(name)

    # Load environment
    env = nao_rl.make(env, headless=False)

    n = 0
    while n < 10:
        total_reward = 0
        done = False
        state = env.reset()
        # Test loop
        while not done:
            action = np.clip(model.action(state), env.action_space.low, env.action_space.high)
            state, reward, done, _ = env.step(action)
            print "Reward: {}".format(reward)
            total_reward += reward
        n += 1
        print "\nTotal reward: {}".format(total_reward)


    

