"""
Script for running trained models

The custom nao_rl trained models can be run either on virtual NAO in V-REP or 
on the real Robot

Positional arguments:
    1. Tensorflow weights file
    2. []
"""

from argparse import ArgumentParser
import time
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

        print(self.sess.run('A:0'))
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

class RestoredModelA3C(object):
    """
    Contains functions for getting actions from loaded policies
    """

    def __init__(self, name, worker):

        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(name + '.meta')
        saver.restore(self.sess,tf.train.latest_checkpoint(nao_rl.settings.TRAINED_MODELS))
        graph = tf.get_default_graph()
        self.state_input = graph.get_tensor_by_name(worker+"/state_input:0")
        self.choose_action = graph.get_tensor_by_name(worker+"/choose_action/choose_action:0")

    def action(self, state):
        """
        Choose an action based on the state using the loaded policy
        """
        state = state[np.newaxis, :]

        return self.sess.run(self.choose_action, {self.state_input: state})[0]



if __name__ == "__main__":

    # Balancing
    # name = 'NaoBalancing_a3c_2019-01-11_10:52:46.cpkt'
    name = 'NaoBalancing_a3c_2019-01-11_11:50:59.cpkt'
    worker = 'W_3'
    env_name = 'NaoBalancing'
    name = nao_rl.settings.TRAINED_MODELS + '/' + name
    model = RestoredModelA3C(name, worker)

    # # Walking
    # name = 'walking.cpkt'
    # worker = 'W_0'
    # env_name = 'NaoWalking'
    # name = nao_rl.settings.TRAINED_MODELS + '/' + name
    # model = RestoredModelA3C(name, worker)

    # Tracking
    # name = 'NaoTracking_a3c_2019-01-11_12:01:41.cpkt'
    # name = 'NaoTracking_a3c_2019-01-11_13:39:18.cpkt'
    # worker = 'W_1'
    # env_name = 'NaoTracking'
    # name = nao_rl.settings.TRAINED_MODELS + '/' + name
    # model = RestoredModelA3C(name, worker)
    
    
    # Load environment
    env = nao_rl.make(env_name, headless=False)
    fps = 30.

    # Test Loop
    n = 0
    attempts = 10
    while n < attempts:
        total_reward = 0
        steps = 0
        done = False
        state = env.reset()
        # Test loop
        while not done:
            action = np.clip(model.action(state), env.action_space.low, env.action_space.high)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(1/fps)
        n += 1
        print "\nAttempt {}/{} | Total reward: {} | Steps: {}".format(n, attempts, total_reward, steps)

    nao_rl.destroy_instances()
    

