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
    Loads an A3C or PPO model from '.cpkt' files and restores the trained weights of tensors that produce actions
    """

    def __init__(self, name, alg):

        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(name + '.meta')
        saver.restore(self.sess, name)
        tf.train
        graph = tf.get_default_graph()
        if alg == 'a3c':
            self.state_input = graph.get_tensor_by_name('W_0/state_input:0')
            self.choose_action = graph.get_tensor_by_name('W_0/choose_action/choose_action:0')
        elif alg == 'ppo':
            self.state_input = graph.get_tensor_by_name("state_input:0")
            self.choose_action = graph.get_tensor_by_name("choose_action:0")
        

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
    alg = 'a3c'
    model = RestoredModel(name, alg)

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
        
        if alg == 'a3c':
            n += 1

    nao_rl.destroy_instances()
    

