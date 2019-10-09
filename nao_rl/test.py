"""
Script for running trained models

The custom nao_rl trained models can be run either on virtual NAO in V-REP or 
on the real Robot

Positional arguments:
    1. Tensorflow weights file
    2. []
"""

import argparse
import json
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
            self.state_input   = graph.get_tensor_by_name('W_0/state_input:0')
            self.choose_action = graph.get_tensor_by_name('W_0/choose_action/choose_action:0')
        elif alg == 'ppo':
            self.state_input   = graph.get_tensor_by_name("state_input:0")
            self.choose_action = graph.get_tensor_by_name("choose_action:0")
        print('Model loaded successfully.')

    def action(self, state):
        """
        Choose an action based on the state using the loaded policy
        """
        state = state[np.newaxis, :]
        print state

        return self.sess.run(self.choose_action, {self.state_input: state})[0]

def find_best():
    pass

def load_logs(filename):

    print file_path
    with open(file_path) as log_file:
        
        data = json.load(log_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser() # Parse command line arguments
    parser.add_argument('filename', type=str, help='A tensorflow ".cpkt" file with stored tensors and operations.')
    parser.add_argument('-e', '--environment', type=str, help='In case the ".cpkt" file was renamed after saving, you need to specify the environment.')
    parser.add_argument('-a', '--algorithm', type=str, help='In case the ".cpkt" file was renamed after saving you need to specify the learning algorithm that was used.', choices=['a3c', 'ppo'])
    parser.add_argument('-n', '--n_attempts', nargs='?', type=int, const=10, help='Number of episodes to repeat the policy. Default = 10')
    parser.add_argument('-r', '--real', action='store_true', help='Use the real NAO (IP and PORT specified in settings.py)')
    args = parser.parse_args()

    # Get command line arguments
    filename   = args.filename.split('/')[-1]
    
    if args.algorithm is not None:
        algorithm = args.algorithm
    else:
        algorithm = filename.split('_')[1]

    if args.environment is not None:
        env_name = args.environment
    else:
        env_name = filename.split('_')[0]

    if args.n_attempts is not None:
        n_attempts = args.n_attempts
    else:
        n_attempts = 10

    # Load model
    file_path = nao_rl.settings.TRAINED_MODELS + '/' + filename
    print "\nLoading {}".format(file_path)
    model = RestoredModel(file_path, algorithm)

    # Load a log file
    try:
        log = nao_rl.utils.Log()
    
        
        log.load_from_file(nao_rl.settings.DATA + filename.split('.')[0]+'.log')
        log.summary()
    except:
        print "Could not load the log file from '/data' directory"

    # Balancing
    # name = 'NaoBalancing_a3c_2019-01-11_10:52:46.cpkt'
    # name = 'NaoBalancing_a3c_2019-01-11_11:50:59.cpkt'
    # # Walking
    # name = 'walking.cpkt'
    # Tracking
    # name = 'NaoTracking_a3c_2019-01-11_12:01:41.cpkt'

    # Create environment
    env = nao_rl.make(env_name, headless=False)
    fps = 30.
    # Test Loop
    n = 0
    while n < n_attempts:
        total_reward = 0
        steps = 0
        done = False
        state = env.reset()
        # Test loop
        while not done:
            raw_input('ENTER TO CONTINUE...')
            action = np.clip(model.action(state), env.action_space.low, env.action_space.high)
            # action = env.f()
            state, reward, done, _ = env.step(np.array(action))
            total_reward += reward
            steps += 1
            time.sleep(1/fps)
            print(action)

        n += 1

    nao_rl.destroy_instances()
    

