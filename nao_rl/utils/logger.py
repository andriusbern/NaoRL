"""
@author: Andrius Bernatavicius, 2018
"""

import nao_rl
import numpy as np
import json

class Log(object):
    """
    Used for storing results of training
    """
    def __init__(self):
        self.date     = None
        self.filename = None
        self.data     = None
        self.parameters = {}
    
    def load_from_file(self, filename):
        """
        Loads the data from a .log file created during the training process
        """
        self.date = filename.split('.')[0].split('_')[-1]
        filename  = filename.split('.')[0] + '.log'
        filename  = filename.split('/')[-1]
        log_path  = nao_rl.settings.DATA + '/' + filename

        with open(filename) as log_file:
            self.data = json.load(log_file)

    def create_from_model(self, model, parameters):
        """
        Creates a .log 
        """
        self.date = model.date
        # Create a training log
        log = parameters.copy()
        log['algorithm'] = model.algorithm
        log['env'] = model.env_name
        log['normalized_reward'] = model.running_reward
        log['episode_reward'] = model.episode_reward

        self.data = log
        

    def summary(self):
        print "Max reward:"

    def peak(self):
        """
        Maximum running reward during training
        """
        return max(self.data['running_reward'])

    def average(self):
        pass
    
    def longest_episode(self):
        pass

    def save(self):
        """
        Saves the self.data dictionary to a .log file
        """
        log_path = '{}/{}_{}_{}.log'.format(nao_rl.settings.DATA, self.data['env'], self.data['algorithm'], self.date)
        with open(log_path, 'w') as logfile:
            logfile.write(json.dumps(self.data))
        print 'Log file saved at {}'.format(log_path)


class Parser(object):

    """
    A class for parsing logs
    """

    def __init__(self, logs=None):
        if logs is not None:
            pass

    def load_directory(self, env=None, ):
        """
        Loads a directory containing .log files created by running the training algorithms
        """

        
    
    def find_best(self):
        """
        Finds the training instances with highest overall scores
        """

    def plot(self, best=0):

        pass
