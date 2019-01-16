"""
@author: Andrius Bernatavicius, 2018
"""

class Log(object):
    """
    Used for storing results of training
    """
    def __init__(self, model):
        self.date = None
        self.algorithm
        self.episode_reward
        self.running_reward
        self.steps
        self.parameters = {}
    
    def load_from_file(self):
        pass

    def summary(self):
        pass

    def peak(self):
        pass

    def average(self):
        pass
    
    def longest_episode(self):
        pass

    def save(self):
        pass


class Parser(object):

    """
    A class for parsing logs
    """

    def __init__(self, logs=None):
        if logs is not None:


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
