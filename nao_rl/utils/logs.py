"""
Author: Andrius Bernatavicius, 2018
"""

### Functionality to be implemented


class Log:
    def __init__(self):
        self.data = {}

    def add_history(self, **kwargs):
        pass

    def plot(self):
        pass
        

    def save_to_file(self, filename):
            pass

    def load_file(self, filename):
        """
        Parse a .log file
        """ 
        pass

    def get_averages(self):
        """
        Get average rewards over all episodes
        """
        pass
