
class Log:
    def __init__(self):
        self.data = {}

    def add_history(self, **kwargs):
        

    def plot(self):
        import matplotlib.pyplot as plt
        for i, episode in enumerate(self.rewards):
            plt.plot(episode)
            # label corresponds to exp_names

        plt.show()

    def save_to_file(self, filename):
        """
        Save data to .log file
        (TRY PICKLE?)
        """
        with open(filename, 'wb') as file:
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
