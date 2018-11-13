import matplotlib.pyplot as plt

class Log:
    def __init__(self):
        self.rewards = []
        self.exp_names = None
        self.steps = []
        self.params = None

    def add_history(self, h, name):
        self.rewards.append(h.history['episode_reward'])
        self.exp_names.append(name)
        self.steps.append(h.history['nb_steps'])

    def plot(self):
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
