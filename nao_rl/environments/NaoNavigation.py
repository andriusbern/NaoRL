# Local imports
from nao_rl.utils import VirtualNAO
from nao_rl.utils.vrep import vrep
from nao_rl.environments import VrepEnv

from gym import spaces

# from image_processing import ImageProcessor

"""
##################
##  Deprecated  ##
##################
"""

class NaoNavigation(VrepEnv):
    """
    A Vrep RL environment where the agent (NAO) navigates a maze-like setting using proximity
    sensor data executing predefined motions provided by NaoQi Python SDK. 
    The environment has the following MDP:
        - continuous state space - [4 distance sensors (spaced 90 deg apart)]
        - discrete action space  - [1 - move 2 steps forward; 
                                    2 - turn 10 deg right;
                                    3 - turn 10 deg left ]
        - Reward function: [-1 for hitting the walls]
    """
    def __init__(self, address, port, path, agent):
        VrepEnv.__init__(self, address, port, path)      
        self.agent = agent
        self.state = []
        self.action_space = spaces.Discrete(3)
        self.max_steps = 100

    def _make_observation(self):
        pass
        # Set to whatever needs to be retrieved from v-rep
        #image = self.agent.get_image(True)
        #self.agent.update_image()
        ## Image processing
        #self.state = image


    def _make_action(self, action):
        # Move forward
        status = ['Walking forward', 'Turning right', 'Turning left']
        print "-------------------------------"
        print "Action: {}".format(status[action])
        self.agent.discrete_move(action)
        

    def step(self, action):
        action = self.action_space.sample()
        self._make_action(action)
        self._make_observation()
        #self.agent.update_image('thresh')
        # Rewards
        reward = 0

        # End conditions 
        done = False
        
        return self.state, reward, done

    def reset(self, mode='human', close=False):
        if self.running:
            self.stop_simulation()
        self.start_simulation()

        action = self.action_space.sample()
        self._make_action(action)
        self._make_observation()

    def run(self):
        self.reset()
        done = False
        while not done:
            action = self.action_space.sample()   
            _, _, done = self.step(action)
        self.stop_simulation()

    def render(self, seed=None):
        pass

    def seed(self, seed=None):
        pass


if __name__ == "__main__":

    nao_ip = '127.0.0.1'
    sim_ip = '127.0.0.1'
    nao_port = 5995
    sim_port = 19998
    nao = VirtualNAO(nao_ip, nao_port)
    env = NaoNavigation(sim_ip, sim_port, '', nao)
    env.connect()
    env.load_scene('/home/andrius/Learning/thesis/code/vrep_tests/Project-NAO-Control/Vrep-Scene/NAO_maze.ttt')
    nao.connect_env(env)
    env.start_simulation()
    nao.initialize()