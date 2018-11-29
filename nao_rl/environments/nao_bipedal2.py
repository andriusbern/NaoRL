
"""
@author: Andrius Bernatavicius, 2018
"""

from __future__ import print_function
import time
import numpy as np
from gym import spaces

# Local imports
from nao_rl.environments import VrepEnv
from nao_rl.utils import VrepNAO
from nao_rl import settings


class NaoWalking2(VrepEnv):
    """ 
    The goal of the agent in this environment is to learn how to walk
    """
    def __init__(self, address=None, port=None, naoqi_port=None, path=None):

        if port is None:
            port = settings.SIM_PORT
        if address is None:
            address = settings.LOCAL_IP
        
        VrepEnv.__init__(self, address, port, path)

        # Vrep
        self.scene = path
        self.initialize()  # Connect to vrep, load the scene and initialize the agent

        # Agent
        self.agent = VrepNAO(True)
        self.active_joints = ["LLeg", "RLeg"]  # Joints, whose position should be streamed continuously
        self.body_parts = ['l_foot', 'r_foot', 'head'] # Body parts which position should be streamed continuously
              
        # Action and state space limits
        self.action_space = spaces.Box(low=np.dot(-1, np.ones(12)),
                                       high=np.dot(1, np.ones(12)),dtype=np.float32)  

        low = np.array([-np.inf] * 14)
        high = np.array([np.inf] * 14)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = np.zeros(14)

        self.fall_threshold = np.pi/4

        # Simulations parameters
        self.done = False
        self.steps = 0

    def initialize(self):
        """
        Connects to vrep, loads the scene and initializes the posture of NAO
        """
        self.connect()

    def _make_observation(self):
        """
        Make an observation: measure the position and orientation of NAO,
        and get current joint angles from motionProxy.

        """

        #### ADD A FLAG FOR FEET PLACEMENT

        self.state[0:12] = self.agent.get_angles()            # Angles of each joint
        self.state[12:14] = self.agent.get_orientation()[0:2] # Tilt around x and y axes


    def _make_action(self, action):
        """
        Perform an action - move each joint by a specific amount
        """

        # Update velocities
        self.agent.move_joints(action/80)


    def step(self, action):
        """
        Step the vrep simulation by one frame, make actions and observations and calculate the resulting 
        rewards
        """
        self.steps += 1
        self._make_action(action) 
        self.step_simulation()
        self._make_observation()
        
        position = self.agent.get_position()
        orientation = self.agent.get_orientation()

        pos = (position[0] - self.agent.initial_nao_position[0])
        orient = (abs(orientation[0]) + abs(orientation[1]))/2

        pos1 = (self.agent.get_position('l_foot')[0] - self.agent.initial_position['l_foot'][0])
        pos2 = (self.agent.get_position('r_foot')[0] - self.agent.initial_position['r_foot'][0])
        
        # reward = (pos1 + pos2)/2 - orient/5 
        reward = ((1-orient)**2)/10 #(pos1 + pos2)/2 + 
        # print("Feet: {}, orient: {}".format((pos1+pos2)/2, abs(orient/10)))
        
        if (orientation[0] < -self.fall_threshold or orientation[0] > self.fall_threshold or
            orientation[1] < -self.fall_threshold or orientation[1] > self.fall_threshold):
            # reward -= 100
            reward -= 10
            # reward += pos * 10
            self.done = True 

        return self.state, reward, self.done, {}

    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 

        # Initial position
        self.stop_simulation()
        self.agent.reset_position()
        self.start_simulation()
        self.done = False
        self.state = np.zeros(14)
        self.step_simulation()
        self._make_observation()
        return np.array(self.state)

    def run(self):
        """
        Run the test simulation without any learning algorithm
        """
        
        t = 0
        while t < 10:
            done = False
            self.start_simulation()
            while not done:
                _, _, done, _ = self.step(self.action_space.sample())

            self.stop_simulation()
            t += 1


if __name__ == "__main__":
    
    """
    If called as a script this will initialize the scene in an open vrep instance 
    """
    # Environment and objects
    import nao_rl
    scene = settings.SCENES + '/nao_test2.ttt'
    env = nao_rl.make('nao-bipedal2', 19996, headless=False)
    env.run()
    
