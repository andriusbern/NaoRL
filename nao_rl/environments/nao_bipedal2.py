
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
    Reward function is proportional to the
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
        self.active_joints = ["LLeg", "RLeg"]
              
        # Action and state spaces
        self.action_space = spaces.Box(low=np.dot(-.005, np.ones(12)),
                                       high=np.dot(.005, np.ones(12)),dtype=np.float32)  

        low = np.array([-np.inf] * 14)
        high = np.array([np.inf] * 14)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = np.zeros(14)

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
        
        self.state[0:12] = self.agent.get_angles()
        orientation = self.agent.get_orientation()
        self.state[12] = orientation[0]
        self.state[13] = orientation[1]

    def _make_action(self, action):
        """
        Perform and action: increase the velocity of either 'HeadPitch' or 'HeadYaw'
        """

        # Update velocities
        self.agent.move_joints(action)

    def step(self, action):
        """
        Step the vrep simulation by one frame
        """
        self.steps += 1
        self._make_action(action) 
        self.step_simulation()
        self._make_observation()

        
        position = self.agent.get_position()
        orientation = self.agent.get_orientation()

        pos = (position[0] - self.agent.initial_nao_position[0])
        orient = (1 - (abs(orientation[0]) + abs(orientation[1])/2)/2)
        
        reward = pos
        
        if orientation[0] < -np.pi/3 or orientation[0] > np.pi/3 or orientation[1] < -np.pi/3 or orientation[1] > np.pi/3:
            # reward -= 100
            reward += pos * 10
            self.done = True 

        return self.state, reward, self.done, {}

    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 

        # Initial position
        
        self.stop_simulation()
        time.sleep(.2)
        self.start_simulation()
        time.sleep(.2)
    
        if self.steps > 1:
            self.agent.active_joint_position = np.zeros(len(self.agent.active_joint_position))

        self.done = False
        self.state = np.zeros(14)

        # Make first observation
        self.step_simulation()
        self._make_observation()
        return np.array(self.state)

    def run(self):
        """
        Run the test simulation without any learning algorithm
        """
        self.start_simulation()
        done = False
        while not done:
            _, _, done, _ = self.step(self.action_space.sample())

        self.stop_simulation()


if __name__ == "__main__":
    
    """
    If called as a script this will initialize the scene in an open vrep instance 
    """
    # Environment and objects
    scene = settings.SCENES + '/nao_walk.ttt'
    env = NaoWalking2(settings.LOCAL_IP, settings.SIM_PORT, settings.NAO_PORT, scene)
    env.agent.connect_env(env)
    env.run()
    
