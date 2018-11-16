
"""
@author: Andrius Bernatavicius, 2018
"""

from __future__ import print_function
import time
import numpy as np
from gym import spaces
import cv2 as cv

# Local imports
from nao_rl.environments import VrepEnv
from nao_rl.utils import VrepNAO
from nao_rl import settings

class NaoWalking(VrepEnv):
    """ 
    The goal of the agent in this environment is to learn how to walk
    Reward function is proportional to the 
    """
    def __init__(self, address, port, naoqi_port, path):
        VrepEnv.__init__(self, address, port, path)

        # Vrep
        self.scene = path
        self.initialize() # Connect to vrep, load the scene and initialize the agent

        # Agent
        self.agent = VrepNAO(True)
        #self.agent.initialize()
        self.active_joints = ["LLeg", "RLeg"]
              
        # Action and state spaces
        self.velocities = np.zeros(12)
        self.velocity_bounds = [-.02, .02]
        self.action_space = spaces.Box(low=np.dot(-.005, np.ones(12)),
                                       high=np.dot(.005, np.ones(12)))  

        low = np.array([-np.inf] * 24)
        high = np.array([np.inf] * 24)

        self.observation_space = spaces.Box(low=low, high=high)
        self.state = np.zeros(24)

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
        self.state[12::] = self.velocities


    def _make_action(self, action):
        """
        Perform and action: increase the velocity of either 'HeadPitch' or 'HeadYaw'
        """

        # Update velocities
        self.velocities = np.add(self.velocities, action) # Add action to current velocities
        self.velocities = np.clip(self.velocities, self.velocity_bounds[0], self.velocity_bounds[1]) # Clip to bounds
        self.agent.move_joints(self.velocities)

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
        
        reward = pos * orient
        # print(pos, orient, reward)
        
        if orientation[0] < -np.pi/3 or orientation[0] > np.pi/3 or orientation[1] < -np.pi/3 or orientation[1] > np.pi/3:
            reward -= 100
            self.done = True 

        return self.state, reward, self.done, {}

    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 
        # Reset state
        # Initial position
        
        self.stop_simulation()
        time.sleep(.2)
        self.start_simulation()
        time.sleep(.2)
    
        if self.steps > 1:
            self.agent.active_joint_position = np.zeros(len(self.agent.active_joint_position))

        self.done = False
        self.velocities = np.zeros(12)
        self.state = np.zeros(24)

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
    env = NaoWalking(settings.LOCAL_IP, settings.SIM_PORT, settings.NAO_PORT, scene)
    env.agent.connect_env(env)
    env.run()
    
