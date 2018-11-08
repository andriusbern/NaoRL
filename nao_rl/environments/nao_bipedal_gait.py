
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
from nao_rl.utils import VirtualNAO
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
        self.agent = VirtualNAO(address, naoqi_port)
        self.agent.initialize()
        self.joints = self.agent.limbs["LLeg"] + self.agent.limbs["RLeg"]
        self.initial_angles = self.agent.get_angles(self.joints, True)
        self.initial_position = None
        self.position = None
        self.orientation = None
              
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
        # self.close()
        self.connect()
        # self.load_scene(self.scene)

    def _make_observation(self):
        """
        Make an observation: measure the position and orientation of NAO,
        and get current joint angles from motionProxy.
        """
        self.orientation = self.get_object_orientation(self.agent.handle)
        self.position = self.get_object_position(self.agent.handle)

        self.state[0:11] = self.agent.get_angles(self.joints, True)
        self.state[12::] = self.velocities

    def _make_action(self, action):
        """
        Perform and action: increase the velocity of either 'HeadPitch' or 'HeadYaw'
        """

        # Update velocities
        self.velocities = np.add(self.velocities, action/1000) # Add action to current velocities
        self.velocities = np.clip(self.velocities, self.velocity_bounds[0], self.velocity_bounds[1]) # Clip to bounds

        self.agent.move_joints(self.joints, list(self.velocities))
        self.agent.naoqi_vrep_sync()

    def step(self, action):
        """
        Step the vrep simulation by one frame
        """
        self.steps += 1
        self._make_action(action) 
        self.step_simulation()
        self.agent.naoqi_vrep_sync()
        self._make_observation()

        # Reward function
        # reward += np.sqrt((self.position[0] - self.initial_position[0])**2 + 
        #                   (self.position[1] - self.initial_position[1])**2) * 10000000
        # reward += (1 - (abs(self.orientation[0]) + abs(self.orientation[1]))/2)

        pos = (self.position[0] - self.initial_position[0]) * 100000000
        orient = (1 - (abs(self.orientation[0]) + abs(self.orientation[1])/2)/2)
        reward = pos * orient
        # print("Pos: {}, orient: {}".format(pos, orient))
        
        if self.orientation[0] < -np.pi/3 or self.orientation[0] > np.pi/3 or self.orientation[1] < -np.pi/3 or self.orientation[1] > np.pi/3:
            reward -= 100
            self.done = True 
            self.agent.motionProxy.killAll()

        return self.state, reward, self.done, {}

    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 
        # Reset state
        print('-------------------------------------------------')
        self.done = False
        self.steps = 0
        self.velocities = np.zeros(12)
        self.state = np.zeros(24)

        # Restart simulation
        if self.running:
            self.stop_simulation()
            time.sleep(.2)
        self.start_simulation()
        time.sleep(.2)

        # Reinitialize
        self.agent.set_joints(self.joints, [0 for _ in range(12)], 1)
        time.sleep(.25)
        # Make first observation
        self.step_simulation()
        self.agent.naoqi_vrep_sync()
        self._make_observation()
        self.initial_position = self.get_object_position(self.agent.handle)
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
    
