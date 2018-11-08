
"""
@author: Andrius Bernatavicius, 2018
"""
from __future__ import print_function
import time, math
import numpy as np
from gym import spaces
import cv2 as cv

# Local imports

from nao_rl.environments import VrepEnv
from nao_rl.utils import VirtualNAO, RealNAO
from nao_rl.utils import Ball
from nao_rl.utils import image_processing

class NaoTracking(VrepEnv):
    """ 
    Environment where the goal is to track the ball by moving two head motors.
    The ball moves randomly within a specified area and the reward is proportional to the distance from
    the center of the ball to the center of NAO's vision sensor.
    """
    def __init__(self, address, port, naoqi_port, path, window=None):
        VrepEnv.__init__(self, address, port, path)

        # Vrep
        self.window = window
        self.scene = path
        self.initialize() # Connect to vrep, load the scene and initialize the agent

        # Objects
        self.ball  = Ball('Sphere1')
        self.agent = VirtualNAO(address, naoqi_port)

        # Connect to environment
        self.agent.initialize()
        

        # Action and state spaces
        self.velocities = [0, 0]
        self.velocity_bounds = [-.02, .02]
        self.action_space = spaces.Box(np.array([-0.002, -0.002]), np.array([+0.002, +0.002]))

        low = np.array([
            0,                       0,                        # Lower bounds for the coordinates of the center of the ball 
            self.velocity_bounds[0], self.velocity_bounds[0]]) # Lower bounds for the velocities of the head motors
        high = np.array([
            1,                       1,                        # Upper bounds for the coordinates of the center of the ball 
            self.velocity_bounds[1], self.velocity_bounds[1]]) # Upper bounds for the velocities of the head motors

        self.observation_space = spaces.Box(low=low, high=high)
        self.state = []

        # Simulations parameters
        self.done = False
        self.steps = 0


    def initialize(self):
        """
        Connects to vrep, loads the scene and initializes the posture of NAO
        """
        self.close() # Reconnect
        self.connect()
        self.load_scene(self.scene)

        # Window for displaying the camera image
        if self.window is not None:
            cv.namedWindow(self.window, cv.WINDOW_NORMAL)
            cv.resizeWindow('s', 600,600)


    def _make_observation(self):
        """
        Make an observation: obtain an image from the virtual sensor
        State space - [normalized coordinates of the center of the ball [0-1], 
                       velocities of the head motors                         ]
        """
        image, resolution = self.agent.get_image(attempts=5)
        _, center, resolution = image_processing.ball_tracking(image, resolution, 'NaoDisplay')

        if center != None:
            coords = [float(center[x])/float(resolution[x]) for x in range(2)]
            self.state = [coords[0], coords[1], self.velocities[0], self.velocities[1]]
        else:
            self.done = True
            self.state = [0, 0, self.velocities[0], self.velocities[1]]
        

    def _make_action(self, action):
        """
        Perform and action: increase the velocity of either 'HeadPitch' or 'HeadYaw'
        """
        
        joints = self.agent.limbs["Head"] # Returns a list of head joint names

        # Update velocities
        for i in range(len(self.velocities)):
            self.velocities[i] += action[i] / 1000
            if self.velocities[i] < self.velocity_bounds[0]: self.velocities[i] = self.velocity_bounds[0]
            if self.velocities[i] > self.velocity_bounds[1]: self.velocities[i] = self.velocity_bounds[1]

        self.agent.move_joints(joints, self.velocities)

    def step(self, action):
        """
        Step the vrep simulation by one frame
        """
        self.steps += 1
        self._make_action(action)
        self.step_simulation()
        self.agent.naoqi_vrep_sync()
        self._make_observation()
        # self.ball.random_motion()
        (center_x, center_y, _, _) = self.state

        # Euclidean distance from the center of the ball
        reward = 1. - np.sqrt((0.5 - center_x)**2 + (0.5 - center_y)**2)
        if center_x > .8 or center_x < .2 or center_y < 0.2 or center_y > .8:
            reward = 0
        #if self.done:
            #reward = -1/self.steps * 200

        return np.array(self.state), reward, self.done, {}

    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 
        # Reset state
        self.done = False
        self.steps = 0
        self.velocities = [0, 0]
        self.state = []

        # Restart simulation
        if self.running:
            self.stop_simulation()
            time.sleep(.1)
        self.start_simulation()
        time.sleep(.1)

        # Reinitialize
        # self.agent.initialize()
        self.agent.set_joints(self.agent.limbs["Head"], [0, 0], 1)
        time.sleep(.2)
        self.agent.naoqi_vrep_sync()
        self.ball.restart()
        
        # Make first observation
        self.step_simulation()
        self.agent.naoqi_vrep_sync()
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


if __name__ == "__main__":

    """
    If called as a script this will initialize the scene in an open vrep instance 
    """
    ip = '127.0.0.1'
    port = 5995
    simulation_port = 19998
    scene = '/home/andrius/thesis/nao_rl/nao_rl/scenes/nao_ball.ttt'

    # Environment and objects
    env = NaoTracking(ip, simulation_port, port, scene)
    env.agent.connect_env(env)
    env.ball.connect_env(env)
    env.run()
    
