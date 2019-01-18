
"""
@author: Andrius Bernatavicius, 2018
"""
from __future__ import print_function
import time, math
import numpy as np
import cv2 as cv
from gym import spaces, Env

# Local imports
from nao_rl.environments import VrepEnvironment

import nao_rl


class NaoTracking(VrepEnvironment):
    """ 
    Environment where the goal is to track the ball by moving two head motors.
    The ball moves randomly within a specified area and the reward is proportional to the distance from
    the center of the ball to the center of NAO's vision sensor.
    """
    def __init__(self, address=None, port=None, naoqi_port=None, use_real_agent=False):
        VrepEnvironment.__init__(self, address, port)
        # super(NaoTracking, self).__init__(address, port)

        self.path = nao_rl.settings.SCENES + '/nao_ball.ttt'
        self.real = use_real_agent

        # Movement and actions
        self.active_joints       = ['Head']
        self.body_parts_to_track = ['Head']
        self.movement_mode       = 'velocity'
        self.joint_speed         = 2
        self.fps                 = 30.
        
        if self.real:
            self.agent = nao_rl.agents.RealNAO(self.active_joints)
        else:
            self.agent = nao_rl.agents.VrepNAO(self.active_joints)
        
        # State and action spaces
        self.n_states  = 4
        self.n_actions = 2
        self.state     = np.zeros(self.n_states)
        self.done      = True
        
       # Action and state space limits
        self.action_space      = spaces.Box(low=np.array([-1] * self.n_actions),
                                            high=np.array([1] * self.n_actions), dtype=np.float32)  

        self.observation_space = spaces.Box(low=np.array([-np.inf] * self.n_states),
                                            high=np.array([np.inf] * self.n_states), dtype=np.float32)

        # Additional parameters
        self.ball = nao_rl.utils.Ball(name='Sphere1')  # Ball object (defined in ../utils/misc.py)
        self.show_display = False  # Display the processed image in a cv2 window (object tracking)


    def initialize(self):
        """
        Connect to V-REP or NAO
        """
        if self.real:
            self.agent.connect(self)
        else:
            self.connect() # Connect python client to VREP
            self.agent.connect(env=self, use_camera=True)
            self.ball.connect_env(self)
        
    def _make_observation(self):
        """
        Make an observation: obtain an image from the virtual sensor
        State space - [normalized coordinates of the center of the ball [0-1], 
                       velocities of the head motors]
        """
        image = self.agent.get_image()
        _, center = nao_rl.utils.ImageProcessor.ball_tracking(image, self.show_display)
        velocities = self.agent.joint_angular_v / self.agent.max_joint_velocity
    
        if center != None:
            resolution = np.shape(image)
            coords = [float(center[0])/float(resolution[1]), float(center[1])/float(resolution[0])]
            self.state = coords + list(velocities)
        else:
            self.done = True
            self.state = [0, 0, velocities[0], velocities[1]]

    def _make_action(self, action):
        """
        Perform and action: increase the velocity of either 'HeadPitch' or 'HeadYaw'
        """
        self.agent.act(action, self.movement_mode, self.joint_speed)

    def step(self, action):
        """
        Step the vrep simulation by one frame
        """
        self._make_action(action)
        if not self.real:
            self.ball.random_motion()
            self.step_simulation()
        self._make_observation()
        
        (center_x, center_y, _, _) = self.state

        # Euclidean distance from the center of the ball
        euclidean_distance = np.sqrt((0.5 - center_x)**2 + (0.5 - center_y)**2)
        reward = .3 - euclidean_distance
        if euclidean_distance < .1:
            reward += .1

        
        # Set the length of one step
        if self.real:
            time.sleep(1/self.fps)

        return np.array(self.state), reward, self.done, {}

    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 

        self.done = False
        if not self.real:
            self.stop_simulation()
            self.ball.restart()
            self.start_simulation()
            self.step_simulation()

        self.agent.reset_position()
        time.sleep(1)
        self._make_observation()

        return np.array(self.state)

    def run(self):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        fps = 30.
        try:
            t = 0
            while t < 30:
                self.done = False
                self.reset()
                
                steps = 0
                while not self.done: 
                    action = self.action_space.sample()
                    state, reward, self.done, _ = self.step(action)
                    time.sleep(1/fps)
                    steps += 1
                    print(reward)
                print('Steps: {}'.format(steps))
                t += 1
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":

    """
    If called as a script this will initialize the scene in an open vrep instance 
    """
    import nao_rl
    env = nao_rl.make('NaoTracking', headless=False, show_display=True)
    env.run()
    nao_rl.destroy_instances()