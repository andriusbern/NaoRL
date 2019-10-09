
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


class NaoReaching(VrepEnvironment):
    """ 
    Environment where the goal is to track the ball by moving two head motors.
    The ball moves randomly within a specified area and the reward is proportional to the distance from
    the center of the ball to the center of NAO's vision sensor.
    """
    def __init__(self, address=None, port=None, naoqi_port=None, use_real_agent=True):

        VrepEnvironment.__init__(self, address, port)
        # super(NaoTracking, self).__init__(address, port)

        self.path = nao_rl.settings.SCENES + '/nao_reach.ttt'
        self.real = use_real_agent

        # Movement and actions
        self.active_joints       = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
        self.body_parts_to_track = ['Torso']
        self.movement_mode       = 'velocity'
        self.collisions          = ['NAO_BALL', 'NAO_TABLE']
        self.joint_speed         = 5
        self.fps                 = 30.
        
        if self.real:
            self.agent = nao_rl.agents.RealNAO(self.active_joints)
        else:
            self.agent = nao_rl.agents.VrepNAO(self.active_joints)
        
        self.agent.max_joint_velocity = .1
        
        # State and action spaces
        self.n_states  = 10
        self.n_actions = 4
        self.state     = np.zeros(self.n_states)
        self.done      = False
        
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
        _, center  = nao_rl.utils.ImageProcessor.ball_tracking(image, self.show_display)
        positions  = self.agent.get_angles()
        velocities = self.agent.joint_angular_v / self.agent.max_joint_velocity # Normalize
    
        if center != None:
            resolution = np.shape(image)
            coords = [float(center[0])/float(resolution[1]), float(center[1])/float(resolution[0])]
        else:
            coords = [0, 0]
        
        self.state = np.array(coords + list(positions) + list(velocities))
            

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
            self.step_simulation()
        self._make_observation()
        
        # Base reward for each step
        reward = -.1 

        # Check collisions
        if not self.real:
            if self.agent.get_collision('NAO_BALL'): # Reward for touching the ball
                self.done = True
                reward += 100
            if self.agent.get_collision('NAO_TABLE'): # Penalty for touching the table
                self.done = True
                reward -= 10
        
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
        data = [[] for i in range(4)]
        try:
            t = 0
            while t < 30:
                self.done = False
                self.reset()
                ep_reward = 0
                steps = 0
                while not self.done and steps < 200: 
                    action = self.action_space.sample()
                    # action = np.zeros(4)
                    state, reward, self.done, _ = self.step(action)
                    time.sleep(1/fps)
                    steps += 1
                    ep_reward += reward
                    # print(reward)
                    for i in range(2,6):
                        data[i-2].append(state[i])
                print('Steps: {}'.format(steps))
                t += 1
                print('Episode reward: {}'.format(ep_reward))
        except KeyboardInterrupt:
            pass
            import matplotlib.pyplot as plt
            for i in range(4):
                plt.plot(data[i])
            
            plt.legend(self.active_joints)
            plt.show()

if __name__ == "__main__":

    """
    If called as a script this will initialize the scene in an open vrep instance 
    """
    import nao_rl
    env = nao_rl.make('NaoReaching', headless=False, show_display=True)
    env.run()
    nao_rl.destroy_instances()