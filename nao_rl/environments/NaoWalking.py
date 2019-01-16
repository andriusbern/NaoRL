
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
from nao_rl.utils import VrepNAO, RealNAO
from nao_rl import settings

class NaoWalking(VrepEnv):
    """ 
    The goal of the agent in this environment is to learn how to walk
    Reward function is proportional to the 
    """
    def __init__(self, address=None, port=None, naoqi_port=None, real=False):

        if port is None:
            port = settings.SIM_PORT
        if address is None:
            address = settings.LOCAL_IP
        
        VrepEnv.__init__(self, address, port)

        # Vrep
        self.real = real
        self.path = settings.SCENES + '/nao_test2.ttt'
        # self.connect() # Connect to vrep, load the scene and initialize the agent

        # Agent
        if self.real:
            self.agent = RealNAO(settings.REAL_NAO_IP, settings.NAO_PORT)
        else:
            self.agent = VrepNAO(True)

        self.active_joints = ["LLeg", "RLeg"]        # Joints, whose position should be streamed continuously
        self.body_parts = ['RFoot', 'LFoot', 'Torso'] # Body parts whose position should be streamed continuously
              
        # Action and state spaces
        self.velocities = np.zeros(12)
        self.velocity_bounds = [-1, 1]
        self.action_space = spaces.Box(low=np.dot(-1, np.ones(12)),
                                       high=np.dot(1, np.ones(12)), dtype=np.float32)  

        low = np.array([-np.inf] * 16)
        high = np.array([np.inf] * 16)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = np.zeros(16)

        # Simulations parameters
        self.fall_threshold = np.pi/5
        self.previous_feet_position = [0, 0]
        self.previous_body_position = 0
        self.done = False


    def initialize(self):

        self.connect() # Connect python client to VREP
        if self.real:
            self.agent.connect(settings.REAL_NAO_IP, settings.REAL_NAO_PORT, env=self)
        else:
            self.agent.connect(env=self)


    def _make_observation(self):
        """
        Make an observation: measure the position and orientation of NAO,
        and get current joint angles from motionProxy.
        """
        
        joint_angles = self.agent.get_angles()
        orientation = self.agent.get_orientation('Torso')[0:2]
        collisions = self.agent.check_collisions()
        self.state = np.hstack([joint_angles, orientation, int(collisions[0]), int(collisions[1])])           # Angles of each joint


    def _make_action(self, action):
        """
        Perform and action: increase the velocity of either 'HeadPitch' or 'HeadYaw'
        """
        self.agent.move_joints(action/40)
        # # Update velocities
        # self.velocities = np.add(self.velocities, action) # Add action to current velocities
        # self.velocities = np.clip(self.velocities, self.velocity_bounds[0], self.velocity_bounds[1]) # Clip to bounds
        # self.agent.move_joints(self.velocities)

    def step(self, action):
        """
        Step the vrep simulation by one frame
        """

        previous_state = self.state
        self._make_action(action) 
        self.step_simulation()
        self._make_observation() # Update state
        
        ###################
        ### Reward function

        body_position = self.agent.get_position('Torso')            # x,y,z coordinates of the agent
        r_foot_collision, l_foot_collision = self.state[-2:] # Feet collision indicators [0/1]
        roll, pitch = self.state[12:14]                      # Roll and pitch of the agent's convex hull

        # Staying upright
        posture = 0
        if abs(roll) > abs(previous_state[12]):
            posture -= .1
        else:
            posture += .125

        if abs(pitch) > abs(previous_state[13]):
            posture -= .1
        else:
            posture += .125
        
        hull = 0
        if abs(roll) < .125 and abs(pitch) < .125:
            posture += .1
            # Lifting feet while upright
            # collisions = np.count_nonzero(self.state[14::])
            # posture = (2 - collisions) * .

            # Hull location
            progress = body_position[0] - self.previous_body_position
            if progress > 0: 
                hull = 0.1 + progress * 40
                if hull > .5: hull = .5
            else:
                hull = -0.1 + progress * 40
                if hull < -.5: hull = -.5
        self.previous_body_position = body_position[0]

        """
        STATE SPACE:
        include:
            1. Angular velocity of the torso (also normal velocity?) both can be obtained through gyro and accelerometer
            2. Change to orientation of the torso instead of convex hull
            3. 
        """

        # Feet distance
        # Use multiplicative reward?
        # Change in feet position along the X axis
        # pos_lfoot = self.agent.get_position('LFoot')[0]
        # pos_rfoot = self.agent.get_position('RFoot')[0]
        # distance_lfoot = (pos_lfoot - self.previous_feet_position[0])
        # distance_rfoot = (pos_rfoot - self.previous_feet_position[1])
        # if self.previous_feet_position[0] != 0:
        #     feet_distance = (distance_lfoot + distance_rfoot) * 100
        # else:
        #     feet_distance = 0

        # self.previous_feet_position = [pos_lfoot, pos_rfoot]

        base = 0.05
        reward = base + posture + hull
        # print('hull: {}'.format(hull))
        # print('posture: {}'.cformat(posture))

        # End condition
        if (abs(roll) > self.fall_threshold or abs(pitch) > self.fall_threshold):
            reward -= 2
            self.done = True 

        # print('Posture: {} \n Hull: {}'.format(posture, hull))
        # print('Total reward: {}'.format(reward))

        return self.state, reward, self.done, {}

    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 
        
        self.stop_simulation()
        self.agent.reset_position()
        self.start_simulation()
        self.done = False
        self.state = np.zeros(16)

        # Reset initial variables
        self.previous_body_position = 0

        self.step_simulation()
        self._make_observation()
        return np.array(self.state)

    def run(self):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        
        t = 0
        while t < 10:
            self.reset()
            self.start_simulation()
            while not self.done:
                raw_input("Press Enter to continue...")
                action = self.action_space.sample()
                print(action)
                state, reward, self.done, _ = self.step(action)
                print('Current state:\n angles: {}'.format(state))
                print('Reward: {}'.format(reward))

            self.stop_simulation()
            t += 1


if __name__ == "__main__":
    
    """
    If called as a script this will initialize the scene in an open vrep instance 
    """

    # Environment and objects
    import nao_rl
    # scene = settings.SCENES + '/nao_test2.ttt'
    env = nao_rl.make('NaoWalking', headless=False)
    env.run()
