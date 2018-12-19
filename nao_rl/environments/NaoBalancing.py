
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


class NaoBalancing(VrepEnv):
    """
    The goal of the agent in this environment is to learn how to walk
    """
    def __init__(self, address=None, port=None, naoqi_port=None):

        if port is None:
            port = settings.SIM_PORT
        if address is None:
            address = settings.LOCAL_IP
        
        VrepEnv.__init__(self, address, port)

        # Vrep
        self.initialize()  # Connect to vrep, load the scene and initialize the agent

        # Agent
        self.agent = VrepNAO(True)
        self.active_joints = ["LLeg", "RLeg"]  # Joints, whose position should be streamed continuously
        self.body_parts = ['RFoot', 'LFoot', 'Head'] # Body parts which position should be streamed continuously
              
        # Action and state space limits
        self.action_space = spaces.Box(low=np.dot(-1, np.ones(12)),
                                       high=np.dot(1, np.ones(12)), dtype=np.float32)  

        self.observation_space = spaces.Box(low=np.array([-np.inf] * 16),
                                            high=np.array([np.inf] * 16), dtype=np.float32)

        ##### State space
        # Consists of 16 variables:
        #    - Angular positions of leg motors (6 for each leg) [12]
        #    - Roll and pitch of the convex hull of the body of the whole robot [2]
        #    - Collision between the floor and each foot [2]
        self.state = np.zeros(16)
        self.previous_feet_position = [0, 0]

        # The environment resets if roll or pitch is above this threshold        
        self.fall_threshold = np.pi/5

        # Simulation parameters
        self.done = False

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

        joint_angles = self.agent.get_angles()
        orientation = self.agent.get_orientation()[0:2]
        collisions = self.agent.check_collisions()
        self.state = np.hstack([joint_angles, orientation, int(collisions[0]), int(collisions[1])])           # Angles of each joint

    def _make_action(self, action):
        """
        Perform an action - move each joint by a specific amount
        """

        # Update velocities
        self.agent.move_joints(action/40)


    def step(self, action):
        """
        Step the vrep simulation by one frame, make actions and observations and calculate the resulting 
        rewards
        """

        previous_state = self.state
        self._make_action(action) 
        self.step_simulation()
        self._make_observation() # Update state
        
        ###################
        ### Reward function

        body_position = self.agent.get_position()            # x,y,z coordinates of the agent
        r_foot_collision, l_foot_collision = self.state[-2:] # Feet collision indicators [0/1]
        roll, pitch = self.state[12:14]                      # Roll and pitch of the agent's convex hull

        # Change in feet position along the X axis
        pos_lfoot = self.agent.get_position('LFoot')[0]
        pos_rfoot = self.agent.get_position('RFoot')[0]
        distance_lfoot = (pos_lfoot - self.previous_feet_position[0])
        distance_rfoot = (pos_rfoot - self.previous_feet_position[1])

        #### Function 1 
        # pos = (position[0] - self.agent.initial_nao_position[0])
        # pos1 = (self.agent.get_position('LFoot')[0] - self.agent.initial_position['LFoot'][0])
        # pos2 = (self.agent.get_position('RFoot')[0] - self.agent.initial_position['RFoot'][0])
        
        # reward = (pos1 + pos2)/2 - orient/5 
        # reward = ((1-orient)**2)/10 #(pos1 + pos2)/2 + 
        # print("Feet: {}, orient: {}".format((pos1+pos2)/2, abs(orient/10)))
        # if reward > 0.09:
        #     reward += .1
        # if reward > .095:
        #     reward += .1


        # #### Function 2
        # orientation_score = (abs(roll) + abs(pitch)) / 2
        # reward = 0
        # # Include negative rewards
        # orientation_difference = abs(orientation_score - self.previous_orientation)
        # if orientation_score < self.previous_orientation:
        #     reward += 100 * orientation_difference
        # else:
        #     reward -= 50 * orientation_difference
        # if self.previous_orientation is None:
        #     reward = 0
        # # else:
        # #     reward += ((1 - orientation_score) ** 4) / 20

        # self.previous_orientation = [roll, pitch]

        
        #### Function 3
        # Staying upright
        reward = 0.05 # Default reward for each step
        if abs(roll) > abs(self.previous_state[12]):
            reward -= .1
        else:
            reward += .125

        if abs(pitch) > abs(self.previous_state[13]):
            reward -= .1
        else:
            reward += .125
        
        # if roll < .1 and pitch < .1:
        #     reward += .2

        if (abs(roll) > self.fall_threshold or abs(pitch) > self.fall_threshold):
            reward -= 2
            self.done = True 
        self.previous_feet_position = [pos_lfoot, pos_rfoot]
        

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
        self.step_simulation()
        self._make_observation()
        return np.array(self.state)

    def run(self):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        self.reset()
        t = 0
        while t < 10:
            self.done = False
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
    env = nao_rl.make('NaoBalancing', headless=False)
    env.run()
