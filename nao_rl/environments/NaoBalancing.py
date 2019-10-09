
"""
@author: Andrius Bernatavicius, 2018
"""

from __future__ import print_function
import time
import numpy as np
from gym import spaces

# Local imports
import nao_rl
from nao_rl.environments import VrepEnvironment

class NaoBalancing(VrepEnvironment):
    """
    The goal of the agent in this environment is to learn how to walk
    """
    def __init__(self, address=None, port=None, naoqi_port=None, real=True):

        super(NaoBalancing, self).__init__(address, port)

        # Vrep
        self.path = nao_rl.settings.SCENES + '/nao_standzero.ttt'
        self.real = real
        self.n_states = 14

        ### Agent settings
        self.active_joints       = ["LLeg", "RLeg"]  # Joints that are going to be used
        self.body_parts_to_track = ['Torso']         # Body parts the position and orientation of which are used as states
        self.movement_mode       = 'position'        # 'position' / 'velocity' / 'torque'
        self.joint_speed         = 1.25
        self.fps                 = 30.
        self.collisions          = None
        
        # Agent
        if self.real:
            self.agent = nao_rl.agents.RealNAO(self.active_joints)
        else:
            self.agent = nao_rl.agents.VrepNAO(self.active_joints)

        # Action and state space limits
        self.action_space = spaces.Box(low=np.dot(-1, np.ones(12)),
                                       high=np.dot(1, np.ones(12)), dtype=np.float32)  

        self.observation_space = spaces.Box(low=np.array([-np.inf] * self.n_states),
                                            high=np.array([np.inf] * self.n_states), dtype=np.float32)

        ##### State space
        # Consists of 16 variables:
        #    - Angular positions of leg motors (6 for each leg) [12]
        #    - Roll and pitch of the convex hull of the body of the whole robot [2]
        #    - Collision between the floor and each foot [2]
        self.state = np.zeros(self.n_states)
        self.previous_state = np.zeros(self.n_states)
        self.previous_orientation = 0

        # The environment resets if roll or pitch is above this threshold        
        self.fall_threshold = np.pi/5

        # Simulation parameters
        self.done = False

    def initialize(self):
        """
        Connect to V-REP or NAO
        This method is automatically called when the environment is created
        """
        if self.real:
            self.agent.connect(self)
        else:
            self.connect() # Connect python client to VREP
            self.agent.connect(self)

    def _make_observation(self):
        """
        Make an observation: measure the position and orientation of NAO,
        and get current joint angles from motionProxy.

        """
        joint_angles = self.agent.get_angles()
        # orientation = self.agent.get_orientation('Torso')[0:2]
        orientation = [0, 0]
        #collisions = self.agent.check_collisions()
        self.state = np.hstack([joint_angles, orientation])           # Angles of each joint

    def _make_action(self, action):
        """
        Perform an action - move each joint by a specific amount
        """

        # Update velocities
        self.agent.act(action, self.movement_mode, self.joint_speed)


    def step(self, action):
        """
        Step the vrep simulation by one frame, make actions and observations and calculate the resulting 
        rewards
        """
        
        previous_state = self.state
        self._make_action(action)
        if not self.real:
            self.step_simulation()
        self._make_observation() # Update state
        
        ###################
        ### Reward function

        roll, pitch = self.state[12:14]                      # Roll and pitch of the agent's convex hull

        # Staying upright
        reward = 0.1 # Default reward for each step
        if abs(roll) > abs(self.previous_state[12]):
            reward -= .1
        else:
            reward += .125

        if abs(pitch) > abs(self.previous_state[13]):
            reward -= .1
        else:
            reward += .125
        
        self.previous_state = self.state
        if abs(roll) < .05 and abs(pitch) < .05:
            reward += .2

        if (abs(roll) > self.fall_threshold or abs(pitch) > self.fall_threshold):
            reward -= 2
            self.done = True 
        
        return self.state, reward, self.done, {}


    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 
        if not self.real:
            self.stop_simulation()    
            self.start_simulation()
            self.step_simulation()

        self.agent.reset_position()
        self.done = False
        self.state = np.zeros(self.n_states)
        
        self._make_observation()
        return np.array(self.state)

    def run(self, policy=None):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        
        t = 0
        while t < 30:
            self.done = False
            self.reset()
            fps = 30.
            while not self.done:
                # raw_input("Press Enter to continue...")
                action = self.action_space.sample()
                print(action)
                state, reward, self.done, _ = self.step(action)
                print('Current state:\n angles: {}'.format(state))
                print('Reward: {}'.format(reward))
                time.sleep(1/fps)

            t += 1


if __name__ == "__main__":
    """
    If called as a script this will initialize the scene in an open vrep instance
    """

    # Environment and objects
    import nao_rl
    env = nao_rl.make('NaoBalancing', headless=False)
    env.run()
    nao_rl.destroy_instances()