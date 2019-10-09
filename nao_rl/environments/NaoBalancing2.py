
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

class NaoBalancing2(VrepEnvironment):
    """
    The agent's goal in this environment is to learn to keep an upright position
    """
    def __init__(self, address=None, port=None, naoqi_port=None, use_real_agent=False):

        VrepEnvironment.__init__(self, address, port)

        # Vrep
        self.real = use_real_agent
        self.path = nao_rl.settings.SCENES + '/nao_standzero.ttt'

        ### Agent settings
        self.active_joints       = ["LLeg", "RLeg"]  # Joints that are going to be used
        self.body_parts_to_track = ['Torso']         # Body parts the position and orientation of which are used as states
        self.movement_mode       = 'torque'        # 'position' / 'velocity' / 'torque'
        self.joint_speed         = .5
        self.fps                 = 30.
        self.collisions = None
        
        # Agent
        if self.real:
            self.agent = nao_rl.agents.RealNAO(self.active_joints)
        else:
            self.agent = nao_rl.agents.VrepNAO(self.active_joints)

        ### State and action spaces
        self.n_states  = 26 # Size of the state space
        self.n_actions = 12
        self.state     = np.zeros(self.n_states)
        self.done      = False

        # Action and state space limits
        self.action_space      = spaces.Box(low=np.array([-1] * self.n_actions),
                                            high=np.array([1] * self.n_actions), dtype=np.float32)  

        self.observation_space = spaces.Box(low=np.array([-np.inf] * self.n_states),
                                            high=np.array([np.inf] * self.n_states), dtype=np.float32)

        # Additional parameters
        self.fall_threshold = np.pi/5


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
        Get the current state of the agent
          Consists of 26 variables:
            - Current angles of active joints (6 for each leg) in radians [12] 
            - Angular velocities of each active joint in radians/s 
            - Roll and pitch of the convex hull of the torso of the robot [2] radians, where [0, 0] means upright position
        """
        joint_angles      = self.agent.get_angles()                 
        joint_velocities  = np.copy(self.agent.joint_angular_v) 
        joint_velocities /= self.agent.max_joint_velocity      
        orientation       = self.agent.get_orientation('Torso')[0:2] 

        self.state = np.hstack([joint_angles, joint_velocities, orientation]) 

    def _make_action(self, action):
        """
        Move joints
        """
        self.agent.act(action, self.movement_mode, self.joint_speed)

    def step(self, action):
        """
        1. Perform an action
        2. Get new observations
        3. Evaluate the reward for current state
        """
        
        previous_state = self.state
        self._make_action(action) 
        self.step_simulation()
        self._make_observation() # Update state
        
        roll, pitch = self.state[-2:]  # Roll and pitch of the agent's torso

        reward = 0.1 # Default reward for each step

        # Staying upright
        # If roll or pitch decreased when compared to previous roll/pitch
        if abs(roll) < abs(previous_state[-2]):  reward += .125
        else: reward -= .1

        if abs(pitch) < abs(previous_state[-1]): reward += .125
        else: reward -= .1
        
        # If close to being completely upright
        if abs(roll) < .05 and abs(pitch) < .05: reward += .2

        # Fall condition
        if (abs(roll) > self.fall_threshold or abs(pitch) > self.fall_threshold):
            reward -= 2
            self.done = True 
        
        return self.state, reward, self.done, {}


    def reset(self, close=False):
        """
        Reset the environment to default state and return the first observation
        """ 
        self.done = False

        if not self.real:
            self.stop_simulation()
            self.start_simulation()
            self.step_simulation()

        self.agent.reset_position()
        self._make_observation()
        return np.array(self.state)

    def run(self):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        data = [[] for x in range(40)]

        fps = 30.
        try:
            t = 0
            while t < 30:
                self.done = False
                self.reset()
                
                steps = 0
                while not self.done: 
                    # raw_input("Press Enter to continue...")
                    action = self.action_space.sample()
                    state, reward, self.done, _ = self.step(action)
                    time.sleep(1/fps)

                    for i in range(6):
                        data[i].append(self.state[i])
                    
                    # data[0].append(self.state[0])
                    # data[1].append(self.state[12])
                    # data[2].append(self.agent.joint_torque[0]/self.agent.max_torque)

                    # for i in range(12):
                    #     data[i].append(state[i])
                    steps += 1
                print('Steps: {}'.format(steps))
                t += 1
        except KeyboardInterrupt:
            import matplotlib.pyplot as plt
            for x in range(6):
                plt.plot(data[x])
            plt.legend(self.agent.active_joints[0:6])
            plt.show()
            

if __name__ == "__main__":
    """
    If called as a script this will initialize the scene in an open vrep instance
    """

    # Environment and objects
    import nao_rl
    env = nao_rl.make('NaoBalancing2', headless=False)
    env.run()
    nao_rl.destroy_instances()