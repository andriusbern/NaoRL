import time, array
import numpy as np


class NAO(object):
    """
    Base class for NAO that contains all the functions that can be used for both real and virtual versions of NAO
        Constructor:
            [joints] - list of joints that are going to be used (default - All joints)
    """
    def __init__(self, joints='Body'):

        self.env = None
        ############
        ####  Joints
        # Ordering corresponds to the one provided by Aldebaran Robotics

        self.joint_names = [
            # "Head"
            'HeadYaw', 'HeadPitch',
            # "LArm"
            'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
            # "LLeg"
            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            # "RLeg"
            'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
            # "RArm"
            'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw'
            # "RHand"
            ]

        # A dictionary for quickly getting the names of joints in each limb
        self.limbs = {
            "Head": self.joint_names[0:2],
            "LArm": self.joint_names[2:7],
            "LLeg": self.joint_names[7:13],
            "RLeg": self.joint_names[13:19],
            "RArm": self.joint_names[19::],
            "Body": self.joint_names
            }

        # Active joints
        self.active_joints   = self.select_active_joints(joints) # List of joint names that are active (used in the learning environment)
        self.joint_position  = np.zeros(len(self.active_joints)) # Angular position of active joints in radians
        self.joint_angular_v = np.zeros(len(self.active_joints)) # Angular velocity of active joints
        self.initial_angles  = np.zeros(len(self.active_joints)) 
        self.joint_torque = np.zeros(len(self.active_joints)) 

        self.max_torque = .005
        self.max_joint_velocity = 0.03


    def select_active_joints(self, name_list):
        """
        Given a list containing limb or joint names create a list of the joints that are going to be active
        """
        active_joints = []
        for name in name_list:
            try:
                active_joints += self.limbs[name] 
            except KeyError:
                if name in self.joint_names:
                    active_joints.append(name)
                else:
                    print "Invalid name of joint '{}'".format(name)

        return active_joints


    def act(self, action, movement_mode='torque', speed=1):
        """
        Given an action vector, perform an action by either changing:
            - Torque
            - Current velocity
            - Current position
        """

        if movement_mode == 'torque':
            scaled_action = (action * speed / 400)
            self.joint_torque = np.clip(self.joint_torque + scaled_action,
                                          -self.max_torque,
                                           self.max_torque)
            
            self.joint_angular_v = np.clip(self.joint_angular_v + self.joint_torque,
                                          -self.max_joint_velocity,
                                           self.max_joint_velocity)

            self.joint_position += self.joint_angular_v

        elif movement_mode == 'velocity':
            scaled_action = (action * speed / 200)
            self.joint_angular_v = np.clip(self.joint_angular_v + scaled_action,
                                          -self.max_joint_velocity,
                                           self.max_joint_velocity)

            self.joint_position += self.joint_angular_v

        elif movement_mode == 'position':
            scaled_action = action / 50 * speed 
            self.joint_position += scaled_action
            
        else:
            print "Invalid mode for actions."
        
        
        self.move_joints(self.joint_position)

    def visualize(self, motors=None):
        pass

    ##########################################
    # METHODS TO BE OVERRIDDEN WITH SUBCLASSES

    def connect(self):
        """
        Connect either to NAO or V-Rep
        """
        pass

    def reset_position(self):
        """
        Method
        """
        self.joint_position  = np.zeros(len(self.active_joints))  
        self.joint_angular_v = np.zeros(len(self.active_joints))  
        self.joint_torque    = np.zeros(len(self.active_joints))


    def move_joints(self, angles):
        """
        Set the angles of active joints
        Number of elements in [angles] must match the active joints list
        """
        pass

    def get_angles(self):
        """
        Get current angles of active joints
        """
        pass

    def get_orientation(self):
        """
        Get the angular orientation of an object (roll, pitch, yaw)
        """
        pass

    def get_image(self):
        """
        Obtain an image from the camera
        """
        pass
