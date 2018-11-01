
"""
@author: Andrius Bernatavicius, 2018
"""

from naoqi import ALProxy
import time, array
import numpy as np
from PIL import Image as I

############
## AGENTS ##
############

class NAO(object):
    """
    Base class for NAO that includes all the functions that can be used for both real and virtual versions of NAO
    Movement commands are issued using ALMotion Proxy
    """
    def __init__(self, ip, port):

        self.ip = ip
        self.port = port

        # Motion and posture proxies of the virtual Nao created by naoqi-bin
        self.motionProxy = ALProxy("ALMotion", self.ip, self.port)
        self.postureProxy = ALProxy("ALRobotPosture", self.ip, self.port)

        ####  Joints
        # Ordering corresponds to the one provided by Aldebaran Robotics
        self.joint_names =  [# "Head"
                            'HeadYaw', 'HeadPitch',
                            # "LArm"
                            'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
                            # "LLeg"
                            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
                            # "RLeg"
                            'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
                            # "RArm"
                            'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw'
                            ]
        
        #   Assign an index to each joint name (this ordering corresponds to list of angles returned by the function call
        # of motionProxy.getAngles())
        self.joint_indices = dict(zip(self.joint_names, range(24)))
    
        # A dictionary for quickly getting the names of joints in each limb
        self.limbs =  {"Head": self.joint_names[0:2],
                       "LArm": self.joint_names[2:7],
                       "LLeg": self.joint_names[7:13],
                       "RLeg": self.joint_names[13:19],
                       "RArm": self.joint_names[19::]} 

        self.body = ["Head", "LArm", "LLeg", "RLeg", "RArm"]


    def initialize(self, reinitialize=False):
        """
        Initialize NAO by setting the stiffness, posture and other parameters
        """ 
        self.motionProxy.stiffnessInterpolation('Body', 1, 1)
        self.postureProxy.goToPosture("Stand", 1)

    def move_to(self, x, y, theta, blocking=False):
        """
        Moves to a specific location using the inbuilt walking behavior from ALMotion proxy 
            - [x, y, rotation] 
            - [blocking] determines whether it's a blocking call (no other commands can be issued until the move command is finished
            during a blocking call) 
        """ 
        if not blocking:
            self.motionProxy.post.moveTo(x, y, theta)
        else:
            self.motionProxy.moveTo(x, y, theta)

    def move_joints(self, joints, angles, speed=.8, blocking=False):
        """
        Moves the joints in a list to specific positions by an increment
            - [joints] and [angles] lists must be of the same length
        """ 
        if not blocking:
            self.motionProxy.post.changeAngles(joints, angles, speed)
        else:
            self.motionProxy.changeAngles(joints, angles, speed)

    def set_joints(self, joints, angles, speed=.8, blocking=False, sync=False):
        """
        Sets the joints to specific positions
        """
        if not blocking:
            self.motionProxy.post.setAngles(joints, angles, speed)
        else:
            self.motionProxy.setAngles(joints, angles, speed)


    def get_angles(self, joints="Body", use_sensors=False, graspers=False):
        """
        Gets the joint angles from ALMotion proxy
        Defaults:
            -"Body" - get all servo angles
            -"use_sensors - [bool] - use the angle sensor information
        """
        angles = self.motionProxy.getAngles(joints, use_sensors)
        if not graspers:
            del angles[7]
        return angles


class RealNAO(NAO):
    """
    Contains the functions for control and obtaining sensor information for 
    the real NAO (using ALProxies)
    """
    def __init__(self, ip, port):
        NAO.__init__(self, ip, port)
        self.cameraProxy = ALProxy('ALVideoDevice', self.ip, self.port)
        self.video_client = None

    def connect_vision(self):
        resolution = 2    # VGA
        colorSpace = 11   # RGB
        self.video_client = self.cameraProxy.subscribe("python_client", resolution, colorSpace, 5)

    def disconnect_vision(self):
        self.cameraProxy.unsubscribe(self.video_client)

    def get_image(self, remote=False):
        """
        """
        if remote:
            image = self.cameraProxy.getImageRemote(self.video_client)
        else:
            image = self.cameraProxy.getImageLocal(self.video_client)

        image = I.fromstring("RGB", (image[0], image[1]), image[6])

        return image


class VirtualNAO(NAO):
    """
    Contains all the functions for:
        - Controlling NAO in vrep,
        - Retrieving agent's positional and sensor information from vrep
        - Retrieving data from the environment, and setting the environment parameters

        Needs to be connected to an environment using self.connect_env(env) 
    """

    def __init__(self, ip, port):
        NAO.__init__(self, ip, port)

        self.name = 'NAO'
        self.handle = None

        # Camera 
        self.camera_name = 'NAO_vision1'
        
        # Handles of joints in VREP (env needs to be connected first with self.connect_env to obtain them)
        self.handles = []
        self.env = None


    def connect_env(self, env):
        """
        Links the instance of VirtualNAO object to a vrep environment so that all VrepEnv methods can be called
        """ 
        self.env = env
        self.get_handles()
        self.handle = self.env.get_handle(self.name)
        self.camera_handle = self.env.get_handle(self.camera_name)
        # self.env.get_vision_image(self.camera_handle, 'streaming')


    def get_handles(self):
        """
        Retrieve the handle (object index in the scene) from a scene in v-rep, based on the object names
        """
        # The joints of NAO are named differently in vrep and they need to be changed to obtain the correct handles
        vrep_joint_names = [name+'3' if i > 1 else name for i, name in enumerate(self.joint_names)] # Add '3' non-head joint names
        vrep_joint_names = [name+'#' for name in vrep_joint_names] # Add '#' to all joint names
        handles = []

        for joint in vrep_joint_names:
            handle = self.env.get_handle(joint)
            print "Getting handle {} - number : {}".format(joint, handle)
            handles.append(handle)
        # Create a dictionary where the joint name gets associated with the corresponding handle in vrep
        self.handles = dict(zip(self.joint_names, handles))
        

    def include(self, targets="Body"):
        """
        Retrieve lists of indices and handles in the target limbs
        args:
            - targets - target limbs to use, default - body
            alternative a list of target limbs can be passed
        """ 
        if type(targets) is not list: targets = [targets]
            
        indices = []
        handles = []
        for limb in targets:
            for joint in self.limbs[limb]:
                indices.append(self.joint_indices[joint])
                handles.append(self.handles[joint])

        return indices, handles


    def naoqi_vrep_sync(self, targets=["Head", "LArm", "LLeg", "RLeg", "RArm"]):
        """
        Capture the current joint positions of the Nao simulated by naoqi-bin and send them to vrep.
        args:
            targets: target limbs for control
        """
        # assert targets in self.body

        indices, handles = self.include(targets)
        angles = self.get_angles(use_sensors=True)
        for i, handle in enumerate(handles):
            self.env.set_joint_position(handle, angles[indices[i]])


    def move(self, distance_x=0.5, distance_y=0, angle=0, sync=False, max=False):
        """ 
        Function for moving a specified distance in all 4 directions or rotation
        args: 
            distance_x: [-n:n] move forward/backwards
            distance_y: [-n:n] strafe right/left
            angle: [-3.14:3.14] rotate clockwise/anticlockwise 
            sync: In case of synchronous simulation must be set to true to advance to the next frame
            max: uses a different gait if true
        """
        
        print "Moving: X: {}; Y: {}, Rotation: {}...".format(distance_x, distance_y, angle)

        if max:
            self.motionProxy.post.moveTo(distance_x, distance_y, angle, self.motionProxy.getMoveConfig("Max"))
        else:
            self.motionProxy.post.moveTo(distance_x, distance_y, angle)
        
        while self.motionProxy.moveIsActive():
            self.naoqi_vrep_sync()
            if sync:
                self.env.step_simulation()


    def discrete_move(self, action_id):
        """
        * MOVE TO nao_navigation.py *
        Executes one of the following actions:
            - [move forward by 0.2]
            - [turn right by 0.2]
            - [turn left by 0.2]
        """
        if action_id == 0: action = [.1,0,0]
        if action_id == 1: action = [0,0,-.2]
        if action_id == 2: action = [0,0,.2]

        self.move_to(action[0], action[1], action[2], True)


    def posture(self, posture):
        """
        Set NAOs posture - ["Stand", "StandInit", "StandZero"]
        """
        print "Setting posture..."
        self.postureProxy.post.goToPosture(posture, 1)
        for i in range(20):
            self.naoqi_vrep_sync()
            self.env.step_simulation()
        print "Done."

    ############
    ## VISION ##
    ############
    #  
    def get_image(self, attempts):
        """
        Retrieves an image from NAOs vision sensor in vrep
        """
        image = []
        _, resolution, image = self.env.get_vision_image(self.camera_handle)
        if len(image) == 0:
            raise RuntimeError('The image could not be retrieved from {}'.format(self.camera_name))
        else:
            image_byte_array = array.array('b',image)
            image = I.frombuffer("RGB", (resolution[0],resolution[1]), image_byte_array, "raw", "RGB", 0, 1)

        return image, resolution


