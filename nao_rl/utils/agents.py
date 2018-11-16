
"""
@author: Andrius Bernatavicius, 2018
"""

from naoqi import ALProxy
import time, array
import numpy as np

############
## AGENTS ##
############

class NAO(object):
    """
    Base class for NAO that contains all the functions that can be used for both real and virtual versions of NAO
    Movement commands are issued using ALMotion Proxy
    """
    def __init__(self, ip, port):

        self.ip = ip
        self.port = port

        # Motion and posture proxies of the virtual Nao created by naoqi-bin
        self.motionProxy = None
        self.postureProxy = None

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
                       "RArm": self.joint_names[19::],
                       "Body": self.joint_names} 

        self.body = ["Head", "LArm", "LLeg", "RLeg", "RArm"]


    def connect_naoqi(self, naoqi_ip, naoqi_port):
        """
        Connect to NaoQI
        """
        self.motionProxy = ALProxy("ALMotion", self.ip, self.port)
        self.postureProxy = ALProxy("ALRobotPosture", self.ip, self.port)

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

    def connect_vision(self, fps):
        resolution = 2    # VGA
        colorSpace = 11   # RGB
        self.video_client = self.cameraProxy.subscribe("python_client", resolution, colorSpace, fps)

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

        self.name = 'NAO#'
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
            # print "Getting handle {} - number : {}".format(joint, handle)
            handles.append(handle)
    
        # Create a dictionary where the joint name gets associated with the corresponding handle in vrep
        self.handles = dict(zip(self.joint_names, handles))
        

    def select_joints(self, targets="Body"):
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

        indices, handles = self.select_joints(targets)
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

class VrepNAO(VirtualNAO):
    """
    NAO that is controlled directly through VREP API instead of the NaoQI proxies
    """
    def __init__(self, streaming_mode=True):
        ip   = None
        port = None
        VirtualNAO.__init__(self, ip, port)

        # Initial params
        self.initial_position = []
        self.initial_orientation = []
        self.initial_nao_position = None

        self.position = np.zeros(len(self.handles))
        self.orientation = []
        
        self.active_joints = None
        self.active_joint_position = None
        self.streaming = streaming_mode

    def reset_position(self):
        
        _, all_handles = self.select_joints()
        # Position and orientation
        for i, handle in enumerate(all_handles):
            self.env.set_object_position(handle, self.initial_position[i])
            self.env.set_object_orientation(handle, self.initial_orientation[i])

        # Joint angles
        self.active_joint_position = np.zeros(len(self.active_joints))
        
        self.env.set_joint_position_multiple(all_handles, np.zeros(len(self.handles)))

    def connect(self, env, joints, position=True, orientation=True):
        """
        Override the default connect_env function from VirtualNAO
        Includes a function to start streaming the positions of specified joints
        """
        self.env = env
        self.get_handles()
        self.handle = self.env.get_handle(self.name)
        _, self.active_joints = self.select_joints(joints)
        _, all_handles = self.select_joints()

        # Collect initial positional parameters
        self.initial_nao_position = self.env.get_object_position(self.handle)
        for handle in all_handles:
            self.initial_position.append(self.env.get_object_position(handle))
            self.initial_orientation.append(self.env.get_object_orientation(handle))

        self.active_joint_position = np.zeros(len(self.active_joints))
        if self.streaming:
            self.start_streaming(position, orientation)
        # self.reset_position()



    def move_joints(self, angles):
        """
        Override the function of NAO class to only operate in VREP 
        """
        self.active_joint_position += np.squeeze(angles)
        self.env.set_joint_position_multiple(self.active_joints, self.active_joint_position)


    def get_angles(self):
        # _, handles = self.select_joints(joints)
        angles = np.zeros(len(self.active_joints))
        if self.streaming:
            for i, handle in enumerate(self.active_joints):
                angles[i] = self.env.get_joint_angle(handle, 'buffer')
        else:
            for i, handle in enumerate(self.active_joints):
                angles[i] = self.env.get_joint_angle(handle)

        return angles

    def get_orientation(self):
        if self.streaming:
            return self.env.get_object_orientation(self.handle, 'buffer')
        else:
            return self.env.get_object_orientation(self.handle)

    def get_position(self):
        if self.streaming:
            return self.env.get_object_position(self.handle, 'buffer')
        else:
            return self.env.get_object_position(self.handle)



    def start_streaming(self, orientation=False, position=False):
        """
        Start streaming the positions of active joints
        Vrep will send joint angle information in every time step
        Removes the need for sending blocking calls with vrep.simx_opmode_blocking in order 
        to obtain the joint positions

        Arguments:
            orientation : if true the angular orientation of NAO will be streamed each frame
            position    : if true the absolute position of NAO will be streamed each frame        
        """
        for joint in self.active_joints:
            self.env.get_joint_angle(joint, 'streaming')
        
        if orientation:
            self.env.get_object_orientation(self.handle, 'streaming')
        
        if position:
            self.env.get_object_position(self.handle, 'streaming')






