
"""
@author: Andrius Bernatavicius, 2018
"""

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
        self.motion_proxy = None
        self.posture_proxy = None
        self.stiffness = .8  # Stiffness of active motors

        ####  Joints
        # Ordering corresponds to the one provided by Aldebaran Robotics
        self.joint_names = [# "Head"
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
        # of motion_proxy.getAngles())
        self.joint_indices = dict(zip(self.joint_names, range(24)))

        # A dictionary for quickly getting the names of joints in each limb
        self.limbs = {"Head": self.joint_names[0:2],
                      "LArm": self.joint_names[2:7],
                      "LLeg": self.joint_names[7:13],
                      "RLeg": self.joint_names[13:19],
                      "RArm": self.joint_names[19::],
                      "Body": self.joint_names}

        self.body = ["Head", "LArm", "LLeg", "RLeg", "RArm"]

        # Active joints
        self.active_joints = None
        self.active_joint_position = None

    # Methods to be overriden


class RealNAO(NAO):
    """
    Contains the functions for control and obtaining sensor information for
    the real NAO (using ALProxies)
    Includes vision, gyroscope information
    """
    def __init__(self, ip, port):
        
        NAO.__init__(self, ip, port)
        self.video_client = None


    def connect(self, naoqi_ip, naoqi_port, resolution=0, colorSpace=0, fps=30):
        """
        Connect to NaoQI Motion, Posture and Vision proxies

        Connects to the vision proxy with specified parameters
        ---
        Resolutions:
            Value:    Resolution:
            8  kQQQQVGA  40x30
            7  kQQQVGA   80x60
            0  kQQVGA    160x120
            1  kQVGA     320x240
            2  kVGA      640x480
            3  k4VGA     1280x960
        """

        from naoqi import ALProxy
        # Motion
        self.motion_proxy = ALProxy("ALMotion", self.ip, self.port)
        self.posture_proxy = ALProxy("ALRobotPosture", self.ip, self.port)
        self.motion_proxy.stiffnessInterpolation(self.active_joints, 1, self.stiffness)
        self.posture_proxy.goToPosture("Stand", 1)

        # Vision
        self.camera_proxy = ALProxy('ALVideoDevice', self.ip, self.port)
        self.video_client = self.camera_proxy.subscribe("python_client", resolution, colorSpace, fps)

 
    def get_angles(self, joints="Body", use_sensors=True):
        """
        Gets the joint angles from ALMotion proxy
        Defaults:
            -"Body" - get all servo angles
            -"use_sensors - [bool] - use the angle sensor information
        """
        angles = self.motion_proxy.getAngles(self.active_joints, use_sensors)

        return angles

    def move_joints(self, joints, angles, speed=.8, blocking=False):
        """
        Moves the joints in a list to specific positions by an increment
            - [joints] and [angles] lists must be of the same length
        """
        current = self.get_angles()
        angles = np.array(angles) + np.squeeze(angles)
        if not blocking:
            self.motion_proxy.post.changeAngles(self.active_joints, angles, speed)
        else:
            self.motion_proxy.changeAngles(joints, angles, speed)

    def set_joints(self, joints, angles, speed=.8, blocking=False, sync=False):
        """
        Sets the joints to specific positions
        """
        if not blocking:
            self.motion_proxy.post.setAngles(self.active_joints, angles, speed)
        else:
            self.motion_proxy.setAngles(self.active_joints, angles, speed)

    def move_to(self, x, y, theta, blocking=False):
        """
        Moves to a specific location using the inbuilt walking behavior from ALMotion proxy 
            - [x, y, rotation]
            - [blocking] determines whether it's a blocking call (no other commands can be issued until the move command is finished
            during a blocking call)
        """
        if not blocking:
            self.motion_proxy.post.moveTo(x, y, theta)
        else:
            self.motion_proxy.moveTo(x, y, theta)

    # Vision

    def get_image(self, remote=True):
        """
        Gets an image from the NAO Image proxy and returns a numpy array with intensity values
        """

        if remote:
            raw_image = self.camera_proxy.getImageRemote(self.video_client)
        else:
            raw_image = self.camera_proxy.getImageLocal(self.video_client)

        if raw_image != None:
            image = np.frombuffer(raw_image[6], dtype='%iuint8' % raw_image[2])
            image = np.reshape(image, (raw_image[1], raw_image[0], raw_image[2]))
        else:
            print "Could not obtain image..."
            time.sleep(1)
            self.get_image()

        return image

    def disconnect_vision(self):
        """
        Disconnects the camera proxy
        """
        self.camera_proxy.unsubscribe(self.video_client)


class VirtualNAO(NAO):
    """
    Agent for using Naoqi within vrep
    Allows inbuilt naoqi behaviours for walking and posture to be used within vrep
    Useful for navigation tasks

    """

    def __init__(self, ip, port):
        NAO.__init__(self, ip, port)

        self.name = 'NAO#'
        self.handle = None

        # Camera
        self.camera_name = 'NAO_vision1'

        # Handles of joints in VREP (env needs to be connected first with self.connect_env to obtain them)
        self.handles = []
        self.camera_handle = None
        self.env = None
        

    def connect(self, env):
        """
        Links the instance of VirtualNAO object to a vrep environment so that all VrepEnv methods can be called
        """
        self.env = env
        self.get_handles()
        self.handle = self.env.get_handle(self.name)
        self.camera_handle = self.env.get_handle(self.camera_name)
        # self.env.get_vision_image(self.camera_handle, 'streaming')

    def select_joints(self, targets="Body"):
        """
        Retrieve lists of indices and handles in the target limbs
        args:
            - targets - target limbs to use, default - body
            alternatively a list of target limbs can be passed
        """ 
        if not isinstance(targets, list):
            targets = [targets]
            
        indices = []
        handles = []
        for limb in targets:
            for joint in self.limbs[limb]:
                indices.append(self.joint_indices[joint])
                handles.append(self.handles[joint])

        return indices, handles


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
            self.motion_proxy.post.moveTo(distance_x, distance_y, angle, self.motion_proxy.getMoveConfig("Max"))
        else:
            self.motion_proxy.post.moveTo(distance_x, distance_y, angle)

        while self.motion_proxy.moveIsActive():
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
        self.posture_proxy.post.goToPosture(posture, 1)
        for _ in range(20):
            self.naoqi_vrep_sync()
            self.env.step_simulation()
        print "Done."

    ############
    ## VISION ##
    ############

    def get_image(self, mode='blocking'):
        """
        Retrieves an image from NAOs vision sensor in vrep
        """
        _, resolution, raw_image = self.env.get_vision_image(self.camera_handle, mode)
        if len(raw_image) == 0:
            raise RuntimeError('The image could not be retrieved from {}'.format(self.camera_name))
        else:
            image = np.array(raw_image) # Reverse list and make values [0;255]
            
            # Reshape and flip the image vertically
            image = np.flip(np.reshape(image, (resolution[1], resolution[0], 3)), 1) 
            image = np.flip(image)
            image = abs(image)

        return image, resolution


class VrepNAO(VirtualNAO):
    """
    NAO that is controlled directly through VREP API instead of the NaoQI proxies
    Allows the parralelization of the learning algorithm
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
        
        # Body part position tracking and collisions
        self.feet_collision_names = ['Collision', 'Collision0'] # Right / left foot
        self.body_parts   = [] # List of tracked body parts
        self.part_handles = {} # Handles of tracked body parts
        

    def connect(self, env, position=True, orientation=True):
        """
        Override the default connect_env function from VirtualNAO
        Includes a function to start streaming the positions of specified joints
        """
        self.env = env
        self.get_handles()
        self.handle = self.env.get_handle(self.name)
        _, self.active_joints = self.select_joints(env.active_joints)
        self.active_joint_position = np.zeros(len(self.active_joints))

        # Camera
        self.handle = self.env.get_handle(self.name)
        self.camera_handle = self.env.get_handle(self.camera_name)

        # Collect initial positional parameters
        self.initial_nao_position = self.env.get_object_position(self.handle)

        # Position tracking
        if env.body_parts is not None:
            self.track_positions()

        # Streaming of angles as positions every simulation frame
        if self.streaming:
            self.start_streaming(position, orientation)

    def track_positions(self):
        """
        Positions of body parts to track
        The list of tracked parts must be defined in the RL environment
        """
        self.part_handles = {'LFoot' : self.env.get_handle('imported_part_18'),
                             'RFoot' : self.env.get_handle('imported_part_37'),
                             'Head'  : self.env.get_handle('imported_part_16_sub0')}
        self.body_parts = self.env.body_parts
        positions = [self.env.get_object_position(self.part_handles[handle]) for handle in self.body_parts]
        self.initial_position = dict(zip(self.body_parts, positions))

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

    def start_streaming(self, orientation=False, position=False):
        """
        Start streaming the angles of active joints or positions of objects
        This information is then sent after every frame of the simulation
        Removes the need for sending blocking calls with vrep.simx_opmode_blocking in order 
        to obtain the joint positions

        Arguments:
            orientation : if true the angular orientation of NAO will be streamed each frame
            position    : if true the absolute position of NAO will be streamed each frame        
        """
        for joint in self.active_joints:
            self.env.get_joint_angle(joint, 'streaming')

        # Track body parts
        for body_part in self.body_parts:
            self.env.get_object_position(self.part_handles[body_part], 'streaming')

        if orientation:
            self.env.get_object_orientation(self.handle, 'streaming')

        if position:
            self.env.get_object_position(self.handle, 'streaming')


    # Motion and position

    def get_angles(self):
        """
        Get current angles of active joints
        """
        if self.streaming:
            angles = [self.env.get_joint_angle(handle, 'buffer') for handle in self.active_joints]
        else:
            angles = [self.env.get_joint_angle(handle) for handle in self.active_joints]

        return np.array(angles)

    def get_body_part_position(self, part):
        """
        Returns the position of a body part
        """
        return self.env.get_object_position(self.part_handles[part])

    def get_orientation(self, part=None):
        """
        Get the angular orientation of a part in radians (axes [x,y,z])
        """
        if part is None:
            part = self.handle
        else:
            part = self.part_handles[part]

        if self.streaming:
            return np.array(self.env.get_object_orientation(part, 'buffer'))
        else:
            return np.array(self.env.get_object_orientation(part))

    def get_position(self, part=None):
        """
        Get the absolute position of a part
        """
        if part is None: part = self.handle
        else: part = self.part_handles[part]

        if self.streaming:
            return self.env.get_object_position(part, 'buffer')
        else:
            return self.env.get_object_position(part)          

    def move_joints(self, angles):
        """
        Override the function of NAO class to only operate in VREP
        """
        self.active_joint_position += np.squeeze(angles)
        self.env.set_joint_position_multiple(self.active_joints, self.active_joint_position)

    def reset_position(self):
        """
        Reinitializes the body position
        """
        self.active_joint_position = np.zeros(len(self.active_joints))  

    def check_collisions(self):
        
        right = self.env.read_collision(self.feet_collision_names[0])
        left  = self.env.read_collision(self.feet_collision_names[1])

        return [right, left]