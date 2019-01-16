
"""
@author: Andrius Bernatavicius, 2018

This file contains the classes to control either real or virtual NAOs
"""

import time, array
import numpy as np



class NAO(object):
    """
    Base class for NAO that contains all the functions that can be used for both real and virtual versions of NAO
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
                    active_joints += name
                else:
                    print "Invalid name of joint '{}'".format(name)

        return active_joints


    def act(self, action, movement_mode='torque', speed=1):
        """
        
        """

        if movement_mode == 'torque':
            self.joint_torque = np.clip(self.joint_torque + (action * speed / 400),
                                          -self.max_torque,
                                           self.max_torque)
            
            self.joint_angular_v = np.clip(self.joint_angular_v + self.joint_torque,
                                          -self.max_joint_velocity,
                                           self.max_joint_velocity)

            self.joint_position += self.joint_angular_v
            self.move_joints(self.joint_position)

        elif movement_mode == 'velocity':
            self.joint_angular_v = np.clip(self.joint_angular_v + (action * speed / 200),
                                          -self.max_joint_velocity,
                                           self.max_joint_velocity)

            self.joint_position += self.joint_angular_v
            self.move_joints(self.joint_position)

        elif movement_mode == 'position':
            self.joint_position += action / 50 * speed 
            self.move_joints(self.joint_position)
        else:
            print "Invalid mode for actions."
        

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


class RealNAO(NAO):
    """
    Contains the functions for control and obtaining sensor information for
    the real NAO (using ALProxies)
    Includes vision, gyroscope information
    """
    def __init__(self, ip, port, joints):
        NAO.__init__(self, joints)

        self.ip = ip
        self.port = port

        self.stiffness   = .8   # Stiffness of active motors [0-1]
        self.joint_speed = .5   # Speed of joint movement    [0-1]
        
        self.camera_proxy = None
        self.video_client = None
        self.memory_proxy = None


    def connect(self, env, resolution=0, colorSpace=11, fps=30):
        """
        Connect to NaoQI Motion, Posture and Vision proxies
        """

        from naoqi import ALProxy
        # OR self.active_joints = self.limbs[env.active_joints[0]] + self.limbs[env.active_joints[1]]

        # Motion proxies
        self.motion_proxy  = ALProxy("ALMotion", self.ip, self.port)
        self.posture_proxy = ALProxy("ALRobotPosture", self.ip, self.port)
        
        # Set initial posture
        self.motion_proxy.stiffnessInterpolation(self.active_joints, .3, .3)
        self.posture_proxy.goToPosture("Stand", 1)
        self.initial_angles = self.get_angles()
    
        # Vision and sensors
        self.camera_proxy = ALProxy('ALVideoDevice', self.ip, self.port)
        self.video_client = self.camera_proxy.subscribe("python_client", resolution, colorSpace, fps)
        self.memory_proxy = ALProxy('ALMemory', self.ip, self.port)

    
    def reset_position(self):
        self.move_joints(self.initial_angles)


    def move_joints(self, angles, blocking=False):
        """
        Moves the joints in a list to specific positions by an increment
            - [joints] and [angles] lists must be of the same length
        """
        current = self.get_angles()
        angles = list(angles)
        
        if not blocking:
            self.motion_proxy.post.changeAngles(self.active_joints, angles, self.joint_speed)
        else:
            self.motion_proxy.changeAngles(self.active_joints, angles, self.joint_speed)


    def get_angles(self, joints=None, use_sensors=True):
        """
        Gets the joint angles from ALMotion proxy
        Defaults:
            -"Body" - get all servo angles
            -"use_sensors - [bool] - use the angle sensor information
        """
        if joints is None:
            joints = self.active_joints

        angles = self.motion_proxy.getAngles(joints, use_sensors)

        return angles


    def get_image(self, remote=True):
        """
        Gets an image from the NAO Image proxy and returns a numpy array with intensity values
        """

        vc = self.camera_proxy.subscribe(str(np.random.random()), 0, 11, 30)
        if remote:
            raw_image = self.camera_proxy.getImageRemote(vc)
        else:
            raw_image = self.camera_proxy.getImageLocal(self.video_client)

        if raw_image != None:
            image = np.frombuffer(raw_image[6], dtype='%iuint8' % raw_image[2])
            image = np.reshape(image, (raw_image[1], raw_image[0], raw_image[2]))
        else:
            print "Could not obtain image from the Camera Proxy..."

        return np.squeeze(image)

    ####################
    # Additional methods

    def set_joints(self, angles, speed=.8, blocking=False, sync=False):
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


    def disconnect_vision(self):
        """
        Disconnects the camera proxy
        """
        self.camera_proxy.unsubscribe(self.video_client)


class VrepNAO(NAO):
    """
    NAO that is controlled directly through VREP API instead of the NaoQI proxies
    Allows the parralelization of the learning algorithms
    """
    def __init__(self, joints, streaming_mode=True):

        NAO.__init__(self, joints)

        self.name = '#NAO'
        self.camera_name = 'NAO_vision1'

        self.handle = None
        self.joint_handles = []
        self.camera_handle = []

        # Initial params
        self.initial_position     = None
        self.initial_orientation  = None
        self.initial_nao_position = None
        self.initial_nao_orientation = None

        self.position = np.zeros(len(self.active_joints))
        self.orientation = np.zeros(len(self.active_joints))

        self.streaming = streaming_mode
        
        # Body part position tracking and collisions

        self.body_parts_to_track  = [] # List of tracked body parts
        self.feet_collision_names = ['Collision', 'Collision0'] # Right / left foot


    def connect(self, env, use_camera=False):
        """
        Override the default connect_env function from VirtualNAO
        Includes a function to start streaming the positions of specified joints
        """
        self.env = env

        self.part_handles = {
            'LFoot' : self.env.get_handle('imported_part_18'),
            'RFoot' : self.env.get_handle('imported_part_37'),
            'Head'  : self.env.get_handle('imported_part_16_sub0'),
            'Torso' : self.env.get_handle('imported_part_20_sub0')
            }

        # Handles
        self.get_handles()
        self.handle        = self.env.get_handle(self.name)
        self.camera_handle = self.env.get_handle(self.camera_name)
        
        # Collect initial positional parameters
        self.initial_nao_position = self.env.get_object_position(self.handle)
        self.initial_nao_orientation  = self.env.get_object_orientation(self.handle)

        # Streaming of angles, orientations and positions of objects every simulation frame
        self.body_parts_to_track = self.env.body_parts_to_track
        if self.streaming:
            self.start_streaming(use_camera)

        # Position tracking
        self.track_positions()


    def reset_position(self):
        """
        Reinitializes the body position
        """
        self.joint_position  = np.zeros(len(self.active_joints))  
        self.joint_angular_v = np.zeros(len(self.active_joints))  
        self.joint_torque    = np.zeros(len(self.active_joints))  

    def move_joints(self, angles):
        """
        Override the function of NAO class to only operate in VREP
        """
        # self.joint_position += np.squeeze(angles)
        self.env.set_joint_position_multiple(self.joint_handles, self.joint_position)

    
    def get_angles(self):
        """
        Get current angles of active joints
        """
        if self.streaming:
            angles = [self.env.get_joint_angle(handle, 'buffer') for handle in self.joint_handles]
        else:
            angles = [self.env.get_joint_angle(handle) for handle in self.joint_handles]

        return np.array(angles)


    def get_orientation(self, part=None):
        """
        Get the angular orientation of a part in radians (axes [x,y,z] / [roll,pitch/yaw])
        """
        if part is None:
            part = self.handle
        else:
            part = self.part_handles[part]

        if self.streaming:
            return np.array(self.env.get_object_orientation(part, 'buffer'))
        else:
            return np.array(self.env.get_object_orientation(part))

    
    def get_image(self, mode='blocking'):
        """
        Retrieves an image from NAOs vision sensor in vrep
        """
        if self.streaming:
            _, resolution, raw_image = self.env.get_vision_image(self.camera_handle, 'buffer')
        else:
            _, resolution, raw_image = self.env.get_vision_image(self.camera_handle, mode)
        if len(raw_image) == 0:
            print "Image could not be obtained from {}".format(self.camera_name)
        else:
            image = np.array(raw_image) # Reverse list and make values [0;255]
            
            # Reshape and flip the image vertically
            image = np.flip(np.reshape(image, (resolution[1], resolution[0], 3)), 1) 
            image = np.flip(image)
            image = abs(image)

        return image

    ####################
    # Additional methods

    def get_handles(self):
        """
        Retrieve the handle (object index in the scene) from a scene in v-rep, based on the object names
        """
        # The joints of NAO are named differently in V-REP and they need to be changed to obtain the correct handles
        names = [name+'3' if 'Head' not in name else name for name in self.active_joints]
        names = [name+'#' for name in names] # Add '#' to all joint names
        handles = []

        for joint in names:
            handle = self.env.get_handle(joint)
            handles.append(handle)

        # Create a dictionary where the joint name gets associated with the corresponding handle in vrep
        self.joint_handles = handles

    def track_positions(self):
        """
        Enable tracking of specific body part positions and orientations (e.g. 'Head' or 'Torso')
        The list of tracked parts must be defined in the RL environment
        """

        positions    = []
        orientations = []
        self.body_parts_to_track = self.env.body_parts_to_track
        for part in self.body_parts_to_track:
            positions.append(self.env.get_object_position(self.part_handles[part], 'buffer'))
            orientations.append(self.env.get_object_orientation(self.part_handles[part], 'buffer'))

        self.initial_position    = dict(zip(self.body_parts_to_track, positions))
        self.initial_orientation = dict(zip(self.body_parts_to_track, orientations))


    def start_streaming(self, use_camera):
        """
        Start streaming the angles of active joints or positions/orientations of objects
        This information is then sent after every frame of the simulation
        Removes the need for sending blocking calls with vrep.simx_opmode_blocking in order 
        to obtain the joint positions and makes the simulation procedure a lot faster`

        Arguments:
            orientation : if true the angular orientation of NAO will be streamed each frame
            position    : if true the absolute position of NAO will be streamed each frame        
        """
        # Joint angles
        for joint in self.joint_handles:
            self.env.get_joint_angle(joint, 'streaming')

        # Position and orientation of NAO's convex hull
        self.env.get_object_orientation(self.handle, 'streaming')
        self.env.get_object_position(self.handle, 'streaming')

        # Track body part position and orientation
        for body_part in self.body_parts_to_track:
            self.env.get_object_position(self.part_handles[body_part], 'streaming')
            self.env.get_object_orientation(self.part_handles[body_part], 'streaming')

        if use_camera:
            self.env.get_vision_image(self.camera_handle, 'streaming')

    # Collisions and position
    def get_position(self, part=None):
        """
        Get the absolute position of a part (default - The whole NAO model)
        """
        if part is None: part = self.handle
        else: part = self.part_handles[part]

        if self.streaming:
            return self.env.get_object_position(part, 'buffer')
        else:
            return self.env.get_object_position(part)          

    def check_collisions(self):
        """
        Return a list of boolean values that indicate the collision of the feet with the floor
        """ 
        right = self.env.read_collision(self.feet_collision_names[0])
        left  = self.env.read_collision(self.feet_collision_names[1])

        return [right, left]


class VirtualNAO(RealNAO, VrepNAO):
    """
    Agent for using Naoqi within vrep
    Allows inbuilt naoqi behaviours for walking and posture to be used within vrep
    Useful for navigation tasks

    """

    def __init__(self, ip, port):
        RealNAO.__init__(self, ip, port)
        VrepNAO.__init__(self, ip, port)

        self.name   = 'NAO#'
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
