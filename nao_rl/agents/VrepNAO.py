from nao_rl.agents import NAO
import numpy as np


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
        image = None
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

