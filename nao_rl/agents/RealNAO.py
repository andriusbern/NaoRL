from nao_rl.agents import NAO
import numpy as np
import nao_rl

class RealNAO(NAO):
    """
    Contains the functions for control and obtaining sensor information for
    the real NAO (using ALProxies)
    Includes vision, gyroscope information
    """
    def __init__(self, joints, ip=None, port=None):
        NAO.__init__(self, joints)

        if ip is None:
            self.ip = nao_rl.settings.REAL_NAO_IP
        else:
            self.ip = ip

        if port is None:
            self.port = nao_rl.settings.REAL_NAO_PORT
        else:
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
