from nao_rl.agents import RealNAO, VrepNAO

class VirtualNAO(VrepNAO):
    """
    Agent for using Naoqi proxies within VREP
    Allows inbuilt naoqi behaviours for walking and posture to be used within vrep
    Useful for navigation tasks

    """

    def __init__(self, ip, port):
        # RealNAO.__init__(self, ip, port)
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
