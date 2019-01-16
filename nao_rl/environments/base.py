
"""
@author: Andrius Bernatavicius, 2018
"""

import time
import gym
from nao_rl.utils.vrep import vrep


class VrepEnv(gym.Env):
    """ 
    Base class for all the Vrep gym environments
    Contains all the basic functions for:
        - Object control and sensor information (getting and setting joint angles)
    """

    def __init__(self, address, port):

        self.address        = address # Local IP
        self.port           = port # Port for Remote API
        self.frames_elapsed = 0
        self.max_attempts   = 10
        self.client_id      = None
        self.connected      = False
        self.running        = False
        self.scene_loaded   = False
        self.headless       = False
        self.start_stop_delay = .3


        self.modes = {'blocking'  : vrep.simx_opmode_blocking,  # Waits until the respose from vrep remote API is sent back
                      'oneshot'   : vrep.simx_opmode_oneshot,   # Sends a one time packet to vrep remote API (used for setting parameters)
                      'streaming' : vrep.simx_opmode_streaming, # Sends a signal to start sending packets every frame of the simulation back to the python client
                      'buffer'    : vrep.simx_opmode_buffer}    # To be used in combination with 'streaming' to obtain the information being streamed


    def connect(self):
        """
        Connect to a running instance of vrep at [self.address]
        and [self.port] and return the client_id
        """
        if self.connected:
            raise RuntimeError('Client is already connected.')
        e = 0
        c_id = 0
        while e < self.max_attempts:
            c_id = vrep.simxStart(self.address, self.port, True, True, 1000, 0)
            if c_id >= 0:
                self.client_id = c_id
                self.connected = True
                print 'Connection to client successful. IP: {}, port: {}, client id: {}'.format(self.address, self.port, c_id)
                break
            else:
                e += 1
                print 'Could not connect to client, attempt {}/{}...'.format(e, self.max_attempts)
                time.sleep(1)


    def disconnect(self):
        if not self.connected:
            raise RuntimeError('Client is not connected.')
        vrep.simxFinish(self.client_id)
        self.connected = False


    def load_scene(self, path):
        """
        Load the vrep scene in the specified path
        """
        if self.scene_loaded:
            raise RuntimeError('Scene is loaded already.')
        vrep.simxLoadScene(self.client_id, path, 0, self.modes['blocking'])
        self.scene_loaded = True


    def close_scene(self):
        if not self.scene_loaded:
            raise RuntimeError('Scene is not loaded')
        vrep.simxCloseScene(self.client_id, self.modes['blocking'])
        self.scene_loaded = False

    
    def start_simulation(self):
        """
        Starts the simulation and sets the according parameters
        """
        vrep.simxSynchronous(self.client_id, True)

        if self.headless:
            self.set_boolean_parameter(vrep.sim_boolparam_threaded_rendering_enabled, True)

        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)
        self.running = True
        time.sleep(self.start_stop_delay)


    def stop_simulation(self):
        """
        Stops the simulation automatically resetting it to initial state
        """
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
        self.running = False
        time.sleep(self.start_stop_delay)


    def step_simulation(self):
        """
        Advance one frame in the vrep simulation
        """
        vrep.simxSynchronousTrigger(self.client_id)
        self.frames_elapsed += 1


    def close(self):
        if self.running:
            self.stop_simulation()
        time.sleep(self.start_stop_delay)
        if self.scene_loaded:
            self.close_scene()
        time.sleep(self.start_stop_delay)
        if self.connected:
            self.disconnect()
        time.sleep(self.start_stop_delay)

    #########################
    ## Getters and setters ##
    #########################

    def set_boolean_parameter(self, parameter_id, value):
        """
        Sets a parameter of the simulation based on the parameter id
        """
        return vrep.simxSetBooleanParameter(self.client_id,
                                            parameter_id, value,
                                            vrep.simx_opmode_blocking)[0]

    def get_boolean_parameter(self, parameter_id):
        """
        Get the state of the parameter identified by [parameter_id]
        """
        return vrep.simxGetBooleanParameter(self.client_id,
                                            parameter_id,
                                            vrep.simx_opmode_blocking)[0]

    def set_float_parameter(self, parameter_id, value):
        """
        Sets a float parameter
        """
        return vrep.simxSetFloatingParameter(self.client_id,
                                             parameter_id,
                                             value,
                                             vrep.simx_opmode_blocking)[0]

    def get_handle(self, name):
        """
        Get a handle of an object identified by [name] in vrep simulation
        """
        return vrep.simxGetObjectHandle(self.client_id,
                                        name,
                                        vrep.simx_opmode_blocking)[1]

    def get_handle_multiple(self, handles):
        """
        Pauses the communication and retrieves all handles at once
        """
        vrep.simxPauseCommunication(self.client_id, True)
        handles = []
        for handle in handles:
            handles.append(self.get_handle(handle))
        vrep.simxPauseCommunication(self.client_id, False)

        return handles

    def set_joint_position(self, handle, angle):
        """
        Set a simulated joint identified by a [handle] to a specific [angle]
        """
        vrep.simxSetJointTargetPosition(self.client_id,
                                        handle,
                                        angle,
                                        vrep.simx_opmode_oneshot)

    def set_joint_position_multiple(self, handles, angles):
        """
        Pauses communication and sends all the joint angle changes in one packet
        """
        # vrep.simxPauseCommunication(self.client_id, True)
        for i, handle in enumerate(handles):
            self.set_joint_position(handle, angles[i])
        # vrep.simxPauseCommunication(self.client_id, False)


    def get_joint_angle(self, handle, mode='blocking'):
        """
        Get the current angle of a joint identified by [handle]
        By default uses a blocking call
        """
        return vrep.simxGetJointPosition(self.client_id,
                                         handle,
                                         self.modes[mode])[1]

    def get_vision_image(self, handle, mode='blocking'):
        """
        Get the image from a virtual image sensor
        """
        res, resolution, image = vrep.simxGetVisionSensorImage(self.client_id,
                                                               handle,
                                                               0,
                                                               self.modes[mode])
        return res, resolution, image

    def get_object_position(self, handle, mode='blocking'):
        """
        Get the position of an object relative to the floor
        """
        return vrep.simxGetObjectPosition(self.client_id,
                                          handle,
                                          -1,
                                          self.modes[mode])[1]

    def set_object_position(self, handle, position):
        """
        Set the position of an object relative to the floor
        """

        vrep.simxSetObjectPosition(self.client_id,
                                   handle,
                                   -1,
                                   position,
                                   vrep.simx_opmode_oneshot)

    def get_object_orientation(self, handle, mode='blocking'):
        """
        Get the angular orientation of the object in vrep
        """
        return vrep.simxGetObjectOrientation(self.client_id,
                                             handle,
                                             -1,
                                             self.modes[mode])[1]

    def set_object_orientation(self, handle, orientation):
        """
        Set the angular orientation of an object in vrep
        """
        vrep.simxSetObjectOrientation(self.client_id,
                                      handle,
                                      -1,
                                      orientation,
                                      vrep.simx_opmode_oneshot)

    def get_children(self, handle):
        """
        Returns a list of handles of all the child objects of a specified object
        """
        child_handles = []
        index = 0
        while True:
            child_handle = vrep.simxGetObjectChild(self.client_id,
                                                   handle,
                                                   index,
                                                   self.modes['blocking'])
            if child_handle == -1:
                break
            child_handles.append(child_handle)
            index += 1

        return child_handles


    def read_collision(self, collision_name):
        """
        Reads a collision of a collision object
        The collision interaction has to be created manually in the VREP scene editor
        Returns a boolean value of whether collision is present
        """
        handle = vrep.simxGetCollisionHandle(self.client_id, collision_name, self.modes['blocking'])[1]
        collision_status = vrep.simxReadCollision(self.client_id, handle, self.modes['blocking'])[1]

        return collision_status
    