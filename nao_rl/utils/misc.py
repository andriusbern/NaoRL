"""
Author: Andrius Bernatavicius, 2018
"""

import random
import numpy as np

class Ball(object):
    """
    Simple ball inside vrep
    """
    def __init__(self, name):
        self.env = None
        self.name     = name
        self.handle   = None
        self.position = None
        self.initial_position = None
        self.velocity     = np.zeros(3)
        self.momentum     = np.zeros(3)
        self.max_velocity = .02
        self.max_momentum = .005
        self.boundaries   = []
        self.motion_axes  = [1, 2] # Axes for motion [x,y,z] -> [0, 1, 2]

    def connect_env(self, env):
        self.env = env
        self.handle = self.env.get_handle(self.name)
        self.initial_position = np.array(self.get_position())
        self.position = np.copy(self.initial_position)
        window_size = .25
        self.boundaries = [self.initial_position[1] - window_size, self.initial_position[1] + window_size,
                           self.initial_position[2] - window_size, self.initial_position[2] + window_size]

    def get_position(self):
        return self.env.get_object_position(self.handle)

    def reset_position(self):
        self.env.set_object_position(self.handle, self.initial_position)

    def update_position(self):
        """
        Perform a random movement
        """
        # Reverse velocity vector if the ball hits the defined boundaries
        if self.position[1] < self.boundaries[0] or self.position[1] > self.boundaries[1]:
            self.velocity[1] *= -1
            self.momentum[1] = 0
        if self.position[2] < self.boundaries[2] or self.position[2] > self.boundaries[3]:
            self.velocity[2] *= -1
            self.momentum[2] = 0

        self.position += self.velocity
        
        self.env.set_object_position(self.handle, self.position)

    def update_velocity(self):
        self.velocity += self.momentum
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        

    def random_motion(self):
        
        # Modify the momentum vector along the specified axes
        for axis in self.motion_axes:
            self.momentum[axis] += (random.random() - 0.5) * 0.0002

        self.momentum = np.clip(self.momentum, -self.max_momentum, self.max_momentum)
        self.update_velocity()
        self.update_position()
    
    def restart(self):
        self.momentum = np.zeros(3)
        self.velocity = np.zeros(3)
        self.position = np.copy(self.initial_position)
        self.reset_position()
        
