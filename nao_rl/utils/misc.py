"""
Author: Andrius Bernatavicius, 2018
"""

import random

class Ball:
    """
    Simple ball inside vrep
    (Functionality still needed)
    """
    def __init__(self, name):
        self.name = name
        self.handle = None
        self.position = None
        self.initial_position = None
        self.velocity = [0, 0, 0]
        self.velocity_bounds = [-.002, .002]
        self.momentum = [0, 0, 0]
        self.momentum_bounds = [-.001, .001]

        # Connect to the vrep environment
        self.env = None

    def connect_env(self, env):
        self.env = env
        self.handle = self.env.get_handle(self.name)
        self.initial_position = self.get_position()
        self.position = self.initial_position
        # self.update_position()

    def get_position(self):
        return self.env.get_object_position(self.handle)

    def reset_position(self):
        self.env.set_object_position(self.handle, self.initial_position)

    def update_position(self, position=[]):
        """
        Perform a random movement
        """
        self.position = [self.position[x] + self.velocity[x] for x in range(3)]
        self.env.set_object_position(self.handle, self.position)

    def update_velocity(self):
        self.velocity = [self.velocity[x] + self.momentum[x] for x in range(3)]

    def random_motion(self):
        axes = [0, 2] # Axes for motion
        for axis in axes:
            self.momentum[axis] += (random.random() - 0.5) * 0.0001
            if self.momentum[axis] < self.momentum_bounds[0]: self.momentum[axis] = self.momentum_bounds[0]
            if self.momentum[axis] > self.momentum_bounds[1]: self.momentum[axis] = self.momentum_bounds[1]
        self.update_velocity()
        self.update_position()

    
    def restart(self):
        self.momentum = [0, 0, 0]
        self.velocity = [0, 0, 0]
        self.position = self.initial_position
        self.reset_position()
        
    def calculate_force(self):
        pass

    def update_position(self):
        pass

    def start_orbiting(self):
        """
        Starts orbiting from a random position in [y,z] plane with a random initial velocity
        Orbits around the center of NAO's initial visual field

        """
