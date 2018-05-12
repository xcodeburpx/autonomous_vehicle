# Import libraries
# It will be modified

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions
import time

#Environment Macros
WIDTH = 900
HEIGHT = 800
BG_COLOR = THECOLORS['black']
SENSORS_COLOR = THECOLORS['white']
ALARM_COLOR = THECOLORS['red']
FLAGS = pygame.DOUBLEBUF

# Unit Macros
UNIT_X = WIDTH/4        # Unit starting point - x value
UNIT_Y = HEIGHT/2       # Unit starting point - y value
UNIT_R = 20             # Unit dimensions - radius(circle type)

# Sensor Macros
DRAW_START = UNIT_R + 10                # Starting point for sensors
DRAW_LENGTH = 100                       # The length of the sensor
DRAW_STOP = DRAW_START + DRAW_LENGTH    # End point of the sensor
COLL_THRESH = -0.92                     # Collision threshold - min. value of the sensor
NEG_REWARD = -100                       # Negative reward

# Network macros
OBSERVATION_SPACE = 5
ACTION_SPACE = 4

# Learning_script macros
MEMORY_SIZE = 4000
MAX_STEPS = 250000
DRAWING = True


# Additional dictionaries
# First is just enumeration dictionary
# Second is used to modify the reward
ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

ACTION_DECREMENT = {
    0: 1.,
    1: 1.4,
    2: 1.4,
    3: 1.4
}

