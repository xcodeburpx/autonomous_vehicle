import pygame
import numpy as np
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions
import time

#Environment Macros
WIDTH = 1000
HEIGHT = 900
BG_COLOR = THECOLORS['black']
SENSORS_COLOR = THECOLORS['white']
ALARM_COLOR = THECOLORS['red']
FLAGS = pygame.DOUBLEBUF

# Unit Macros
UNIT_X = WIDTH/6        # Unit starting point - x value
UNIT_Y = HEIGHT/2       # Unit starting point - y value
# Unit size
UNIT_WITDH = 20
UNIT_LENGTH = int(1.34 * UNIT_WITDH)
UNIT_CROSS =  np.int(np.sqrt(UNIT_WITDH**2 + UNIT_LENGTH**2))

# Sensor Macros
# TODO: RADIUS FOR A RECTANGLE
DRAW_START_FB =  UNIT_CROSS-4           # Starting point for sensors
DRAW_LENGTH_FB = 100                       # The length of the sensor
DRAW_STOP_FB = DRAW_START_FB + DRAW_LENGTH_FB    # End point of the sensor

DRAW_START_LR = UNIT_WITDH-2             # Starting point for sensors
DRAW_LENGTH_LR = 100                       # The length of the sensor
DRAW_STOP_LR = DRAW_START_LR + DRAW_LENGTH_LR    # End point of the sensor

COLL_THRESH = -0.90                     # Collision threshold - min. value of the sensor
BASE_REWARD = 10
NEG_REWARD = -100                       # Negative reward

# Network macros
OBSERVATION_SPACE = 8
ACTION_SPACE = 4

# Learning_script macros
MEMORY_SIZE = 4000
MAX_STEPS = 255000
DRAWING = True
SAVE_THRESH = 5000


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
    0: 10,
    1: 15,
    2: 15,
    3: 15
}