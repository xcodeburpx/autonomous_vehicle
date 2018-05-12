import math
import numpy as np

from macro_utils import *


# Unit class - our circle car
# TODO: Rectangle car
class Unit:

    def __init__(self, env, x, y, r, name="user", color=THECOLORS['yellow']):

        # Sensor parameters - angle
        self.sensors = [-math.pi / 2., -math.pi / 4., 0., math.pi / 4., math.pi / 2.]
        self.sensors_offset = math.pi / 2

        # Position in the environment
        self.positions = []

        # Other paramenters
        self.create(env, x, y, r, name, color)

    def create(self, env, x, y, r, name, color):
        # Initial variables
        self.name = name
        self.speed = 0
        self.mass = 1

        # Objects have moment
        self.moment = pymunk.moment_for_circle(self.mass, 0, r, (0, 0))
        print(self.moment)
        self.body = pymunk.Body(self.mass, self.moment)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, r)
        self.shape.color = color
        self.shape.elasticity = 1.0

        # Check this variable
        self.body.angle = -math.pi / 8

        self.direction = Vec2d(1, 0).rotated(self.body.angle)
        self.body.apply_impulse_at_local_point(self.direction)
        self.env = env
        self.env.space.add(self.body, self.shape)

    # Method - to move our object
    def move(self, speed=None, angle=None):

        # First - if we have no speed
        if not speed:
            # If we have no angle
            if not angle:
                self.speed = 30
                self.body.velocity = self.direction * self.speed
            else:
                self.speed = 30
                self.body.angle -= angle
                self.direction = Vec2d(1, 0).rotated(self.body.angle)
                self.body.velocity = self.direction * self.speed
        else:
            # If we have no angle
            if not angle:
                self.speed = speed
                self.body.velocity = self.direction * self.speed
            else:
                self.speed = speed
                self.body.angle -= angle
                self.direction = Vec2d(1, 0).rotated(self.body.angle)
                self.body.velocity = self.direction * self.speed

        #self.body.update_position(self.body, 0.01)

    # Method - get sensor data - collect state values
    def get_sensor_data(self):

        data = []  # Collected data - normalized state values
        unit_x, unit_y = self.body.position  # Get actual position
        draw_x, draw_y = 0, 0

        for sangle in self.sensors:
            draw_end = DRAW_STOP

            # Get actual value of angle
            sen_angle = math.fmod(-self.body.angle + sangle + self.sensors_offset, 2 * math.pi)
            dx = math.sin(sen_angle)
            dy = math.cos(sen_angle)

            # Get data every step
            step = 1

            # main loop
            for i in range(DRAW_START, draw_end, step):

                # Get point of measurement
                x = unit_x + dx * i
                y = HEIGHT - (unit_y + dy * i)

                if i == DRAW_START:
                    draw_x = x
                    draw_y = y

                # If it is out of bound
                if x >= WIDTH or y >= HEIGHT or x < 0 or y < 0:
                    # Option to draw
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (
                        int(draw_x), int(draw_y)), (int(x), int(y)), 1)

                    # Calculate ceil value of point - normalized
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))
                    val = val / DRAW_LENGTH * 1.0 - 1

                    # Optional threshold
                    if draw_end != DRAW_STOP:
                        val += 0.5
                    data.append(val)
                    break

                # End of the sensor
                if i == draw_end - step:
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (int(draw_x), int(draw_y)), (int(x), int(y)), 1)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2)) / 50.0 - 1
                    if draw_end != DRAW_STOP:
                        val += .5
                    data.append(val)
                    break

                # Get color at this point
                color = self.env.display.get_at((int(x), int(y)))

                # If color is different than scene or sensor color
                if color == BG_COLOR or color == SENSORS_COLOR or color == ALARM_COLOR:
                    continue

                else:
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (
                        int(draw_x), int(draw_y)), (int(x), int(y)), 10)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2)) / 50.0 - 1
                    if draw_end != DRAW_STOP:
                        val += .5
                    data.append(val)
                    break

        return data

    # Method - collision recovery
    def is_collision(self, ang):

        # Small step count

        count = 6
        for _ in range(count):

            # Optional draw options
            draw_options = DrawOptions(self.env.display)

            self.move(speed = 60, angle=-ang/5)

            if self.name == "user":
                self.env.display.fill(ALARM_COLOR)
            self.env.space.debug_draw(draw_options)

            self.env.clock.tick()

    # Method - action move - move a car depending on action

    def action_move(self, action):
        if action == 0:             # UP
            #print("W", end='\r', flush=True)
            self.move(speed=100)
            # print(self.body.angle)
        if action == 1:             #DOWN
            #print("S", end='\r', flush=True)
            self.move(speed=-80)
        if action == 2:             # LEFT
            #print("D", end='\r', flush=True)
            self.move(angle=-0.2)
        if action == 3:             # RIGHT
            #print("A", end='\r', flush=True)
            self.move(angle=0.2)


    # Method - reward function - depends on state and action
    # TODO - Change reward system to more robust if this fails
    def get_reward(self, data, action):

        for d in data:
            if d <= COLL_THRESH:
                reward = NEG_REWARD
                # Turn around in opossite angle
                if data.index(d) == 2:
                    #print("MIDDLE FLIP")
                    self.is_collision(math.pi/5)
                else:
                    #print("SIDE FLIP - ", self.sensors[data.index(d)], "INDEX-", data.index(d))
                    self.is_collision(self.sensors[data.index(d)])
                return reward

        reward = np.sum(data)-ACTION_DECREMENT[action]

        #reward = 0

        return reward

