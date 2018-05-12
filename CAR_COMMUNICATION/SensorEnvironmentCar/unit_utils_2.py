import math
import numpy as np

from macro_utils_2 import *


# TODO: CHECK THIS UNIT - SENSOR ANGLES, CREATION, MOVEMENT

class Unit:

    def __init__(self, env, x, y, w, l, name="user", color=THECOLORS["yellow"]):

        # Sensors paramenters - angle
        self.front_back_sensors = [-math.pi * 7 / 8, -math.pi / 8, math.pi / 8, math.pi * 7 / 8]
        self.left_right_sensors = [-math.pi * 3 / 5, -math.pi / 2.5, math.pi / 2.5, math.pi * 3 / 5]

        # Position in the environment
        self.positions = []

        self.sensors_offset = math.pi / 2
        self.color = color

        self.create(env, x, y, w, l, name, color)

        self.action_n_1 = 0
        self.action_count = 0

    def create(self, env, x, y, w, l, name, color):

        self.name = name
        self.speed = 0
        self.mass = 1

        # Create moment of box

        self.moment = pymunk.moment_for_box(self.mass, (w, l))
        self.body = pymunk.Body(self.mass, self.moment)
        self.body.torque = 0
        self.body.force = 0, 0
        self.body.angular_velocity = 0
        self.body.position = x, y
        self.shape = pymunk.Poly.create_box(self.body, (w, l), 0)
        self.shape.color = color
        self.shape.elasticity = 1.5

        # Body angle
        self.body.angle = -math.pi / 8

        self.direction = Vec2d(1, 0).rotated(self.body.angle)
        self.body.apply_impulse_at_local_point(self.direction)
        self.env = env
        self.env.space.add(self.body, self.shape)

        self.print_data = []

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

        self.body.torque = 0
        self.body.force = 0, 0
        self.body.angular_velocity = 0

    def get_sensor_data(self):

        data = []  # Collected data - normalized state values
        unit_x, unit_y = self.body.position  # Get actual position
        draw_x, draw_y = 0, 0
        self.print_data = []
        t_fb = []
        t_lr = []

        for sangle in self.front_back_sensors:
            draw_end = DRAW_STOP_FB

            # Get actual value of angle
            sen_angle = math.fmod(-self.body.angle + sangle + self.sensors_offset, 2 * math.pi)
            dx = math.sin(sen_angle)
            dy = math.cos(sen_angle)

            # Get data every step
            step = 1

            # main loop
            for i in range(DRAW_START_FB, draw_end, step):

                # Get point of measurement
                x = unit_x + dx * i
                y = HEIGHT - (unit_y + dy * i)

                if i == DRAW_START_FB:
                    draw_x = x
                    draw_y = y

                # If it is out of bound
                if x >= WIDTH or y >= HEIGHT or x < 0 or y < 0:
                    # Option to draw
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (
                        int(draw_x), int(draw_y)), (int(x), int(y)), 1)

                    # Calculate ceil value of point - normalized
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))
                    val = val / DRAW_LENGTH_FB * 1.0 - 1

                    # Optional threshold
                    if draw_end != DRAW_STOP_FB:
                        val += 0.5
                    t_fb.append(val)
                    self.print_data.append("OUT OF RANGE")
                    break

                # End of the sensor
                if i == draw_end - step:
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (int(draw_x), int(draw_y)), (int(x), int(y)), 1)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))
                    val = val / DRAW_LENGTH_FB * 1.0 - 1
                    if draw_end != DRAW_STOP_FB:
                        val += .5
                    t_fb.append(val)
                    self.print_data.append("END POINT")
                    break

                # Get color at this point
                color = self.env.display.get_at((int(x), int(y)))

                # If color is different than scene or sensor color
                if color == BG_COLOR or color == SENSORS_COLOR or color == ALARM_COLOR or color == self.color:
                    continue

                else:
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (
                        int(draw_x), int(draw_y)), (int(x), int(y)), 10)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))
                    val = val / DRAW_LENGTH_FB * 1.0 - 1
                    if draw_end != DRAW_STOP_FB:
                        val += .5
                    t_fb.append(val)
                    self.print_data.append("THE COLOR:" + str(color))
                    break

        for sangle in self.left_right_sensors:
            draw_end = DRAW_STOP_LR
            # Get actual value of angle
            sen_angle = math.fmod(-self.body.angle + sangle + self.sensors_offset, 2 * math.pi)
            dx = math.sin(sen_angle)
            dy = math.cos(sen_angle)

            # Get data every step
            step = 1

            # main loop
            for i in range(DRAW_START_LR, draw_end, step):

                # Get point of measurement
                x = unit_x + dx * i
                y = HEIGHT - (unit_y + dy * i)

                if i == DRAW_START_LR:
                    draw_x = x
                    draw_y = y

                # If it is out of bound
                if x >= WIDTH or y >= HEIGHT or x < 0 or y < 0:
                    # Option to draw
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (
                        int(draw_x), int(draw_y)), (int(x), int(y)), 1)

                    # Calculate ceil value of point - normalized
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))
                    val = val / DRAW_LENGTH_LR * 1.0 - 1

                    # Optional threshold
                    if draw_end != DRAW_STOP_LR:
                        val += 0.5
                    t_lr.append(val)
                    self.print_data.append("OUT OF RANGE")
                    break

                # End of the sensor
                if i == draw_end - step:
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (int(draw_x), int(draw_y)), (int(x), int(y)), 1)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))
                    val = val / DRAW_LENGTH_LR * 1.0 - 1
                    if draw_end != DRAW_STOP_LR:
                        val += .5
                    t_lr.append(val)
                    self.print_data.append("END POINT")
                    break

                # Get color at this point
                color = self.env.display.get_at((int(x), int(y)))

                # If color is different than scene or sensor color
                if color == BG_COLOR or color == SENSORS_COLOR or color == ALARM_COLOR or color == self.color:
                    continue

                else:
                    pygame.draw.line(self.env.display, SENSORS_COLOR, (
                        int(draw_x), int(draw_y)), (int(x), int(y)), 10)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))
                    val = val / DRAW_LENGTH_LR * 1.0 - 1
                    if draw_end != DRAW_STOP_LR:
                        val += .5
                    t_lr.append(val)
                    self.print_data.append("THE COLOR:" + str(color))
                    break
        data = [t_fb[0], t_lr[0], t_lr[1], t_fb[1], t_fb[2], t_lr[2], t_lr[3], t_fb[3]]
        return data

    # Method - collision recovery
    def is_collision(self, ang):

        # Small step count
        count = 6
        for _ in range(count):

            # Optional draw options
            draw_options = DrawOptions(self.env.display)

            self.move(speed=60, angle=-ang / 2.5)
            self.get_sensor_data()

            if self.name == "user":
                self.env.display.fill(ALARM_COLOR)
            self.env.space.debug_draw(draw_options)

            self.env.clock.tick()

    # Method - action move - move a car depending on action

    def action_move(self, action):
        if action == 0:  # UP
            # print("W", end='\r', flush=True)
            self.move(speed=100)
        if action == 1:  # DOWN
            # print("S", end='\r', flush=True)
            self.move(speed=-80)
        if action == 2:  # LEFT
            # print("D", end='\r', flush=True)
            self.move(angle=-0.2)
        if action == 3:  # RIGHT
            # print("A", end='\r', flush=True)
            self.move(angle=0.2)

    # Method - reward function - depends on state and action
    # TODO - Change reward system to more robust if this fails
    def get_reward(self, data, action):

        for d in data:
            if d <= COLL_THRESH:
                #print("COLLISION:", data.index(d), "Reason:", self.print_data[data.index(d)])
                reward = NEG_REWARD
                # Turn around in opossite angle
                if data.index(d) <= 1 or data.index(d) >= 6:
                    self.is_collision(0)
                else:
                    self.is_collision(math.pi/2)
                return reward

        # if self.action_n_1 == action:
        #     self.action_count += 1
        #     reward = BASE_REWARD - self.action_count * ACTION_DECREMENT[action]
        # else:
        #     self.action_count = 0
        #     reward = BASE_REWARD - ACTION_DECREMENT[action]

        # reward = ACTION_DECREMENT[action]
        reward = sum(data) - ACTION_DECREMENT[action]

        return reward
