import numpy as np
import math
import sys
import pygame
from pygame.locals import *

pygame.init()

from macro_utils_2 import *
from unit_utils_2 import Unit

class Environment:

    def __init__(self):

        # Create the chaos - plain world
        self.display = pygame.display.set_mode((WIDTH, HEIGHT), FLAGS)
        self.display.set_alpha(None)
        self.clock = pygame.time.Clock()

        # Create value space
        self.space = pymunk.Space()
        self.space.gravity = ((0.0, 0.0))  # No gravity - at any direction
        self.drawing_on = not DRAWING

        # Green borders
        self.borders = []
        self.borders.append(pymunk.Segment(
            self.space.static_body, (0, 0), (0, HEIGHT), 5))
        self.borders.append(pymunk.Segment(
            self.space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 5))
        self.borders.append(pymunk.Segment(
            self.space.static_body, (WIDTH - 1, HEIGHT), (WIDTH - 1, 1), 5))
        self.borders.append(pymunk.Segment(
            self.space.static_body, (0, 1), (WIDTH, 1), 5))

        # Border attributes
        for border in self.borders:
            border.friction = 1.05
            border.filter = pymunk.ShapeFilter(group=1)
            border.collision_type = 2
            border.color = THECOLORS['green']
        self.space.add(self.borders)

        # Movable circular obstacles
        self.obs_count = 0
        self.obstacles = []
        self.obstacles.append(self.create_circle(500, 550, 20, 0))
        self.obstacles.append(self.create_circle(300, 500, 50, 1))
        self.obstacles.append(self.create_circle(550, 300, 30, 0))
        self.obstacles.append(self.create_circle(550, 150, 40, 1))
        self.obstacles.append(self.create_circle(750, 250, 60, 0))
        self.obstacles.append(self.create_circle(400, 700, 60, 1))
        self.obstacles.append(self.create_circle(350, 150, 40, 0))
        self.obstacles.append(self.create_circle(650, 500, 50, 1))
        self.obstacles.append(self.create_circle(550, 500, 20, 1))
        self.obstacles.append(self.create_circle(730, 780, 40, 1))



        # Create user unit
        self.unit = Unit(self, UNIT_X, UNIT_Y, UNIT_LENGTH, UNIT_WITDH, color=THECOLORS['blue'])
        self.enemy = Unit(self, UNIT_X+20, UNIT_Y+15, UNIT_LENGTH, UNIT_WITDH, color=THECOLORS['brown'])

    # Method - create circular obstacle
    def create_circle(self, x, y, r, kind):

        # Two kind of circles - dynamic or static

        if kind == 0:
            circle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            circle_body.name = 'static'
        else:
            circle_body = pymunk.Body(10000, 1)
            circle_body.name = 'dynamic'

        circle_body.position = x, y
        circle_shape = pymunk.Circle(circle_body, r)
        circle_shape.elasticity = 1.0
        circle_shape.color = THECOLORS['yellow']
        circle_body.angle = 0
        self.space.add(circle_body, circle_shape)
        return circle_body

    # Method - move dynamic object randomly
    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            if obstacle.name == "dynamic":
                speed = 4
                direction = Vec2d(1, 0).rotated(np.random.randint(-8, 8) * math.pi / 4)
                obstacle.velocity = speed * direction


    # Main function - get state, reward, info, done

    def step(self, car_action, enemy_action):
        for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_o:
                    self.drawing_on = not self.drawing_on

        # Move unit
        self.unit.action_move(car_action)
        self.enemy.action_move(enemy_action)

        # Start moving obstacles
        if self.obs_count == 500:
            self.move_obstacles()
            self.obs_count = 0
        else:
            self.obs_count += 1

        # Distance measurements
        self.unit.positions.append(self.unit.body.position)
        if len(self.unit.positions) > 10:
            self.unit.positions.pop(0)

        # # TODO: Connect this with reward
        # if len(self.unit.positions) == 10:
        #     base_position = self.unit.positions[0]
        #     positions_to_check = np.array(self.unit.positions[1:])
        #     positions_to_check[:, 0] -= base_position[0]
        #     positions_to_check[:, 1] -= base_position[1]
        #     distances = np.sqrt(np.sum(positions_to_check ** 2, 1))
        #     car_max_distance = np.max(distances)

        # State and reward
        car_data = self.unit.get_sensor_data()
        car_state = np.array(car_data)
        car_reward = self.unit.get_reward(car_data, car_action)

        # If there is a collision
        if car_reward == NEG_REWARD:
            car_done = True
        else:
            car_done = False

        # State and reward
        enemy_data = self.enemy.get_sensor_data()
        enemy_state = np.array(enemy_data)
        enemy_reward = self.enemy.get_reward(enemy_data, enemy_action)

        # If there is a collision
        if car_reward == NEG_REWARD:
            enemy_done = True
        else:
            enemy_done = False

        return car_state, car_reward, car_done, enemy_state, enemy_reward, enemy_done

    # Drawing option - increase computation speed by separating drawing from getting state and reward
    def render(self):

        draw_options = DrawOptions(self.display)

        self.display.fill(BG_COLOR)
        self.space.debug_draw(draw_options)

        self.unit.get_sensor_data()
        self.enemy.get_sensor_data()

        self.space.step(1. / 10)
        if not self.drawing_on:
            pygame.display.flip()
        self.clock.tick()


# Main test
# Unfortunately we need to use render - step doesn't change positions of objects on the map
# Will fix next time
if __name__ == '__main__':
    env = Environment()
    action = 3
    car_zero, _, _, enemy_zero,_,_= env.step(action, action)

    for i in range(100000):

        # action = np.random.randint(0,4)
        action = 0
        car_state, car_reward, car_done, enemyy_state, enemy_reward, enemy_done = env.step(action,action)
        env.render()
        # print(state, reward, position)
        time.sleep(0.06)
