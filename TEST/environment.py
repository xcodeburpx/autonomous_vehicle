
#Import libraries and modules
import sys
import math
import numpy as np
import enum

import pygame
from pygame.locals import *
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions

#Macro initial values
WIDTH = 900
HEIGHT = 900
FPS = 60.0
BG_COLOR = THECOLORS['black']
SENSORS_COLOR = THECOLORS['white']
ALARM_COLOR = THECOLORS['black']

CAR_X = WIDTH/4
CAR_Y = HEIGHT/1.2
CAR_R = 20

DRAW_START = CAR_R+10
DRAW_STOP = DRAW_START+100
COLL_THRESH = -1.0

# BASE_REWARD = -30

flags = pygame.DOUBLEBUF
#Start pygame
pygame.init()
display = pygame.display.set_mode((WIDTH, HEIGHT), flags)
display.set_alpha(None)
clock = pygame.time.Clock()

#For testing -> enum class for action
class Action(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 3
    RIGHT = 2

class Unit:
    def __init__(self,space,x,y,r,name='enemy',color=THECOLORS['blueviolet']):
        #Sensor hyperparameters
        self.sensors=[-math.pi/6,0,math.pi/6,math.pi*.85,math.pi*1.15,math.pi*1.5,math.pi*.5]
        self.short_sensors = [math.pi*1.5,math.pi*.5]
        self.sensors_offset=math.pi/2
        self.position_check=[]

        self.name=name
        self.speed = 0
        self.mass = 1
        self.moment = pymunk.moment_for_circle(self.mass, 0, r, (0,0))
        self.body = pymunk.Body(self.mass, self.moment)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, r)
        self.shape.color = color
        self.shape.elasticity = 1.0
        self.body.angle = -math.pi/8
        self.direction = Vec2d(1,0).rotated(self.body.angle)
        self.body.apply_impulse_at_local_point(self.direction)
        self.space = space
        self.space.add(self.body, self.shape)

    def move(self,speed=None,angle=None):
        if not speed:
            if not angle:
                self.speed = 100
                self.body.velocity = self.direction * self.speed
            else:
                self.speed = 50
                self.body.angle -= angle
                self.direction = Vec2d(1, 0).rotated(self.body.angle)
                self.body.velocity = self.direction * self.speed
        else:
            if not angle:
                self.speed = speed
                self.body.velocity = self.direction * self.speed
            else:
                self.speed = speed
                self.body.angle -= angle
                self.direction = Vec2d(1, 0).rotated(self.body.angle)
                self.body.velocity = self.direction * self.speed
    def get_sensor_data(self):

        # One of main function -> detection system
        # Credits to Mateusz Jakubiec
        data = []
        player_x, player_y = self.body.position
        draw_x, draw_y = 0,0


        for sangle in self.sensors:

            if sangle in self.short_sensors:
                draw_end=DRAW_STOP-60
            else:
                draw_end=DRAW_STOP

            sangle = math.fmod(-self.body.angle + sangle + self.sensors_offset, 2 * math.pi)
            dx = math.sin(sangle)
            dy = math.cos(sangle)

            step=4

            for i in range(DRAW_START,draw_end,step):
                x = player_x + dx * i
                y = HEIGHT - (player_y + dy * i)

                if i == DRAW_START:
                    draw_x = x
                    draw_y = y

                if x >= WIDTH or y >= HEIGHT or x < 0 or y < 0:
                    pygame.draw.line(display, SENSORS_COLOR, (int(draw_x), int(draw_y)), (int(x), int(y)), 1)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))/50.0-1
                    if draw_end!=DRAW_STOP:
                        val+=.5
                    data.append(val)
                    break

                color = display.get_at((int(x), int(y)))

                if (i == draw_end-step):
                    pygame.draw.line(display, SENSORS_COLOR, (int(draw_x), int(draw_y)), (int(x), int(y)), 1)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))/50.0-1
                    if draw_end!=DRAW_STOP:
                        val+=.5
                    data.append(val)
                    break

                if color == BG_COLOR or color == SENSORS_COLOR or color == ALARM_COLOR:
                    continue

                else:
                    pygame.draw.line(display, SENSORS_COLOR, (int(draw_x), int(draw_y)), (int(x), int(y)), 10)
                    val = math.ceil(math.sqrt((draw_x - x) ** 2 + (draw_y - y) ** 2))/50.0-1
                    if draw_end!=DRAW_STOP:
                        val+=.5
                    data.append(val)
                    break
        return data

    def action_move(self, action):
        if action == 0:             # UP
            #print("W", end='\r', flush=True)
            self.move(speed=100)
        if action == 1:             #DOWN
            #print("S", end='\r', flush=True)
            self.move(speed=-80)
        if action == 2:             # RIGHT
            #print("D", end='\r', flush=True)
            self.move(angle=0.2)
        if action == 3:             # LEFT
            #print("A", end='\r', flush=True)
            self.move(angle=-0.2)

        return action
    def is_collision(self,ang):
        for _ in range(10):
                draw_options = DrawOptions(display)
                self.move(speed = 60, angle=ang)
                if self.name=="car":
                    display.fill(ALARM_COLOR)
                self.space.debug_draw(draw_options)
                # self.space.step(1./10)

                # pygame.display.flip()
                clock.tick()
    #Reward function with action penalty
    def reward_func(self, data, action,max_distance):
        if COLL_THRESH in data:
            reward = -1
            self.is_collision(self.sensors[data.index(COLL_THRESH)])
        else:
            # if min(data)<=0.2:
            #     reward=0
            # else:
            #     reward=1
            # data.sort()
            # reward = (np.sum(data[:4])+4)*10-30
            # if action == 0:
            #     reward = BASE_REWARD + int((np.min(data)+6)*5)
            # if action == 1:
            #     reward-=.2

            reward = np.sum(data)/6.0
            if action == 1:
                reward-=.03
            if max_distance<50:
                reward-=(50-max_distance)/350.

            # if action == 0:
            #     reward+=10
            # if action == 2 or action == 3:
            #     reward-=5
        return reward


# Class Environment
class Env:
    def __init__(self):

        #Start pymunk
        self.space = pymunk.Space()
        self.space.gravity = ((0.0, 0.0))
        self.drawing_on=False


        #Green borders
        self.borders = []
        self.borders.append(pymunk.Segment(
            self.space.static_body, (0,0), (0,HEIGHT), 3))
        self.borders.append(pymunk.Segment(
            self.space.static_body, (0,HEIGHT), (WIDTH,HEIGHT), 3))
        self.borders.append(pymunk.Segment(
            self.space.static_body, (WIDTH-1, HEIGHT), (WIDTH-1, 1), 3))
        self.borders.append(pymunk.Segment(
            self.space.static_body, (0,1), (WIDTH,1), 3))

        for border in self.borders:
            border.friction  = 1.05
            border.filter = pymunk.ShapeFilter(group=1)
            border.collision_type = 2
            border.color = THECOLORS['green2']
        self.space.add(self.borders)

        # TO DO -> create some obstacles
        self.obstacles_count=0
        self.obstacles = []
        self.obstacles.append(self.create_circle(500, 550, 20,0))
        self.obstacles.append(self.create_circle(300, 500, 50,1))
        self.obstacles.append(self.create_circle(550, 300, 30,0))
        self.obstacles.append(self.create_circle(550, 150, 40,1))
        self.obstacles.append(self.create_circle(200, 400, 60,0))
        self.obstacles.append(self.create_circle(150, 800, 60,1))
        self.obstacles.append(self.create_circle(250, 150, 40,0))
        self.obstacles.append(self.create_circle(650, 500, 50,1))
        self.obstacles.append(self.create_circle(250, 500, 50, 1))
        self.obstacles.append(self.create_circle(250, 200, 30, 1))
        self.obstacles.append(self.create_circle(700, 750, 40, 0))

        # create enemy and a car units
        self.enemy=Unit(self.space, WIDTH / 4, HEIGHT / 4, 20)
        self.car=Unit(self.space, CAR_X, CAR_Y, CAR_R,"car",THECOLORS['orange'])

    def create_wall(self, a, b, r):
        self.wall_shape = pymunk.Segment(self.space.static_body, a, b, r)
        self.wall_shape.elasticity = 0.8
        self.wall_shape.color  = THECOLORS['yellow']
        self.space.add(self.wall_shape)

    def create_circle(self, x, y, r, kind):
        if kind == 0:
            circle_body = pymunk.Body(body_type = pymunk.Body.STATIC)
            circle_body.name='static'
        else:
            circle_body = pymunk.Body(10000,1)
            circle_body.name='dynamic'
        circle_body.position = x, y
        circle_shape = pymunk.Circle(circle_body, r)
        circle_shape.elasticity = 1.
        circle_shape.color = THECOLORS['yellow']
        circle_body.angle = 0
        # circle_direction = Vec2d(1, 0).rotated(circle_body.angle)
        self.space.add(circle_body, circle_shape)
        return circle_body
    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            if obstacle.name=="dynamic":
                speed = 4
                direction = Vec2d(1, 0).rotated(np.random.randint(-8, 8)*math.pi/4)
                obstacle.velocity = speed * direction


    #MAIN FUNCTION
    def screen_snap(self, actions):
        draw_options = DrawOptions(display)

        for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_o:
                    self.drawing_on = not self.drawing_on
                #else:
                #   self.controller(event)   #-> only works here(pygame.event.get() function works once for iteration)

        # moving objects
        self.car.action_move(actions[0])
        self.enemy.action_move(actions[1])
        if self.obstacles_count == 500:
            self.move_obstacles()
            self.obstacles_count = 0
        else:
            self.obstacles_count+=1

        self.car.position_check.append(self.car.body.position)
        if len(self.car.position_check)>10:
            self.car.position_check.pop(0)
        self.enemy.position_check.append(self.enemy.body.position)
        if len(self.enemy.position_check) > 10:
            self.enemy.position_check.pop(0)

        
        # drawing things
        display.fill(BG_COLOR)
        self.space.debug_draw(draw_options)

        if len(self.car.position_check)==10:
            base_position = self.car.position_check[0]
            positions_to_check = np.array(self.car.position_check[1:])
            positions_to_check[:, 0] -= base_position[0]
            positions_to_check[:, 1] -= base_position[1]
            distances = np.sqrt(np.sum(positions_to_check**2, 1))
            car_max_distance=np.max(distances)
        else:
            car_max_distance=99999

        if len(self.enemy.position_check)==10:
            base_position = self.enemy.position_check[0]
            positions_to_check = np.array(self.enemy.position_check[1:])
            positions_to_check[:, 0] -= base_position[0]
            positions_to_check[:, 1] -= base_position[1]
            distances = np.sqrt(np.sum(positions_to_check**2, 1))
            enemy_max_distance=np.max(distances)
        else:
            enemy_max_distance=99999

        car_data = self.car.get_sensor_data()
        car_state = np.array(car_data)
        enemy_data = self.enemy.get_sensor_data()
        enemy_state = np.array(enemy_data)

        car_reward = self.car.reward_func(car_data,actions[0],car_max_distance)
        enemy_reward = self.enemy.reward_func(enemy_data,actions[1],enemy_max_distance)

        # print car_state,"$",car_reward
        # print enemy_state,"$",enemy_reward

        self.space.step(1./10)
        if not self.drawing_on:
            pygame.display.flip()
        clock.tick()

        return car_reward, car_state, enemy_reward, enemy_state

if __name__ == "__main__":
    env = Env()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            if event.type == KEYDOWN:
                if event.key == K_w:
                    action = np.random.randint(0,4)
                    reward, state, _, _ = env.screen_snap([np.random.randint(0,2),np.random.randint(0,4)])
