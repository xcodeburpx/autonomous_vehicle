import math
import time

WIDTH = 800
HEIGHT = 1000
FPS = 60.0

UNIT_X = int(WIDTH/4)
UNIT_Y = int(HEIGHT/1.5)
UNIT_R = 20

player_angle = 2*math.pi/4

number_of_sensors = 4

def get_sensor_data():
    sensor_angle = 2 * math.pi / number_of_sensors
    #sensor_angle += math.pi/8
    data = []
    colors = []
    for n in range(number_of_sensors):
        sangle = math.fmod(player_angle + n * sensor_angle, 2 * math.pi)
        dx = math.sin(sangle)
        dy = math.cos(sangle)
        print("Right now used sensor: " + str(n+1))
        for i in range(10):
            x = int(UNIT_X + dx * i)
            y = int(UNIT_Y + dy * i)
            print(x,y)
            time.sleep(1)


get_sensor_data()

"""screen.get_at(rotated_p)"""