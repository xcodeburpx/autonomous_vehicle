from unit_utils import Unit
from macro_utils import *
from map_environment import Environment

import numpy as np
from math import acos, degrees, asin
from sys import float_info
from scipy.spatial import distance

# TODO: DEBUGING KĄTÓW -> CHYBA TRACI STABILNOŚĆ
# TODO: TABLICA KARNAUGH -> SPRAWDZIĆ POPRAWNOŚĆ
# TODO: RYSOWANIE PRZEBIEGU -> ROZWIĄZANIE BARDZO BLISKO!


car_n_1 = None

target = np.array((WIDTH/2, HEIGHT))


def bool_value(x):
    if x >= 0:
        return True
    else:
        return False


# def steer_sign(tx, ty, xn, yn):
#     if (not tx and yn) or (tx and ty and xn) or (tx and not ty and not xn):
#         return -1
#     else:
#         return 1
#
#
# def get_steer_sign(dtx, dty, dxn, dyn):
#     tx = bool_value(dtx)
#     ty = bool_value(dty)
#     xn = bool_value(dxn)
#     yn = bool_value(dyn)
#
#     return steer_sign(tx, ty, xn, yn)


def get_angle(t_car_v, t_car_d, car_n_d):
    tar_car_norm = np.linalg.norm(t_car_d)
    car_n_norm = np.linalg.norm(car_n_d)

    if tar_car_norm * car_n_norm < 0.0001:
        return 0
    acsin = t_car_v/(tar_car_norm * car_n_norm)

    angl = asin(acsin)
    if angl > np.pi/2:
        return np.pi-angl

    return angl/4



def testing_steer_data(car: Unit):
    # Count the distance between two points

    car_n = np.array(car.positions)
    car_n = car_n[0]
    target_car_n = target - car_n
    # target_car_n_1 = target - car_n_1
    car_n_dist = car_n - car_n_1

    # targ_car_n_dist = np.linalg.norm(target_car_n)
    # targ_car_n_1_dist = np.linalg.norm(target_car_n_1)
    # car_dist = np.linalg.norm(car_n_dist)

    targ_car_vec = target_car_n[0]*car_n_dist[1] - target_car_n[1] * car_n_dist[0]


    # print("Target distance:",targ_car_n_dist)
    # print("Target - n-1 distance:",targ_car_n_1_dist)
    # print("Car distance:",car_dist)

    # Get steering sign
    # Get the angle

    angle = get_angle(targ_car_vec, target_car_n, car_n_dist)

    return angle


if __name__ == '__main__':
    env = Environment()
    action = 0
    zero_state, _, _, _ = env.step(action)
    car_n_1 = np.array(env.unit.positions)
    car_n_1 = car_n_1[0]
    print(car_n_1)

    for i in range(100000):
        if i % 4 == 0 and i != 0:
            angle = testing_steer_data(env.unit)
            env.unit.move(60, angle=angle)
            car_n_1 = np.array(env.unit.positions)
            car_n_1 = car_n_1[0]
        else:
            action = np.random.randint(0,1)
            state, reward, done, position = env.step(action)
        env.render()
        # print(state, reward, position)
        time.sleep(0.06)
