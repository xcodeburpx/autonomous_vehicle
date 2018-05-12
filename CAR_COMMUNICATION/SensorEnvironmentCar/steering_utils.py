import numpy as np
from math import asin, pi

def steering_angle(target_position, car_position,past_position):

    divid = 2

    def get_angle(t_c_mult, t_c_vec, c_p_vec):
        target_car_norm = np.linalg.norm(t_c_vec)
        car_past_norm = np.linalg.norm(c_p_vec)
        
        if target_car_norm * car_past_norm < 1e-8:
            return 0

        sinus = t_c_mult/(target_car_norm*car_past_norm + 1e-6)

        ang = asin(sinus)
        # if ang > pi:
        #     return (pi-ang)/divid

        return 2*ang/divid

    # Get vectors
    target_car_vector = target_position - car_position
    car_past_vector = car_position - past_position
    target_past_vector = target_position - past_position

    # Get Vector Cross Product
    target_car_mult = target_car_vector[0] * car_past_vector[1] - target_car_vector[1] * car_past_vector[0]
    #target_car_mult = target_car_vector[1]*car_past_vector[0] - target_car_vector[0]*car_past_vector[1]

    angle = get_angle(target_car_mult, target_car_vector, car_past_vector)
    if np.linalg.norm(target_past_vector) > np.linalg.norm(target_car_vector):
         angle *= 1
    # else:
    #     angle *= -1

    return angle

