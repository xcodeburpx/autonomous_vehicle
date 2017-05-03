import environment
import numpy as np
from keras.models import load_model

NUM_SENSORS = 7

def model_test(model):

    distance = 0
    world = environment.Env()

    _,state,_,enemy_state = world.screen_snap([0,0])

    while True:
        distance += 1

        state = state.reshape(1,NUM_SENSORS)
        enemy_state = enemy_state.reshape(1,NUM_SENSORS)
        #print(state)
        qval = model.predict(state,batch_size=1)
        enemy_qval = model.predict(enemy_state,batch_size=1)
        # print(qval)
        action = np.argmax(qval)
        enemy_action = np.argmax(enemy_qval)

        #print(action)
        reward, state, _, enemy_state = world.screen_snap([action,enemy_action])

        if reward == -100 or distance % 1000 == 0:
            print("Distance: %d"%distance)


if __name__ == '__main__':

    model = load_model('saved_models/kazimierz150000.model')
    model_test(model)