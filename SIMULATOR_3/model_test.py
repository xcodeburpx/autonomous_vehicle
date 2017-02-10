import environment
import numpy as np
from keras.models import load_model

NUM_SENSORS = 4

def model_test(model):

    distance = 0
    world = environment.Env()

    _,state = world.screen_snap(0)

    while True:
        distance += 1

        state = state.reshape(1,NUM_SENSORS)
        qval = model.predict(state,batch_size=1)
        print(qval)
        action = np.argmax(qval)

        #print(action)
        reward, state = world.screen_snap(action)

        if reward == -700 or distance % 1000 == 0:
            print("Distance: %d"%distance)


if __name__ == '__main__':

    model = load_model('saved_models/kazimierz75000.model')
    model_test(model)