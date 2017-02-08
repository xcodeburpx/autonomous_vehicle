import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils.visualize_util import plot

data_dim = 4
timesteps = 1
nb_classes = 4



def Jarvis():


    model = Sequential()
    model.add(LSTM(128,return_sequences=True ,input_shape=(timesteps, data_dim)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, init='uniform',activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, init='normal', activation='sigmoid'))

    model.compile(loss = 'mse', optimizer='rmsprop', metrics=['accuracy'])

    return model

def test():
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, nb_classes))

    # generate dummy validation data
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, nb_classes))

    model = Jarvis()

    model.fit(x_train, y_train,
              batch_size=16, nb_epoch=5,
              validation_data=(x_val, y_val))

    plot(model, "jarvis.png", show_shapes=True, show_layer_names=True)

    print("FINISHED!!")


