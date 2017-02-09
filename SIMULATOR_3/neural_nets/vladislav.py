import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.utils.visualize_util import plot

data_dim = 4
timesteps = 1
nb_classes = 4

def Vladislav():

    model = Sequential()

    model.add(Convolution1D(64, 2,input_dim=data_dim, activation='tanh'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=nb_classes, init='uniform', activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def test():
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, timesteps, nb_classes))

    # generate dummy validation data
    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, timesteps, nb_classes))

    model = Vladislav()

    plot(model, "vladislav.png", show_shapes=True, show_layer_names=True)
    model.fit(x_train, y_train,
              batch_size=16, nb_epoch=5,
              validation_data=(x_val, y_val))


    print("FINISHED!!")

test()