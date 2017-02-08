import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization
from keras.utils.visualize_util import plot

data_dim = 4
timesteps = 1
nb_classes = 4



def Kazimierz():
    
	
    model = Sequential()

    model.add(Dense(128, init='uniform', activation='tanh', input_shape=(data_dim,)))
    model.add(ActivityRegularization(l1=0.05, l2=0.15))
    model.add(Dropout(0.2))
    model.add(Dense(32, init='normal', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(ActivityRegularization(l1=0.05, l2=0.15))
    model.add(Dense(nb_classes, activation='sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    return model


def test():
    x_train = np.random.random((1000, data_dim))
    y_train = np.random.random((1000, nb_classes))

    # generate dummy validation data
    x_val = np.random.random((100, data_dim))
    y_val = np.random.random((100, nb_classes))

    model = Kazimierz()

    model.fit(x_train, y_train,
              batch_size=1, nb_epoch=5,
              validation_data=(x_val, y_val))

    xx = np.array([0,1,4,2])
    xx = xx.reshape(1,4)
    model.predict(xx, batch_size=1)

    plot(model, "kazimierz.png", show_shapes=True, show_layer_names=True)

    print("FINISHED!!")


