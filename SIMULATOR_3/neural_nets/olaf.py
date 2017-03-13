import tensorflow as tf
import tensorflow.contrib.slim as slim

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, GaussianNoise, GaussianDropout

data_dim = 4
timesteps = 1
nb_classes = 4

class DDD_Q_Network:
    def __init__(self, data_dim, nb_classes):

        self.inputs = tf.placeholder(shape=[None,data_dim], dtype=tf.float32)
        self.temp = tf.placeholder(shape=None, dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=None, dtype=tf.float32)

        with tf.device("/cpu:0"):
            hidden = Dense(64, activation=tf.nn.tanh, init='he_normal', bias=False)(self.inputs)
            hidden = slim.dropout(hidden, self.keep_per)

            hidden = Dense(32, activation=tf.nn.tanh, init='he_normal', bias=False)(hidden)
            hidden = slim.dropout(hidden, self.keep_per)

            self.Q_out = Dense(nb_classes, activation=None, bias=False)(hidden)

            self.predict = tf.argmax(self.Q_out, 1)
            self.Q_dist = tf.nn.softmax(self.Q_out / self.temp)

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)
        with tf.device('/gpu:0'):
            self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), reduction_indices=1)

            self.nextQ = tf.placeholder(shape=[None], dtype=tf.float32)
            loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))
        with tf.device("/cpu:0"):
            trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
            self.updateModel = trainer.minimize(loss)
