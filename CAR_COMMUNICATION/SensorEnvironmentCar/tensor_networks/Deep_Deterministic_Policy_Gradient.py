import numpy as np
import tensorflow as tf


# Deep Deterministic Policy Gradient
class DDPG:

    def __init__(self,
                 n_actions,
                 state_shape,
                 memory_size=3000,
                 learning_rate=0.01,
                 tau=0.01,
                 gamma=0.9,
                 batch_size=64,
                 sess=None):

        # Initialization
        self.n_actions = n_actions
        self.n_features = state_shape
        self.lr = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = self.memory = np.zeros((self.memory_size, state_shape * 2 + self.n_actions + 1))
        self.memory_counter = 0

        self.S = tf.placeholder(tf.float32, [None, state_shape], 'S')
        self.S_ = tf.placeholder(tf.float32, [None, state_shape], "S_")
        self.R = tf.placeholder(tf.float32, [None, 1], 'R')

        self.A = self._build_actor(self.S)
        Q = self._build_critic(self.S, self.A)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor")
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Critic")
        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        a_ = self._build_actor(self.S_, reuse=True, custom_getter=ema_getter)
        q_ = self._build_critic(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = tf.reduce_mean(Q)
        self.Atrain = tf.train.AdamOptimizer(self.lr).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):
            q_target = self.R + self.gamma * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=Q)
            self.Ctrain = tf.train.AdamOptimizer(self.lr * 2).minimize(td_error, var_list=c_params)

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

    def choose_action(self, s):
        return self.sess.run(self.A, {self.S: s[np.newaxis, :]})[0]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition

        self.memory_counter += 1

    def learn(self):
        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory_size[indices, :]
        bs = bt[:, :self.n_features]
        ba = bt[:, self.n_features: self.n_features + self.n_actions]
        br = bt[:, -self.n_features - 1:-self.n_features]
        bs_ = bt[:, -self.n_features:]

        self.sess.run(self.Atrain, {self.S: bs})
        self.sess.run(self.Ctrain, {self.S: bs, self.A: ba, self.R: br, self.S_: bs_})

    def _build_actor(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope("Actor", reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 50, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 30, activation=tf.nn.relu6, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.n_actions, activation=tf.nn.tanh, name='A', trainable=trainable)
            return a

    def _build_critic(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope("Critic", reuse=reuse, custom_getter=custom_getter):
            n_l1 = 50
            w1_s = tf.get_variable('w1_s', [self.n_features, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.n_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)


