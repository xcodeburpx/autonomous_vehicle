# Import libraries

import numpy as np
import tensorflow as tf

SEED = 1

# Set a seed number
np.random.seed(SEED)
tf.set_random_seed(SEED)

# Double Dueling Q Network class
# TODO: SAVE MODEL AND LATER RELOAD IT
# TODO: DUELING DOUBLE DEEP Q NETWORK TO CREATE!
# TODO: CHECK THE CODE
# TODO: CHECK THE MEMORY RELAY INFO
# TODO: FIX IT FAST!


class DoubleQNetwork:
    def __init__(
            self,
            n_actions,                 # Number of action in this world
            state_shape,                # Number of params of state
            learning_rate=0.005,       # Learning rate
            reward_decay=0.9,          # Gamma - reward decay for action taken
            epsilon=0.9,              # Epsilon - probability of choosing action instead random action
            replace_target_iter=200,   # How many times values of nets must switch
            memory_size=3000,          # Sample memory delay size
            batch_size=64,             # Number of states to analise
            e_greedy_increment=None,   # If you want to change epsilon in runtime
            output_graph=False,        # If you want to have output_graph
            double_q=True,             # Double DQN construction
            sess=None,                 # If you have a session
    ):

        # Initialization
        self.n_actions = n_actions
        self.n_features = state_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q  # decide to use double q or not

        # Just to trigger replacement
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, state_shape * 2 + 2)) # check what is this - answer(size of state, size of action, size of reward, size of next state)
        self._build_net()                                              # build network
        t_params = tf.get_collection('target_net_params')              # Take parameters of target network
        e_params = tf.get_collection('eval_net_params')                # Take parameters of evaluation network
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # Replace values

        # Session initialization
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        # Tensorboard graph - if you want to look
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        # Loss history
        #self.cost_his = []

    # Build nets
    def _build_net(self):
        # 3-layer fully-connected neural network
        def build_layers(s, c_names, n_l1, n_l2, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out

        # Proper Double Deep Q Network

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input - current state
        # The answer from target network
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 512, 512, \
                tf.contrib.layers.xavier_initializer(seed=SEED), tf.constant_initializer(0.3)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, n_l2, w_initializer, b_initializer)

        # Calculate loss of prediction
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input - future state
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, n_l2, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'): # Memory initalization
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_)) # current state, [action, reward], future state
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :] # Extra dimension - to calm Tensorflow down
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        if self.epsilon_increment is not None:
            self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value) # Modification of reward - running reward
            self.q.append(self.running_q)

        # E-greedy policy - choose random action or network response
        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # Target - evaluation value swap - robustness
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\nTARGET NET PARAMETERS RELPACED!\n')

        if self.memory_counter > self.memory_size: # Taking or memory decay
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # print("SIZE OF BATCH_MEMORY: {}".format(batch_memory.shape))

        # Get current Q value and future Q value
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # current observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
        q_target = q_eval.copy() # Copy q_table - changes in this table

        # Get proper indices from target network Q values
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # Get actions and rewards
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # Changes - double DQN or normal DQN
        if self.double_q:
            max_act4next = np.argmax(q_eval4next,
                                     axis=1)  # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        # IMPORTANT!
        # Update values
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        #self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1