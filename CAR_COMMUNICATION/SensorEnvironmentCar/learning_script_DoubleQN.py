from macro_utils import *
from unit_utils import Unit
from map_environment import Environment
from tensor_networks.Double_Deep_Q_Network import DoubleQNetwork
from tensor_networks.Dueling_Deep_Q_Network import DuelingDQN

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO: COMMENT THIS CODE!

# If you want to test or train network
TESTING = True

# Network argument values
if TESTING:
    e_gred = None
    eps = 1.0
else:
    e_gred = 1. / MAX_STEPS
    eps = 0.95

# Create environment
env = Environment()

# Create Tensorflow session
sess = tf.Session()

# Create Our Network
with tf.variable_scope("Double_DQN"):
    double_DQN = DoubleQNetwork(n_actions=ACTION_SPACE, state_shape=OBSERVATION_SPACE, memory_size=MEMORY_SIZE,
                                e_greedy_increment=e_gred, epsilon=eps, double_q=True, sess=sess, output_graph=False)

# with tf.variable_scope("Dueling_DQN"):
#     dueling_DQN = DuelingDQN(n_actions=ACTION_SPACE, state_shape=OBSERVATION_SPACE, memory_size=MEMORY_SIZE,
#                              e_greedy_increment=e_gred, epsilon=eps, double_q=True, dueling=True, sess=sess,
#                              output_graph=False)

sess.run(tf.global_variables_initializer())

# Save models to the path
path = os.path.abspath("models")

if not os.path.exists(path):
    os.mkdir(path)

saver = tf.train.Saver(max_to_keep=0)


# Just to check if everything works fine
# actions = []
# done_arr = []


def train(RL):
    # Steps and initial state
    total_steps = 0
    state, _, _, _ = env.step(0)

    while True:

        # Render - update positions
        env.render()

        # Get action value
        action = RL.choose_action(state)

        # Get environment response
        state_, reward, done, position = env.step(action)

        # Save observation  - (current state, action, reward, future state)
        RL.store_transition(state, action, reward, state_)

        if total_steps > MEMORY_SIZE:  # Learning
            RL.learn()

        # End of learning
        if total_steps - MEMORY_SIZE > MAX_STEPS:
            model = "DoubleDQN-" + str(total_steps-1) + ".ckpt"
            saver.save(sess, path + "/" + model)
            print("{} has been saved".format(model))
            break

        # Print probe
        if total_steps % 50 == 0:
            print("Step: {}, Act: {}, Rrd: {}, Done: {} Eps:{}, Pos: {}".format(total_steps,
                                                                                ACTION_NAMES[action], reward, done,
                                                                                RL.epsilon, position))

        # Save model every 25000 steps
        if total_steps % 2500 == 0 and total_steps != 0:
            model = "DoubleDQN-" + str(total_steps) + ".ckpt"
            saver.save(sess, path + "/" + model)
            print("{} has been saved".format(model))

        # Switch state value to future value and increment the step
        state = state_
        total_steps += 1
        # actions.append(action)
        # done_arr.append(done)

    return RL.q


def check(RL):
    # First check which model is the latest
    # Load model
    # #180000 -> UP
    # 190000
    # 232500
    # 237500
    # 172500!!!!
    model = path + "/DoubleDQN-172500.ckpt"

    # Load the model
    saver.restore(sess, model)

    # Initial state
    state, _, _, _ = env.step(0)

    # Test the network
    for _ in range(10000):
        env.render()

        action = RL.choose_action(state)

        state_, reward, done, position = env.step(action)

        print("Step: {}, Act: {}, Rrd: {}, Done: {} Eps:{}, Pos: {}".format(_,
                                                                            ACTION_NAMES[action], reward, done,
                                                                            RL.epsilon, position))
        state = state_

        time.sleep(0.03)


# Main function
if __name__ == '__main__':

    # Depends if you want to train or to test the network

    if TESTING:
        print("TESTING Double DQN")
        check(double_DQN)
    else:
        print("DOUBLE DEEP Q LEARNING")
        q_double = train(double_DQN)

        # Plot Q values - running values(they are the modified values of q_eval)
        plt.plot(np.array(q_double), c='b', label='double')
        plt.legend(loc='best')
        plt.ylabel('Q eval')
        plt.xlabel('Training step')
        plt.grid()
        plt.show()

        # plt.plot(np.array(actions), c='b', label='double')
        # plt.legend(loc='best')
        # plt.ylabel('Actions')
        # plt.xlabel('Training step')
        # plt.grid()
        # plt.show()
        #
        # plt.plot(np.array(done_arr), c='b', label='double')
        # plt.legend(loc='best')
        # plt.ylabel('Collisions')
        # plt.xlabel('Training step')
        # plt.grid()
        # plt.show()
