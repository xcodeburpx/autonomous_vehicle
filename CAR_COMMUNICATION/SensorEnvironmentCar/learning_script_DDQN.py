from macro_utils import *
from unit_utils import Unit
from map_environment import Environment
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

tf.reset_default_graph()
sess = tf.Session()

# Create our network
with tf.variable_scope("Dueling_DQN"):
    dueling_DQN = DuelingDQN(n_actions=ACTION_SPACE, state_shape=OBSERVATION_SPACE, memory_size=MEMORY_SIZE,
                             e_greedy_increment=e_gred, epsilon=eps, double_q=True, dueling=True, sess=sess,
                             output_graph=False)

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
            model = "DuelingDQN-" + str(total_steps) + ".ckpt"
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
            model = "DuelingDQN-" + str(total_steps) + ".ckpt"
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
    #230000
    #180000
    #130000!!
    model = path + "/DuelingDQN-130000.ckpt"

    # Load the model
    saver.restore(sess, model)

    # Initial state
    state, _, _, _ = env.step(0)

    # Test the network
    for _ in range(1000):
        env.render()

        action = RL.choose_action(state)

        state_, reward, done, position = env.step(action)

        print("Step: {}, Act: {}, Rrd: {}, Done: {} Eps:{}, Pos: {}".format(_,
                                                                            ACTION_NAMES[action], reward, done,
                                                                            RL.epsilon, position))
        state = state_

        time.sleep(0.01)


# Main function
if __name__ == '__main__':

    # Depends if you want to train or to test the network

    if TESTING:
        print("TESTING Dueling DQN")
        check(dueling_DQN)
    else:
        print("DUELING DEEP Q NETWORK")
        q_duel = train(dueling_DQN)

        # Plot Q values - running values(they are the modified values of q_eval)
        plt.plot(np.array(q_duel), c='b', label='dueling')
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
