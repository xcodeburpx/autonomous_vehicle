from macro_utils_2 import *
from map_environment_2 import Environment
from tensor_networks.Dueling_Deep_Q_Network import DuelingDQN

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# If you want to test or train network
TESTING = True

path = os.path.abspath("models_mldrive")

model = path + "/DuelingDQN-225000.ckpt"

# Network argument values
if TESTING:
    e_gred = None
    eps = 1.0
else:
    eps = 0.95
    e_gred = eps / MAX_STEPS

# Create environment
env = Environment()

# Create Tensorflow session
sess = tf.Session()

# Create our network
with tf.variable_scope("Dueling_DQN"):
    dueling_DQN = DuelingDQN(n_actions=ACTION_SPACE, state_shape=OBSERVATION_SPACE, memory_size=MEMORY_SIZE,
                             e_greedy_increment=e_gred, epsilon=eps, double_q=True, dueling=True, sess=sess,
                             output_graph=False)

sess.run(tf.global_variables_initializer())

if not os.path.exists(path):
    os.mkdir(path)

saver = tf.train.Saver(max_to_keep=0)


def train(RL):
    # Steps and initial state
    total_steps = 0
    car_state, _, _, enemy_state, _, _ = env.step(0, 0)

    while True:

        # Render - update positions
        env.render()

        # Get action value
        car_action = RL.choose_action(car_state)
        enemy_action = RL.choose_action(enemy_state)

        # Get environment response
        car_state_, car_reward, car_done, enemy_state_, enemy_reward, enemy_done = env.step(car_action, enemy_action)

        # Save observation  - (current state, action, reward, future state)
        RL.store_transition(car_state, car_action, car_reward, car_state_)
        RL.store_transition(enemy_state, enemy_action, enemy_reward, enemy_state_)

        if total_steps > MEMORY_SIZE:  # Learning
            RL.learn()

        # End of learning
        if total_steps - MEMORY_SIZE > MAX_STEPS:
            break

        # Print probe
        if total_steps % 50 == 0:
            print("Step: {}, car_Act: {}, car_Rrd: {}, car_Done: {} enem_Act: {}, enem_Rrd: {}, enem_Done: {}".format(
                total_steps,
                ACTION_NAMES[car_action], car_reward, car_done,
                ACTION_NAMES[
                    enemy_action],
                enemy_reward,
                enemy_done))

        # Save model every 25000 steps
        if total_steps % SAVE_THRESH == 0 and total_steps != 0:
            model = "DuelingDQN-" + str(total_steps) + ".ckpt"
            saver.save(sess, path + "/" + model)
            print("{} has been saved".format(model))

        # Switch state value to future value and increment the step
        car_state = car_state_
        enemy_state = enemy_state_
        total_steps += 1
        # actions.append(action)
        # done_arr.append(done)

    return RL.q


def check(RL):
    # First check which model is the latest
    # Load model

    # Load the model
    saver.restore(sess, model)

    # Initial state
    car_state, _, _, enemy_state, _, _ = env.step(0, 0)

    # Test the network
    for _ in range(10000):
        env.render()

        # Get action value
        car_action = RL.choose_action(car_state)
        enemy_action = RL.choose_action(enemy_state)

        # Get environment response
        car_state_, car_reward, car_done, enemy_state_, enemy_reward, enemy_done = env.step(car_action, enemy_action)

        print("car_Act: {}, car_Rrd: {}, car_Done: {} enem_Act: {}, enem_Rrd: {}, enem_Done: {}".format(
            ACTION_NAMES[car_action], car_reward, car_done,
            ACTION_NAMES[
                enemy_action],
            enemy_reward,
            enemy_done))
        # Switch state value to future value and increment the step
        car_state = car_state_
        enemy_state = enemy_state_

        time.sleep(0.03)


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
