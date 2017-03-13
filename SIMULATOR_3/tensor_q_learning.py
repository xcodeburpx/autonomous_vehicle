from environment import Env
import numpy as np
from neural_nets.olaf import DDD_Q_Network
from neural_nets.history_loss import LossHistory
import timeit
import random
from keras import backend as K
import tensorflow as tf


class experience_buffer():
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:int(total_vars/2)]):
        op_holder.append(tfVars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tfVars[idx+int(total_vars/2)].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

y = .99
num_episodes = 2000000
tau = 0.001
batch_size = 32
startE = 1
endE = 0.001
annealing_steps = 20000
pre_train_steps = 20000
eps_delay = 10000

BUFFER = 10000

tf.reset_default_graph()

q_net = DDD_Q_Network(4,4)
target_net = DDD_Q_Network(4,4)

init = tf.global_variables_initializer()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)
myBuffer = experience_buffer()

rList = []
rMeans = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    K.set_session(sess)
    e = startE
    stepDrop = (startE - endE)/annealing_steps
    total_steps = 0
    env = Env()

    _, state = env.screen_snap(0)

    # Used to create a car distance plot
    max_car_distance = 0
    car_distance = 0
    data_collect = []
    counter = 0

    # Replay memory and loss log lists
    replay_mem = []
    loss_log = []

    start_time = timeit.default_timer()

    for i in range(num_episodes):

        rAll = 0

        car_distance += 1
        counter += 1
        total_steps += 1

        #Boltzmann policy

        Q_d, allQ = sess.run([q_net.Q_dist, q_net.Q_out],
                             feed_dict={q_net.inputs: [state], q_net.temp: e, q_net.keep_per: 1.0})
        action = np.random.choice(Q_d[0], p=Q_d[0])
        action = np.argmax(Q_d[0] == action)

        # Reward and new state
        reward, new_state = env.screen_snap(action)
        # print(new_state)
        if reward == -700:
            d = True
        else:
            d = False

        # Storing the (S,A,R,S') tuple in replay memory
        myBuffer.add(np.reshape(np.array([state,action,reward,new_state,d]),[1,5]))

        if e > endE and total_steps > pre_train_steps:
            e -= stepDrop


        # If the counter has delayed -> start learning the net
        if total_steps > pre_train_steps and total_steps % 5 == 0:

            # Random sample
            trainBatch = myBuffer.sample(batch_size)

            Q1 = sess.run(q_net.predict, feed_dict={q_net.inputs: np.vstack(trainBatch[:, 3]), q_net.keep_per: 1.0})
            Q2 = sess.run(target_net.Q_out,
                          feed_dict={target_net.inputs: np.vstack(trainBatch[:, 3]), target_net.keep_per: 1.0})
            end_multiplier = -(trainBatch[:, 4] - 1)
            doubleQ = Q2[range(batch_size), Q1]
            targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
            _ = sess.run(q_net.updateModel,
                         feed_dict={q_net.inputs: np.vstack(trainBatch[:, 0]), q_net.nextQ: targetQ, q_net.keep_per: 1.0,
                                    q_net.actions: trainBatch[:, 1]})
            updateTarget(targetOps, sess)

         # Info window
        if reward == -700:
                data_collect.append([counter, car_distance])

                # Update max.
                if car_distance > max_car_distance:
                    max_car_distance = car_distance

                # Time it.
                tot_time = timeit.default_timer() - start_time
                fps = car_distance / tot_time

                # Output some stuff so we can watch.
                print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                      (max_car_distance, counter, e, car_distance, fps))

                # Reset.
                car_distance = 0
                start_time = timeit.default_timer()

        if i % 25000 == 0 and i != 0:
            save_path = saver.save(sess, "./tens_save/olaf_{}.ckpt".format(i))
            print("Model saved in file: %s" % save_path)
        # S <- S'
        state = new_state
        rAll += reward
        rList.append(rAll)
        if i % 100 == 0 and i != 0:
            r_mean = np.mean(rList[-100:])
            print("Mean Reward: " + str(r_mean) + " Total Steps: " + str(total_steps) + " t: " + str(e))

print("\n\nPercent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")