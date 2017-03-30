import environment
import numpy as np
from neural_nets import kazimierz
from neural_nets.history_loss import LossHistory
import sys
import timeit
import random
import csv

# Macros for q-learning
GAMMA = 0.95
BATCH_SIZE = 360
BUFFER = 50000
VAL_BUFFER = 500
NUM_INPUT = 7

"""Credits to  Matt Harvey (harvitronix)"""


# TO DO -> create class 'Q-learining' and 'TD(lambda)-learning'

def q_learning(model, name):
    model_name = name

    # Initial values for epsilon policy
    eps_delay = 1000
    epsilon = 1
    train_frames = 150000
    counter = 0

    # Used to create a car distance plot
    max_car_distance = 0
    car_distance = 0
    data_collect = []

    # Replay memory and loss log lists
    replay_mem = []
    val_replay_mem = []
    loss_log = []
    val_loss_log = []

    # Start the environment
    world = environment.Env()

    # Initial reward and state -> ignoring the reward
    _, state, _, enemy_state = world.screen_snap([0, 0])

    # Measure the speed of environment -> fps
    start_time = timeit.default_timer()

    # MAIN LOOP
    while counter < train_frames:

        car_distance += 1
        counter += 1

        # Epsilon policy
        if np.random.rand() < epsilon or counter < eps_delay:
            action = np.random.randint(0, 4)
        # action = 0
        else:
            state = state.reshape(1, NUM_INPUT)
            qval = model.predict(state, batch_size=1)
            action = np.argmax(qval)
            state = state.reshape(NUM_INPUT, )
        # print action, qval

        # enemy movse based on car's model
        if counter < eps_delay:
            enemy_action = np.random.randint(0, 4)
        else:
            enemy_state = enemy_state.reshape(1, NUM_INPUT)
            enemy_qval = model.predict(enemy_state, batch_size=1)
            enemy_action = np.argmax(enemy_qval)
            enemy_state = enemy_state.reshape(NUM_INPUT, )

        # Reward and new state
        reward, new_state, enemy_reward, enemy_new_state = world.screen_snap([action, enemy_action])
        # print(new_state)

        # Storing the (S,A,R,S') tuple in replay memory
        replay_mem.append(np.concatenate((state, [action, reward], new_state)))
        val_replay_mem.append(np.concatenate((enemy_state, [enemy_action, enemy_reward], enemy_new_state)))

        if (len(val_replay_mem)) > VAL_BUFFER:
            val_replay_mem.pop(0)
            # If the counter has delayed -> start learning the net
        if counter > eps_delay:

            # Only the fresh data we need
            if (len(replay_mem)) > BUFFER:
                replay_mem.pop(0)

                # Random sample
            minibatch = random.sample(replay_mem, BATCH_SIZE)

            # Update the reward -> take the importance of states into account
            X_train, y_train = update_func(minibatch, model, BATCH_SIZE)
            X_val, y_val = update_func(val_replay_mem, model, VAL_BUFFER)

            # Train the model on this batch.

            history = LossHistory()

            model.fit(
                X_train, y_train, epochs=1,
                batch_size=int(BATCH_SIZE / 4), verbose=0, callbacks=[history]
            )

            val_loss = model.evaluate(X_val, y_val, batch_size=VAL_BUFFER, verbose=0)
            val_loss_log.append([val_loss])
            # print("Model trained!\n")
            loss_log.append(history.losses)
        # S <- S'
        state = new_state
        enemy_state = enemy_new_state

        # Epsilon decrementation
        if epsilon > 0.1 and counter > eps_delay:
            epsilon -= (1.0 / train_frames * 1.2)

            # Info window
        # if counter % 100 == 0:
        # 	tot_time = timeit.default_timer() - start_time
        # 	fps = car_distance / tot_time
        # 	print fps

        if reward == -1:
            data_collect.append([counter, car_distance])

            # Update max.
            if car_distance > max_car_distance:
                max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time

            # Output some stuff so we can watch.
            print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                  (max_car_distance, counter, epsilon, car_distance, fps))

            # Reset.
            car_distance = 0
            start_time = timeit.default_timer()

            # Saving the model and weights
        if counter % 25000 == 0:
            model.save_weights("saved_weights/" + model_name +
                               str(counter) + ".h5")
            model.save("saved_models/" + model_name +
                       str(counter) + ".model")
            print("Saving model %s - %d" % (model_name, counter))

    log_results(model_name, data_collect, loss_log, val_loss_log)


def log_results(filename, data_collect, loss_log, val_loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)
    with open('results/val_loss_data-' + filename + '.csv', 'w') as vlf:
        wr = csv.writer(vlf)
        for val_loss_item in val_loss_log:
            wr.writerow(val_loss_item)


def update_func(minibatch, model, batch_len):
    X_train = []
    y_train = []

    minibatch = np.array(minibatch)
    # Get stored values.
    old_state = minibatch[:, :NUM_INPUT]
    new_state = minibatch[:, NUM_INPUT + 2:]
    # Get prediction on old state.
    old_qval = model.predict(old_state, batch_size=batch_len)
    # Get prediction on new state.
    newQ = model.predict(new_state, batch_size=batch_len)
    # Get our best move.
    maxQ = np.max(newQ, 1)
    X_train = np.zeros((batch_len, NUM_INPUT))
    X_train[:, :] = old_state[:, :]
    # preset y_train
    y_train = np.zeros((batch_len, 4))
    y_train[:, :] = old_qval[:, :]

    action_m = minibatch[:, NUM_INPUT]
    # action_m = map(int, action_m)
    action_m = action_m.astype(int)
    reward_m = np.zeros((batch_len))
    reward_m[:] = minibatch[:, NUM_INPUT + 1]
    reward_m[reward_m != -1] = reward_m[reward_m != -1] + (GAMMA * maxQ[reward_m != -1])
    y_train[range(batch_len), action_m] = reward_m

    return X_train, y_train


if __name__ == '__main__':
    model = kazimierz.Kazimierz()
    name = 'kazimierz'
    q_learning(model, name)
