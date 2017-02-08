import environment
import numpy as np
from neural_nets import kazimierz, jarvis
from neural_nets.history_loss import LossHistory
import sys
import timeit
import random
import csv

#Macros for q-learning
GAMMA=0.9
BATCH_SIZE=32
BUFFER = 30000
NUM_INPUT = 4

"""Credits to  Matt Harvey (harvitronix)"""

# TO DO -> create class 'Q-learining' and 'TD(lambda)-learning'

def q_learning(model, name):

    model_name = name

    #Initial values for epsilon policy
    eps_delay=1000
    epsilon=1
    train_frames = 1000000
    counter = 0

    #Used to create a car distance plot
    max_car_distance = 0
    car_distance = 0
    t = 0
    data_collect = []

    #Replay memory and loss log lists
    replay_mem = []
    loss_log = []

    #Start the environment
    world = environment.Env()

    #Initial reward and state -> ignoring the reward
    _, state = world.screen_snap(0)

    #Measure the speed of environment -> fps
    start_time = timeit.default_timer()

    #MAIN LOOP
    while counter < train_frames:

        car_distance += 1
        counter += 1

        #Epsilon policy
        if random.random() < epsilon or counter < eps_delay:
            action = np.random.randint(0,4)
        else:
            qval = model.predict(state, batch_size=1)
            action = np.argmax(qval)

        # Reward and new state
        reward, new_state = world.screen_snap(action)

        # Storing the (S,A,R,S') tuple in replay memory
        replay_mem.append((state, action, reward, new_state))

        #If the counter has delayed -> start learning the net
        if counter > eps_delay:

            #Only the fresh data we need
            if (len(replay_mem)) > BUFFER:
                replay_mem.pop(0)

            #Random sample
            minibatch = random.sample(replay_mem, BATCH_SIZE)

            #Update the reward -> take the importance of states into account
            X_train, y_train = update_func(minibatch, model)

            # Train the model on this batch.
            history = LossHistory()
            model.fit(
                X_train, y_train, batch_size=BATCH_SIZE,
                nb_epoch=1, verbose=0, callbacks=[history]
            )
            #print("Model trained!\n")
            loss_log.append(history.losses)
        # S <- S'
        state = new_state

        #Epsilon decrementation
        if epsilon > 0.1 and counter > eps_delay:
            epsilon -=(1/train_frames)

        #Info window
        if reward == -700:
            data_collect.append([t, car_distance])

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

        #Saving the model and weights
        if counter % 25000 == 0:
            model.save_weights("saved_weights/" + model_name +
                               str(counter) + ".h5")
            model.save("saved_models/" + model_name +
                       str(counter) + ".model")
            print("Saving model %s - %d" %(model_name, t))

    log_results(model_name, data_collect, loss_log)

def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/sonar-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/sonar-frames/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

def update_func(minibatch, model):
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_state_m = old_state_m.reshape(1,NUM_INPUT)
        old_qval = model.predict(old_state_m, batch_size=1)
        # Get prediction on new state.
        new_state_m = new_state_m.reshape(1,NUM_INPUT)
        newQ = model.predict(new_state_m, batch_size=1)
        # Get our best move. I think?
        maxQ = np.max(newQ)
        y = np.zeros((1, 4))
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != -700:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT, ))
        y_train.append(y.reshape(4, ))

    X_train = np.array(X_train)
    y_train = np.array(y_train)


    return X_train, y_train

if __name__ == '__main__':
    model = kazimierz.Kazimierz()
    name = 'kazimierz'
    q_learning(model, name)