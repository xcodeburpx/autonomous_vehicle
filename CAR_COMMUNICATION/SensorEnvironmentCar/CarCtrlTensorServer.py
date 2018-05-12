import os
import socket
import pickle
import cv2
import tensorflow as tf
import numpy as np
from tensor_networks.Double_Deep_Q_Network import DoubleQNetwork
from tensor_networks.Dueling_Deep_Q_Network import DuelingDQN
from steering_utils import steering_angle
from cv2_get_coodinates import get_coodinates, get_target_coords
from macro_utils import *

# Macros to change
NUM_SENSORS = 5
# TARGET_POS = np.array([100,400])
SENSOR_THRESH = -0.3
DUELING = False

# Save models to the path
path = os.path.abspath("models")
if not DUELING:
    model = path + "/DoubleDQN-237500.ckpt"
else:
    model = path + "/DuelingDQN-130000.ckpt"


e_gred = None
eps = 1.0

tf.reset_default_graph()

# Model loading
# Create Our Network
print("Loading NN model...")
sess = tf.Session()
if not DUELING:
    with tf.variable_scope("Double_DQN"):
        modelNetwork = DoubleQNetwork(n_actions=ACTION_SPACE,reward_decay=0.8, state_shape=OBSERVATION_SPACE, memory_size=MEMORY_SIZE,
                                    e_greedy_increment=e_gred, epsilon=eps, double_q=True, sess=sess, output_graph=False,replace_target_iter=400)
else:
    with tf.variable_scope("Dueling_DQN"):
        modelNetwork = DuelingDQN(n_actions=ACTION_SPACE, state_shape=OBSERVATION_SPACE, memory_size=MEMORY_SIZE,
                                 e_greedy_increment=e_gred, epsilon=eps, double_q=True, dueling=True, sess=sess,
                                 output_graph=False)


sess.run(tf.global_variables_initializer())
# Load the model
saver = tf.train.Saver()
saver.restore(sess, model)
print("Model loaded")

# Camera connection
print("Getting camera ready..")
cap = cv2.VideoCapture(1)
print("Is camera connected:", cap.isOpened())

# Connection established
print("Creating socket...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ("192.168.2.100", 9865)
print("starting up on %s port %s" % (server_address[0], server_address[1]))
sock.bind(server_address)
print("Socket created")


# Connection information
answer = [6, 0] # STOP action
answer = pickle.dumps(answer)

print("\n waiting for client")
data, address = sock.recvfrom(4096)
while pickle.loads(data) != "READY":
    data, address = sock.recvfrom(4096)
print("received message from %s", address)

sent = sock.sendto(answer, address)
print("\n Car prepared for remote control")


data = input("Are you ready?")

# Getting data for initialization
TARGET_POS = np.array([])
past_position = np.array([])

print("Target coordinates")
while TARGET_POS.shape[0] == 0 or TARGET_POS[0] == 0 or TARGET_POS[1] == 0:
    TARGET_POS = get_target_coords(cap)

print("Target position:", TARGET_POS)

while past_position.shape[0] == 0 or past_position[0] == 0 or past_position[1] == 0:
    past_position = get_coodinates(cap)
print("Past position:", past_position)

counter = 1

while True:
    try:
        data, address = sock.recvfrom(4096)
        data = pickle.loads(data)
        if data == "EXIT":
            sock.close()
            break

        car_position = get_coodinates(cap)
        if car_position.shape[0] == 0:
            angle = 0
        else:
            angle = steering_angle(TARGET_POS, car_position, past_position)
            if counter % 2 == 0:
                past_position = car_position
                counter = 0
            else:
                counter += 1


        target_past_vec = TARGET_POS - past_position
        if np.linalg.norm(target_past_vec) <= 25:
            print("TARGET REACHED!!!!")
            to_sent = pickle.dumps([7, 0])
            sent = sock.sendto(to_sent, address)
            print("Closing server")
            sock.close()
            print("Closing camera capture")
            cap.release()
            cv2.destroyAllWindows()
            print("Closing session")
            sess.close()
            break

        # print("\n CAR shape:",car_position.shape,"\n")
        if car_position.shape[0] != 0:
            if car_position[0] <=30 or car_position[0] >=600 or car_position[1] <= 30 or car_position[1] >= 440:
                angle = np.pi
                action = 0

        state = data.reshape(NUM_SENSORS,)
        if any(i < SENSOR_THRESH for i in state):
            action = modelNetwork.choose_action(state)
        else:
            action = 0
        if any(i < -0.8 for i in state):
            action = 1

        print("Action: ", action, " Angle: ", angle, " Position: ", car_position)
        to_sent = [action, angle]
        to_sent = pickle.dumps(to_sent)
        sent = sock.sendto(to_sent, address)
    except KeyboardInterrupt:
        to_sent = pickle.dumps([7,0])
        sent = sock.sendto(to_sent, address)
        print("Closing server")
        sock.close()
        print("Closing camera capture")
        cap.release()
        cv2.destroyAllWindows()
        print("Closing session")
        sess.close()
        break

# print("Closing server")
# sock.close()
# print("Closing camera capture")
# cap.release()
# cv2.destroyAllWindows()
# print("Closing session")
# sess.close()
