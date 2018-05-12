import socket
import pickle
import numpy as np
from keras.models import load_model

NUM_SENSORS = 5

print("Loading model...")
model = load_model('saved_models_1/kazimierz200000.model')
print("Loading model complete!")
#SOCKET setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ("192.168.2.100", 9865)
print("starting up on %s port %s" % (server_address[0], server_address[1]))
sock.bind(server_address)

answer = 6
answer = pickle.dumps(answer)

print("\n waiting for client")
data, address = sock.recvfrom(2048)
while pickle.loads(data) != "READY":
    data, address = sock.recvfrom(2048)
print("received message from %s", address)

sent = sock.sendto(answer, address)
print("\n Car prepared for remote control")


while True:
    try:
        data, address = sock.recvfrom(2048)
        data = pickle.loads(data)
        if data == "EXIT":
            sock.close()
            break
        state = data.reshape(1, NUM_SENSORS)
        qval = model.predict(state, batch_size=1)
        action = np.argmax(qval)
        print("Answer: %s     " % action, sep='', end='\r', flush=True)
        action = pickle.dumps(action)
        sent = sock.sendto(action, address)
    
    except KeyboardInterrupt:
        action = pickle.dumps(7)
        sent = sock.sendto(action, address)
        print("closing server")
        sock.close()
        break

