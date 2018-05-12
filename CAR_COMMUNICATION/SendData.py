import socket
import pickle
import numpy as np
import time


def action_to_send(data):
    if(data[0][0] < 0.6):
        return 0
    elif(data[0][1] < 0.6):
        return 1
    elif(data[0][2] < 0.6):
        return 2
    elif(data[0][3] < 0.6):
        return 3
    else:
        return 0
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ("192.168.0.32", 9865)
print("starting up on %s port %s" % (server_address[0], server_address[1]))
sock.bind(server_address)

print("\n waiting for client")
data, address = sock.recvfrom(2048)
while data.decode("utf-8") != "READY":
    data, address = sock.recvfrom(2048)
print("received message from %s", address)

answer = 0

answer = pickle.dumps(answer)
sent = sock.sendto(answer, address)

while True:
    try:
        data, address = sock.recvfrom(2048)
        data = pickle.loads(data)
        if np.array_equal(data, np.ones([1,4])):
            sock.close()
            break
        print("Data received:",data)
        action = action_to_send(data)
        sent = sock.sendto(pickle.dumps(action), address)
        time.sleep(0.5)

    except KeyboardInterrupt:
        sent = sock.sendto(pickle.dumps(4), address)
        print("closing server")
        sock.close()
        break

