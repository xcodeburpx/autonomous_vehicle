import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ("192.168.2.100", 9865)
print("Starting up on %s port %s" % server_address)
sock.bind(server_address)

while True:
    print("\nwatiing for receive message")
    data, address = sock.recvfrom(4096)
    
    print("received %s bytes from %s" % (len(data), address))
    print(data.decode("utf-8"))
    
    if data:
        sent = sock.sendto(data, address)
        print("sent %s bytes back to %s" % (sent, address))

