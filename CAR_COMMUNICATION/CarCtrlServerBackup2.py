import socket
import pygame
from pygame import *

#PYGAME setup
HEIGHT = 500
WIDTH = 500
#BG_COLOR = THECOLORS['black']
flags = pygame.DOUBLEBUF

pygame.init()
display = pygame.display.set_mode((WIDTH, HEIGHT),flags)
clock = pygame.time.Clock()

#SOCKET setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ("192.168.2.100", 9865)
print("starting up on %s port %s", server_address)
sock.bind(server_address)

# MOTION functions
def DriveUp():
    return "UP"

def DriveDown():
    return "DOWN"

def DriveRight():
    return "RIGHT"

def DriveLeft():
    return "LEFT"

def RotateRight():
    return "RIGHT_R"

def RotateLeft():
    return "LEFT_R"

def Stop():
    return "STOP"

#INITIALIZATION
answer = Stop()

print("\n waiting for client")
data, address = sock.recvfrom(2048)
while data.decode("utf-8") != "READY":
    data, address = sock.recvfrom(2048)
print("received message from %s", address)

answer = answer.encode("utf-8")
sent = sock.sendto(answer, address)
print("\n Car prepared for remote control")


pygame.key.set_repeat(1,1)
#MAIN function
while True:
    try:
        #message, address = sock.recvfrom(2048)
        #message = message.decode("utf-8")
        #if message == "EXIT":
        #    sock.close()
        #    break

        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_w]:
                    if keys[pygame.K_d]:
                        answer = DriveRight()
                    elif keys[pygame.K_a]:
                        answer = DriveLeft()
                    else:
                        answer = DriveUp()
                elif keys[pygame.K_s]:
                    answer = DriveDown()
                elif keys[pygame.K_d]:
                    answer = RotateRight()
                elif keys[pygame.K_a]:
                    answer = RotateLeft()
                else:
                    continue
            else:
                answer = Stop()

        print("Send %s       " % answer, sep="", end="\r", flush=True)
        #print("Send %s     " %  answer, sep="", end="\r", flush=True)
        sent = sock.sendto(answer.encode("utf-8"), address)
        pygame.display.flip()
        #pygame.display.update()
        clock.tick(60)
    
    except KeyboardInterrupt:
        sent = sock.sendto("EXIT".encode("utf-8"), address)
        print("closing server")
        sock.close()
        break

