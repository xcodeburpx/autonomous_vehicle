# Code to check connectivity between Pi and laptop over WiFi
# and to check DC engines
# Import libraries
import RPi.GPIO as GPIO
import socket
import time

# Establish UPC socket and add information about server
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('192.168.2.102', 9865)

# Prepare engines to work
# No need of enable pin - using external motor controller
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

Motor1A = 17
Motor1B = 23
Motor2A = 27
Motor2B = 24

GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor2A,GPIO.OUT)
GPIO.setup(Motor2B,GPIO.OUT)

# Define methods to control engine
def DriveUp():
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)

def DriveDown():
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.HIGH)

def DriveRight():
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)

def DriveLeft():
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.LOW)

def RotateRight():
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)

def RotateLeft():
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.HIGH)

def Stop():
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.LOW)

# Python doesn't have switch structure
# So I just improvised a little
# Pi makes a decision from the information that
# arrives from laptop

def take_action(action):
    print("Action:  %s       " % action, sep="", end="\r", flush=True)
    if action == "UP":
        DriveUp()
    elif action == "DOWN":
        DriveDown()
    elif action == "RIGHT":
        DriveLeft()
    elif action == "LEFT":
        DriveRight()
    elif action == "RIGHT_R":
        RotateLeft()
    elif action == "LEFT_R":
        RotateRight()
    elif action == "STOP":
        Stop()
    else:
        Stop()


# Main part - check if everything works

if __name__ == "__main__":

    Stop()
    data = "READY"

    sent = sock.sendto(data.encode("utf-8"), server_address)

    while True:
        try:
            data, address = sock.recvfrom(2048)
            data = data.decode("utf-8")
            if data == "EXIT":
                GPIO.cleanup()
                sock.close()
                break
            else:
                take_action(data)
        except KeyboardInterrupt:
            print("closing client")
            GPIO.cleanup()
            sock.close()
            break


