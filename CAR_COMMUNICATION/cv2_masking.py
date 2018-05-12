import cv2
import numpy as np

# Set the camera
cap = cv2.VideoCapture(1)
# cap.set(3, 1280)
# cap.set(4, 1028)

# Set filters
red = np.uint8([[[173,198,0]]])
blue = np.uint8([[[255,0,0]]])

# For later use
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)

# Set thresholds for a color
lower_blue = np.array([hsv_blue[0][0][0]-10, 100, 100])
upper_blue = np.array([hsv_blue[0][0][0]+10, 255,255])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# Check
# print(lower_blue)
# print(upper_blue)

while True:
    # Capture the frame
    _, frame = cap.read()

    # BGR -> HSL
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a color mask
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask
    res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
    # print(type(res_blue))

    # HSV -> BRG
    res_blue = cv2.cvtColor(res_blue, cv2.COLOR_HSV2BGR)
    # BGR -> GRAY
    res_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)

    # Create the circles
    circles = cv2.HoughCircles(res_blue, cv2.HOUGH_GRADIENT, 1, 30,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    # check if there is none
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Visualise this circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0,255,0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0,0,255), 3)

    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('blue_mask', res_blue)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cv2.VideoCapture(1).release()
