import cv2
import numpy as np

# MACROS
# Filter for our car
BLUE = np.uint8([[[255, 234, 116]]])
GREEN = np.uint8([[[156,243,149]]])
# GREEN = np.uint8([[[20,255,20]]])

# BGR -> HSV
HSV_BLUE = cv2.cvtColor(BLUE, cv2.COLOR_BGR2HSV)
HSV_GREEN = cv2.cvtColor(GREEN, cv2.COLOR_BGR2HSV)

# Set thresholds for a color
LOWER_BLUE = np.array([HSV_BLUE[0][0][0] - 15, 100, 100])
UPPER_BLUE = np.array([HSV_BLUE[0][0][0] + 15, 255, 255])

LOWER_GREEN = np.array([HSV_GREEN[0][0][0]-15, 50, 50])
UPPER_GREEN = np.array([HSV_GREEN[0][0][0]+15, 255,255])



def get_coodinates(cap):
    # Connect the camera
    #cap = cv2.VideoCapture(1)

    # For later use
    # cap.set(3, 1280)
    # cap.set(4, 1028)

    # Get the frame
    rate, frame = cap.read()

    output = frame.copy()

    # Blurring for getting circle shape
    frame = cv2.GaussianBlur(frame,(9,9),0)
    frame = cv2.medianBlur(frame,5)

    # BGR -> HSL
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a color mask
    mask_blue = cv2.inRange(hsv_frame, LOWER_BLUE, UPPER_BLUE)

    # Apply the mask
    res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # HSV -> BRG
    res_blue = cv2.cvtColor(res_blue, cv2.COLOR_HSV2BGR)
    # BGR -> GRAY
    res_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)

    # Threshold for something
    res_blue = cv2.adaptiveThreshold(res_blue,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY,11,3.5)

    # Erode and dilate -> adapt kernel
    kernel = np.ones((4,4), np.uint8)
    res_blue = cv2.erode(res_blue, kernel,iterations=1)
    res_blue = cv2.dilate(res_blue, kernel, iterations=1)

    # Create the circles -> Adapt parameters
    circles = cv2.HoughCircles(res_blue, cv2.HOUGH_GRADIENT, 1.2, 400, param1=80, param2=40, minRadius=30, maxRadius=60)

    coords = []
    coords = np.array(coords)

    if circles is not None and len(circles) == 1:
        coords = [circles[0,0,0], circles[0,0,1]]
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow('frame', output)
    # cv2.imshow('blue_mask', res_blue)
    return np.array(coords)

def get_target_coords(cap):
    rate, frame = cap.read()

    output = frame.copy()

    # Blurring for getting circle shape
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    frame = cv2.medianBlur(frame, 5)

    # BGR -> HSL
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a color mask
    mask_green = cv2.inRange(hsv_frame, LOWER_GREEN, UPPER_GREEN)

    # Apply the mask
    res_green = cv2.bitwise_and(frame, frame, mask=mask_green)

    # HSV -> BRG
    res_green = cv2.cvtColor(res_green, cv2.COLOR_HSV2BGR)
    # BGR -> GRAY
    res_green = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)

    # Threshold for something
    res_green = cv2.adaptiveThreshold(res_green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY, 11, 3.5)

    # Erode and dilate -> adapt kernel
    kernel = np.ones((4, 4), np.uint8)
    res_green = cv2.erode(res_green, kernel, iterations=1)
    res_green = cv2.dilate(res_green, kernel, iterations=1)

    # Create the circles -> Adapt parameters
    circles = cv2.HoughCircles(res_green, cv2.HOUGH_GRADIENT, 1.2, 400, param1=80, param2=40, minRadius=0, maxRadius=30)

    coords = []
    coords = np.array(coords)

    print("Number of circles:")
    if circles is not None and len(circles) == 1:
        coords = [circles[0, 0, 0], circles[0, 0, 1]]
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow('frame', output)
    # cv2.imshow('blue_mask', res_green)
    return np.array(coords)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())

    while True:
        # coords = get_coodinates(cap)
        coords = get_target_coords(cap)
        print("Coordinates:",coords, "       ", sep="", end="\n")

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
