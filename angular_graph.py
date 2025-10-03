
import cv2
import numpy as np

def nothing(x):
    pass

# Open video (or webcam if you prefer)
cap = cv2.VideoCapture("pos80_trial1.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a window for the trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for lower HSV
cv2.createTrackbar("LH", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("LS", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 0, 255, nothing)

# Create trackbars for upper HSV
cv2.createTrackbar("UH", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show windows
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered Result", result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):  # press q to quit
        print("Lower HSV:", lower.tolist())
        print("Upper HSV:", upper.tolist())
        break

cap.release()
cv2.destroyAllWindows()
