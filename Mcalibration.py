import cv2
import numpy as np

img = cv2.imread('ros_frame.jpg')
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0,0,255), -1)
        cv2.imshow('img', img)
        if len(points) == 4:
            print("4 points:", points)
            np.save("spot_corners.npy", np.array(points, dtype=np.float32))

cv2.imshow('img', img)
cv2.setMouseCallback('img', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
