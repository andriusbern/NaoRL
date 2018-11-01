import numpy as np
import time
import cv2 as cv

def threshold(image, low, high):
    """
    Apply thresholding to an image
    """ 
    im = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
    retval, thresholded = cv.threshold(im, low, high, cv.THRESH_BINARY)
    return thresholded


def resize(image, dim):
    """
    Resize an image
    """
    if type(dim) is not tuple: dim = tuple(dim)
    
    resized = cv.resize(image, dim)
    return resized


def ball_tracking(im, resolution, display=None, draw=True):
    """
    Tracks a green ball
    """ 
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    scale = 1

    
    hsv = cv.cvtColor(np.asarray(im), cv.COLOR_BGR2HSV)
    im = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
    small = cv.resize(hsv, (0,0), fx=scale, fy=scale) 
    
    # Create a mask for the green areas of the image
    mask = cv.inRange(small, greenLower, greenUpper)
    # Erosion and dilation to remove imperfections in masking
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # Find the contours of masked shapes
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[1]
    center = None
    
    # If there is a masked object
    if len(contours) > 0:
        # Largest contour
        c = max(contours, key=cv.contourArea)
        # Radius
        ((x, y), radius) = cv.minEnclosingCircle(c)
        # Moments of the largest contour
        moments = cv.moments(c)
        center = (int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]))

        if draw:
            # Draw appropriate circles
            if radius > 2:
                cv.circle(im, (int(x/scale), int(y/scale)), int(radius/scale), (0, 255, 255), 2)
                cv.circle(im, (int(center[0]/scale), int(center[1]/scale)), 5, (0, 0, 255), -1)

    if display is not None:
        cv.imshow(display, cv.flip(im, 0))
        key = cv.waitKey(1) & 0xFF

    return im, center, resolution
