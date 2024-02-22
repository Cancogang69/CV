import numpy as np
import cv2

cap = cv2.VideoCapture(0)
    
# take first frame of the video
ret, first_frame = cap.read()

# reduce noise
#first_frame = cv2.medianBlur(first_frame, 5)

# setup initial location of window
#track_window = [topLeftX, topLeftY, width, height]
trackWindow = cv2.selectROI("Select object", first_frame, False)

# set up the ROI for tracking
roi = first_frame[int(trackWindow[1]):int(trackWindow[1]+trackWindow[3]), 
                      int(trackWindow[0]):int(trackWindow[0]+trackWindow[2])]

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
h_bins = 30
v_bins = 32
histSize = [h_bins, v_bins]
h_range = [0, 180]
v_range = [0, 256]
ranges = h_range + v_range
channels = [0, 2]
# Get the Histogram and normalize it
hist_roi = cv2.calcHist([hsv_roi], channels, None, histSize, ranges, accumulate=False)
cv2.normalize(hist_roi, hist_roi, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], channels, hist_roi, ranges, 1)
        cv2.imshow("debug mode", dst)

        # apply meanshift to get the new location
        ret, trackWindow = cv2.meanShift(dst, trackWindow, term_crit)

        # Draw it on image
        x,y,w,h = trackWindow
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('result', img2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()