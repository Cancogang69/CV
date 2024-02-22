import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def read_data(path, dir_name):
  frame_file = open(os.path.join(path, dir_name + "_frames.txt"))
  start_frame, end_frame = [int(float(x)) for x in frame_file.read().split(",")]
  groundtruth_file = os.path.join(path, dir_name + "_gt.txt")
  gt = []
  with open(groundtruth_file) as file:
    for line in file:
      gt.append([int(float(x)) for x in line.strip().split(",")])
  frame_path = os.path.join(path, "img")
  frames = []
  for filename in os.listdir(frame_path):
    if(filename.endswith(".jpg") == False):
      continue
    img = cv2.imread(os.path.join(frame_path, filename))
    if img is not None:
        frames.append(img)
  if(len(frames)<end_frame):
    back = end_frame - len(frames)
    start_frame -= back
    end_frame -= back
  return start_frame, end_frame, gt, frames

def IoU(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[0] + ground_truth[2], pred[0] + pred[2])
    iy2 = np.minimum(ground_truth[1] + ground_truth[3], pred[1] + pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] + 1
    gt_width = ground_truth[2] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] + 1
    pd_width = pred[2] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

def removeBackground(img, debug=False):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # onvert image to black and white
  _, image_edges = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

  # create background mask
  mask = np.zeros(img.shape, np.uint8)
  mask.fill(255)

  # get most significant contours
  contours_mask, _ = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  # most significant contours traversal
  for contour in range(len(contours_mask)):
    # create mask
    if contour != 1:
      cv2.fillConvexPoly(mask, contours_mask[contour], (0, 0, 0))

  img2 = cv2.bitwise_and(img, mask)
  if debug:
    cv2.imshow('Original', img)
    cv2.imshow('image after masked', img2)
  return img2

def remove_background2(img, bbox):
  mask =	np.zeros(img.shape[:2],np.uint8)
 
 
  bgdModel =  np.zeros((1,65),np.float64)
  fgdModel =  np.zeros((1,65),np.float64)
  
  cv2.grabCut(img,mask,bbox,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
  
  
  mask2  =  np.where((mask==2)|(mask==0),0,1).astype('uint8')
  mask2 = (mask2 * 255).astype("uint8")
  if debug:
    cv2.imshow("mask", mask2)
  output = cv2.bitwise_and(img, img, mask=mask2)
  return output

def remove_background3(img):
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
  #Take S and remove any value that is less than half
  s = img_hsv[:,:,1]
  s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded

  # We increase the brightness of the image and then mod by 255
  v = (img_hsv[:,:,2] + 127) % 255
  v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask

  # Combine our two masks based on S and V into a single "Foreground"
  foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer

  background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
  background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
  foreground=cv2.bitwise_and(img,img,mask=foreground) # Apply our foreground map to original image
  finalimage = background+foreground # Combine foreground and background

  return finalimage

def mean_shift(start_frame, end_frame, gt, frames, rmbg_method=0):
  # take first frame of the video
  first_frame = frames[start_frame]

  # setup initial location of window
  # track_window = [topLeftX, topLeftY, width, height]
  track_window = gt[0]
  # set up the ROI for tracking
  x, y, w, h = track_window
  roi = first_frame[int(y):int(y+h), int(x):int(x+w)]

  if rmbg_method == 0:
    roi = removeBackground(roi, True)
  elif rmbg_method == 1:
    ff_rmbg = remove_background2(first_frame, track_window) 
    roi = ff_rmbg[int(y):int(y+h), int(x):int(x+w)]
  elif rmbg_method == 2:
    roi = remove_background3(roi)
  
  cv2.imshow("roi", roi)

  #channel 0 is hue and 1 is value
  #h_bins, v_bins = hue and value levels
  h_bins = 30
  v_bins = 25
  histSize = [h_bins, v_bins]
  h_range = [0, 180]
  v_range = [0, 256]
  ranges = h_range + v_range # Concat list
  channels = [0, 2]
  #convert roi to hsv color space
  hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
  #calculate and normalize histogram of roi
  hist_roi = cv2.calcHist([hsv_roi], channels, None, histSize, ranges, accumulate=False)
  cv2.normalize(hist_roi, hist_roi, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  
  TP = 0
  FP = 0
  
  # Setup the termination criteria, either 10 iteration or move by at least 1 pt
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

  for i in range(start_frame+1, end_frame):
    frame = frames[i]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], channels, hist_roi, ranges, 1)
    dst = cv2.medianBlur(dst, 5)
    
    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    gt_index = i - start_frame
    if debug:
      x,y,w,h = track_window
      xt, yt, wt, ht = gt[gt_index]
      img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
      img2 = cv2.rectangle(frame, (xt, yt), (xt+wt, yt+ht), (0,0,255), 2)
      cv2.imshow("back projection image", dst)
      cv2.imshow('result', img2)

      #press esc to stop tracking
      k = cv2.waitKey(60) & 0xff
      if k == 27:
          break
    
    if(IoU(gt[gt_index], track_window) >= 0.5):
      TP+=1
    else:
      FP+=1

  return TP, FP

path = [".\\Airport_ce\\Airport_ce", ".\\Basketball\\Basketball", ".\\Busstation_ce1\\Busstation_ce1"]
dir_name = ["Airport_ce", "Basketball", "Busstation_ce1"]
item = 1
debug = True
rmbg_method = -1
start_frame, end_frame, gt, frames = read_data(path[item], dir_name[item])
TP, FP = mean_shift(start_frame, end_frame, gt, frames, rmbg_method)
print(TP/(TP+FP))