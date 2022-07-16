import cv2 
import numpy as np

def motion_channel(img1, img2, img3):
    hsv_fut  = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    hsv_now  = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hsv_pre  = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    hsv_fut = hsv_fut.astype(np.float64)
    hsv_now = hsv_now.astype(np.float64)
    hsv_pre = hsv_pre.astype(np.float64)

    tot_diff  = (hsv_pre[:,:,2]- hsv_now[:,:,2]) + (hsv_fut[:,:,2] - hsv_now[:,:,2])
    tot_diff  = (tot_diff - np.min(tot_diff))/(np.max(tot_diff) - np.min(tot_diff)) * 255

    return tot_diff.astype(np.uint8)

def motion_channelV2(img1, img2, img3):
    hsv_fut  = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    hsv_now  = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hsv_pre  = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    hsv_fut = hsv_fut.astype(np.float64)
    hsv_now = hsv_now.astype(np.float64)
    hsv_pre = hsv_pre.astype(np.float64)

    tot_diff  = np.abs((hsv_fut[:,:,2]) - hsv_pre[:,:,2]) + np.abs((hsv_fut[:,:,2] - hsv_now[:,:,2]))
    tot_diff  = ((tot_diff - np.min(tot_diff))/(np.max(tot_diff) - np.min(tot_diff)) * 255).astype(np.uint8)

    return tot_diff