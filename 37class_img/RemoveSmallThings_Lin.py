# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:46:05 2018

@author: NCHC
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

from skimage import data, io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
import os, sys

DATA_DIR = "./Src/"
MODEL_DIR = "./modelBW2/"
for filename in os.listdir( DATA_DIR ):

    if filename.find('.png') == -1:
        continue
    print(filename)
    filepath = DATA_DIR + filename
    destfilepath = MODEL_DIR + filename

    img =cv2.imread( filepath )
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    img_gray= cv2.erode(img_gray,kernel,iterations = 1)
    (thresh, im_bw) = cv2.threshold(img_gray, 128, 255, 0)
    
    

    (_, cnts, hierarchy) = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("##################")
    print(hierarchy[0])
    #cv2.imshow("threshold", im_bw)
    #cv2.waitKey(0)
    clone = im_bw.copy()
	
   
  
    for i in range(len(cnts)):
        if hierarchy[0][i][2] == -1 :
            if hierarchy[0][hierarchy[0][i][3]][3]==0:
                cv2.drawContours(clone, cnts, i, (255), cv2.FILLED, 4)
            else:
                cv2.drawContours(clone, cnts, i, (0), cv2.FILLED, 4)            
        else:
            cv2.drawContours(clone, cnts, i, (255), cv2.FILLED, 4)
            
                
    cv2.imwrite( destfilepath, clone)