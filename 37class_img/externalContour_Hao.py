# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 18:09:44 2018

@author: NCHC
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

plt.close('all')

inpath = './Src/'
externalContour = './external_contour_goldansample/'

for filename in os.listdir(inpath):
    if filename.find('.png') == -1:
        continue
    infile = inpath + filename
    print(infile)
    img = cv2.imread(infile)
    #print(len(img.shape))
    height, width = img.shape[:2]
    
    # RGB to Gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5,5),0)
    
    # Gray to Binary
    retval, img_binary = cv2.threshold(blur, 216, 255, cv2.THRESH_BINARY_INV)
    
    # Copy the thresholded image.
    im_floodfill = img_binary.copy()
    
    # Mask used to flood filling.
    mask = np.zeros((height+2, width+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the external contour.
    im_out = img_binary | im_floodfill_inv
    
    '''
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    eroded = cv2.erode(img_binary,kernel)

    cv2.imshow('Thresholded', img_binary)
    cv2.imshow('Inverted Floodfilled', im_floodfill_inv)
    cv2.imshow('Output', im_out)
    cv2.waitKey(0)
    '''
    # Save image
    cv2.imwrite(externalContour + filename, im_out)
    
cv2.destroyAllWindows()
   
