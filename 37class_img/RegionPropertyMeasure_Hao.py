#Get goldan model's feature by regionproperty, and store inforation to txt
import math
import os, sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import figure

# #scikit image
from skimage import data, io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
# from skimage.feature import (match_descriptors, corner_harris,
#                              corner_peaks, ORB, plot_matches)
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops

# scikit learning 
from sklearn.cluster import MeanShift, estimate_bandwidth

# moments
from ZernikeMoments import describe_shapes, parse_shape

MODEL_DIR = "./RESULT/"

# # initialize classifier
# descriptor_extractor1 = ORB(n_keypoints=40, downscale=1.4)
# descriptor_extractor2 = ORB(n_keypoints=20, downscale=1.3)

# thresh_Orientation = 20
#enumerate(sorted(os.listdir(input_path), key=lambda x:int(x[0:-4]))):
for modelname in os.listdir( MODEL_DIR ):
    if modelname.find('.png') == -1:
        print("Log: "+modelname+" not png")
        continue
    print("Log: "+modelname)
    modelpath = MODEL_DIR + modelname
    img_model = io.imread( modelpath )
    label_img = label( img_model )
    moment_image = cv2.imread( modelpath )

    regions = regionprops( label_img )
    
    #======  extract Goldan image feature vector   ======#
    # inter hole
    cnts = label_img.astype('uint8')
    imgbinary ,contours,hierarchy = cv2.findContours(cnts,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print("there are " + str(len(contours)-1) + " inter-contours")
    
    # Solidity, axis ratio
    #inputStr = "%s, %3.5f, %3.5f, %d, %d\n" % (modelname, regions[0].solidity, (regions[0].major_axis_length/regions[0].minor_axis_length), abs(regions[0].euler_number)+1, int(len(contours)-1))
    inputStr = "%s, %3.5f, %3.5f, %d" % (modelname, regions[0].solidity, (regions[0].major_axis_length/regions[0].minor_axis_length), int(len(contours)-1))
    
    # Moments
    moment_shape = describe_shapes( moment_image )
    moment_string = parse_shape( moment_shape )
    
    inputStr = inputStr + moment_string + "\n"
    nameStr = "%s;" % (modelname)
    SaveFilePath = MODEL_DIR + "modelFeatures.txt"
    fo = open( SaveFilePath, "a" )
    fo.write( inputStr )
    fo.close()

