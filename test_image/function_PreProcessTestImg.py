import numpy as np
from matplotlib import pyplot as plt

from skimage import data, io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
import os, sys


def PreProcessTestImg(test_img):
    img_gray = rgb2gray( test_img )
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh

    filled = remove_small_objects(binary, min_size=int(binary.size/25), connectivity=2 ) 

    filled = filled * 1.0
    return filled

'''
DATA_DIR = "test_image\\Src\\"
DEST_DIR = "test_image\\BW\\"
for filename in os.listdir( DATA_DIR ):

    if filename.find('.jpg') == -1:
        continue
    if filename.find('original') == -1:
        continue
    
    filepath = DATA_DIR + filename
    destfilepath = DEST_DIR + filename
    pngfilepath = destfilepath.replace('jpg','png')  

    img =io.imread( filepath )
    img_gray = rgb2gray( img )
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh

    filled = remove_small_objects(binary, min_size=int(binary.size/15), connectivity=2 ) 

    filled = filled * 1.0
    io.imsave( pngfilepath, filled )
'''
    
