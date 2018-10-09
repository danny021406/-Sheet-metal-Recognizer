#此程式用來將測試影像進行預處理，去除二值影像中的內外太小的點

import numpy as np
from matplotlib import pyplot as plt

from skimage import data, io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
import os, sys

#DATA_DIR = "test_image\\Src\\"
#DEST_DIR = "test_image\\BW\\"

DATA_DIR = "./Src/"
DEST_DIR = "./BW/"
for filename in os.listdir( DATA_DIR ):

    if filename.find('.jpg') == -1:
        continue
    #if filename.find('original') == -1:
        #continue
    
    filepath = DATA_DIR + filename
    destfilepath = DEST_DIR + filename
    pngfilepath = destfilepath.replace('jpg','png')  # 建議轉存成png

    img =io.imread( filepath )
    img_gray = rgb2gray( img )
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh

    filled = remove_small_objects(binary, min_size=int(binary.size/25), connectivity=2 ) # 移除小於6.7%以下的物件

    filled = filled * 1.0
    io.imsave( pngfilepath, filled )

    
