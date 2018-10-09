# 用來生成無孔洞的二值影像

import numpy as np
from matplotlib import pyplot as plt

from skimage import data, io, exposure, img_as_bool
from skimage.morphology import disk, reconstruction
from skimage.filters.rank import gradient
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import os, sys

DATA_DIR = "./Src/"
MODEL_DIR = "./external_contour_goldansample/"
MODEL_DIR2 = "./modelBW2/"
RESULT_DIR = "./RESULT/"
for filename in os.listdir( DATA_DIR ):

    if filename.find('.png') == -1:
        continue
    print(filename)
    modelpath1 = MODEL_DIR + filename
    modelpath2 = MODEL_DIR2 + filename
    Resultfilepath = RESULT_DIR + filename
    model1 =io.imread( modelpath1 )
    model2 = io.imread( modelpath2 )

    model1 = model1 > 0.85
    model2 = model2 > 0.85


    final = model1 & model2

    final = final * 1.0
    io.imsave(Resultfilepath, final)
    # fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    # binary = binary * 1.0
    # ax[0].imshow(binary, interpolation='nearest', cmap=plt.cm.gray)
    # ax[0].axis('off')  
    # filled = filled * 1.0
    # ax[1].imshow(filled, interpolation='nearest', cmap=plt.cm.gray)
    # ax[1].axis('off')

    # plt.tight_layout()
    # plt.show()
