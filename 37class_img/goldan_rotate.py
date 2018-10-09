import cv2
import os
import math
import numpy as np
from matplotlib import pyplot as plt
import random

plt.close('all')

inpath = './RESULT/' #can't have chinese name in path
outpath = './37class_rotate/hor_name/'

name_txt = outpath + "contour_info.txt"
f_feature = open(name_txt, 'w')

# find the horizontal weight
def horizontal_weight(img,left_x,left_y,h,w):
    #h,w = img.shape[:2]
    h_half = int(h/2)
    w = int(w)
    left_x = int(left_x)
    left_y = int(left_y)
    top_sum = 0
    bot_sum = 0
    rotate = False
    for j in range(h_half):
        for i in range(w):
            x = i+left_x
            top_y = j+left_y
            bot_y = j+left_y+h_half
            p_top = img[top_y,x]
            p_bot = img[bot_y,x]
            top_sum += p_top
            bot_sum += p_bot
    
    if(top_sum > bot_sum):
        rotate = True

    return rotate

# and images
def and_images(img,mask):
    h,w = img.shape[:2]
    out_img = img
    for j in range(h):
        for i in range(w):
            if mask[j,i][1] == 0:
                out_img[j,i] = 0
    
    return out_img

# find contour and rotate the image
def feature_extract(infile,name):
    
    print(name)
    img = cv2.imread(infile)
    height, width = img.shape[:2]
    #print(height,width)
    
    # RGB to Gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #find contour
    finded = False
    _, contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(contours)):
        #cnt = contours[0]
        
        rect = cv2.minAreaRect(contours[cnt])
        x, y = rect[0]  # center
        w,h = rect[1]   # width height
        
        contour_size = w*h
        image_size = width*height
        
        print(image_size, contour_size)
        
        if contour_size > 0 and finded == False:  ###  image_size/16
            x,y = rect[0]  # center
            w,h = rect[1]   # width height
            angle = rect[2] # angle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # if h>w
            if h>w :
                angle -= 90
            
            
            print('%s    center:(%d,%d),w:%d,h:%d,angle:%d\n'%(name,x,y,w,h,angle))
            #contour_info = '%s    center:(%d,%d),w:%d,h:%d,angle:%d\n'%(name,x,y,w,h,angle)
            #f_feature.write(contour_info)
            
            cv2.drawContours(img_gray,[box], -1, (0, 255, 0), 2)
            #cv2.drawContours(img_gray,[box],-1,(0,0,255),2)
            
            mask = np.zeros(img.shape, dtype="uint8")  # define contour mask
            cv2.drawContours(mask, [box], -1, (255,255,255), -1) # in mask is white
            object_img = and_images(img_gray,mask)     # denoise, get object
            
        
            rows, cols = img.shape[:2]
            length = max(rows,cols)+500
            #@ transfer and rotation the object
            s_x = length/2 - x
            s_y = length/2 - y
            print(s_x,s_y)
            H = np.float32([[1,0,s_x],[0,1,s_y]])
            img2 = cv2.warpAffine(object_img, H, (length, length))
            M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1)
            img3 = cv2.warpAffine(img2, M, (length, length))
            
#            name_png2 = outpath + 'denoise_' + name
#            cv2.imwrite(name_png2,img3)
            
            #@ change center to the buttom (rotate 180)
            #get no rotate object
            _2, contours2, hierarchy2 = cv2.findContours(img3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt_f = contours2[0]
            for cnt2 in range(len(contours2)):
                rect2 = cv2.minAreaRect(contours2[cnt2])
                w2, h2 = rect2[1]   # width height
                print('123 ',w2,h2)
                if w2*h2 > h*w/2:
                    cnt_f = contours2[cnt2]
                    break
                
            
            #cnt2 = contours2[1]
            rect2 = cv2.minAreaRect(cnt_f)
            x2, y2 = rect2[0]  # center
            w2, h2 = rect2[1]   # width height
            
            print('%s    center:(%d,%d),w:%d,h:%d,length/2:%d\n'%(name,x2,y2,w2,h2,length/2))
#            
            left_x = x2-w2/2
            left_y = y2-h2/2
            rotate180 = horizontal_weight(img3,left_x,left_y,h2,w2)
            if rotate180 == True:
                rotate180 = 180
            else:
                rotate180 = 0
            
            print(rotate180)
            M = cv2.getRotationMatrix2D((length/2, length/2), rotate180, 1)
            img4 = cv2.warpAffine(img3, M, (length, length))
            name_png2 = outpath + 'hor180_' + name
            cv2.imwrite(name_png2,img4)
            
            # crop final output
            s_x = int(left_x)
            s_y = int(left_y)
            e_x = int(s_x+w2)
            e_y = int(s_y+h2)
            crop_img = img4[s_y:e_y, s_x:e_x]
            name_png2 = outpath + '0000_' + name
            #cv2.imwrite(name_png2,crop_img)
            
            finded = True
            
            return crop_img

# @@ process single image
#infile = 'D:\\Research\\Project2\\original\\binary\\bin_1536158910.69_original.jpg.png'
#name = 'bin_1536158910.69_original.jpg.png'
#output = feature_extract(infile,name)

# @@ process folder
for root, dirs, files in os.walk(inpath, topdown=False):
    idx = 0
    for name in files:
        if name.endswith(".png"):
            infile = os.path.join(root, name)
            
            feature_extract(infile,name)
            #rotate(infile,name)
            #affine_rotate(infile,name)
            
            print(idx)
            idx += 1
            plt.close('all')

f_feature.close()
