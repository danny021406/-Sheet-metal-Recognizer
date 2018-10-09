#載入必要模組
from scipy.spatial import distance as dist
import numpy as np
import mahotas
import cv2, os
#import imutils
import argparse
import matplotlib.pyplot as plt
 
#接收葉片類表圖及目標葉片的path
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, help="Path to the source of shapes")
ap.add_argument("-f", "--features", required=False, help="text file for features")
args = vars(ap.parse_args())
 
def describe_shapes(image, extract_feature = True):
	# 此陣列用來放置相片中各物件的moments
	shapeFeatures = []
	#將相片進行灰階、高斯模糊及二極化處理。
	if(extract_feature):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = (image)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]
	#針對圖片進行侵蝕（Erosion）與膨脹（dilation），可消除過於細膩的邊緣形狀及內部的小空洞。
	thresh = cv2.dilate(thresh, None, iterations=4)
	thresh = cv2.erode(thresh, None, iterations=2)
	#找出輪廓
	imgbinary, cnts, hierarchy  = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	c = cnts[0]
	#產生該輪廓的mask，並繪出其邊緣
	mask = np.zeros(image.shape[:2], dtype="uint8")
	#取得該輪廓四邊形外框的x, y, w, h
	(x, y, w, h) = cv2.boundingRect(c)
	#從mask圖檔中取該四邊形，此即為該物件的shape
	cv2.drawContours(mask, [c], -1, 255, -1)
	roi = mask[y:y + h, x:x + w]
	#cv2.imwrite("img/" + name, roi)
	#features = cv2.HuMoments(cv2.moments(roi)).flatten()
	#計算該物件的Zernike moments，半徑r用cv2.minEnclosingCircle指令取得。
	features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
	 
	#傳回物件及Zernike moments
	return [features]

def parse_shape(shape):
	inputStr = "" 
	for i in range(len(shape[0])): 
		inputStr = inputStr + ', ' + str(shape[0][i])
	return inputStr

if __name__ == '__main__':
	#取得Zernike moments
	for modelname in os.listdir( args["source"] ):
	    if modelname.find('.png') == -1:
	        print("Log: "+modelname+" not png")
	        continue
	    print("Log: "+modelname)
	    modelpath = args["source"] + modelname
		#read image
	    shapesImage = cv2.imread(modelpath)
	    #zernike moment function
	    shapeFeatures = describe_shapes(shapesImage)
	    #parse string to feature text file
	    inputStr = parse_shape(shapeFeatures)
	    
	    fo = open( args["features"], "a" )
	    fo.write( inputStr )
	    fo.close()
