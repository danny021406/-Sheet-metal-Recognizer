import os
import cv2
#import json
import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib import figure

# #scikit image
from skimage import io
#from skimage.color import rgb2gray
#from skimage.filters import threshold_otsu
# from skimage.feature import (match_descriptors, corner_harris,
#                              corner_peaks, ORB, plot_matches)
#from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops

# scikit learning 
#from sklearn.cluster import MeanShift, estimate_bandwidth
from function_shape_extract import feature_extract, feature_extract2, feature_extract3,  get_compare_imgs
from function_PreProcessTestImg import PreProcessTestImg
import time

# moments
from scipy.spatial import distance as dist
from ZernikeMoments import describe_shapes

fixed_dict_37class = {}   # dictionary for goldan name
list_37class = []         # list for goldan feature
Results_DIR = "./Results/"
Result_txt_path = Results_DIR+"RESULT.txt"
Result2_txt_path = Results_DIR+"RESULT2.txt"

match_sum = 0
match_number = 0
final_candidate_number = 0


def feature_compare(test_path, dict_goaldan):
    
    fo_2 = open( Result2_txt_path, "a" )
    
    fo_2.write( "FileName = "+ os.path.basename(test_path) +"\n")
    fo_2.write( "Total candicates is " + str(len(dict_goaldan)) + "\n")
    
    goldenpath = '../37class_img/37class_rotate/hor_name/' #can't have chinese name in path
    in_img = cv2.imread(test_path)
    in_crop_img = feature_extract2(in_img)
    in_h,in_w = in_crop_img.shape[:2]
    print(in_w,in_h)
    
    MSE_array = np.zeros([len(dict_goaldan),2], dtype=np.float64)
    
    MSE_idx_name = []
    idx = 0
    for key,values in dict_37class.items():
        #ody, ext = os.path.splitext(os.path.basename(values))
        file_name = goldenpath + 'hor180_' + str(values)
        print(file_name) 
        img = cv2.imread(file_name)
        height, width = img.shape[:2]
        #img = cv2.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_LINEAR)
        #height, width = img.shape[:2]
    
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        x,y = rect[0]  # center
        w,h = rect[1]   # width height here is some problem
        if h > w: # find long side is width
            t = w
            w = h
            h = t
        #print('W',h,w)
        left_x = x-w/2+1
        left_y = y-h/2+1
        
        # crop golden object
        s_x = int(left_x)
        s_y = int(left_y)
        e_x = int(s_x+w)
        e_y = int(s_y+h)
        
        crop_img = img_gray[s_y:e_y, s_x:e_x]
        #cv2.imshow('goldan_ori', crop_img)
        cp_h,cp_w = crop_img.shape[:2]
        print("goaldan image crop image size",(cp_w,cp_h))
        
        B = cv2.resize(crop_img, (int(cp_w*in_w/cp_w), int(cp_h*in_w/cp_w)), interpolation=cv2.INTER_LINEAR)            
        A = in_crop_img     
        cp_h,cp_w = B .shape[:2]
        #cv2.imshow('goldan_resize', B)
        if cp_h > in_h :
            edge = int((cp_h-in_h)/2)
            A = cv2.copyMakeBorder(A,edge,edge,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        else:
            edge = int((in_h-cp_h)/2)
            B = cv2.copyMakeBorder(B,edge,edge,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        
        A_img, B_img0 = get_compare_imgs(A,B)
        sse0 = np.square(np.subtract(A_img, B_img0)).mean() #sum of square error
        # 180 degree
#        h,w = B_img0.shape[:2]
#        M = cv2.getRotationMatrix2D((w/2, h/2), 180, 1)
#        B_img1 = cv2.warpAffine(B_img0, M, (w, h))
        B_img1 = cv2.flip(B_img0, 1)
        sse1 = np.square(np.subtract(A_img, B_img1)).mean()
        #cv2.imshow('ori', A_img)
        #cv2.imshow('goldan1', B_img0)
        #cv2.imshow('goldan2', B_img1)
        #cv2.waitKey(0)            
        print('idx:%02d,SSE0:%d,SSE2:%d' %(idx,sse0,sse1))
        MSE_array[idx,0] = sse0
        MSE_array[idx,1] = sse1
        fo_2.write( "candidate = "+ str(values) +", MSE1: "+ str(sse0) +", MSE2: "+ str(sse1)+";\n")
        MSE_idx_name.append(values)
        idx += 1
    #fo_2.write( "**********************************************************\n")
    
        
    similiar_idx = int(MSE_array.argmin()/2)
    print(similiar_idx)
    #find_idx = MSE_array.argmin()
    file_name = goldenpath + 'hor180_' + str(MSE_idx_name[similiar_idx])
    ans_img = cv2.imread(file_name)

    fo_2.write( "Shape compare ans : " + str(os.path.basename(file_name)) + "\n")
    fo_2.write(str(time.time() - process_start_time)+ "\n")
    fo_2.write( "**********************************************************\n")
    fo_2.close()
    body, ext = os.path.splitext(os.path.basename(test_path))
    cv2.imwrite(Results_DIR+body+"_ans.png", ans_img)
    return ans_img


#====== Load feature for goldan model ======#
model_features = open("../37class_img/RESULT/modelFeatures.txt", "r")
idx = 0
for line in model_features.readlines():
    line = ''.join(line).strip('\n')    
    features = line.split(',')
    fixed_dict_37class[idx] = features[0]
    
    #moment
    features_set_moment = ()
    for i in range(len(features)):
        if(i <= 3):
            continue
        features_set_moment = features_set_moment + (float(features[i]),)
    
    features_set = (float(features[1]), float(features[2]), int(features[3]), [ np.array(features_set_moment).flatten() ])
    list_37class.append(features_set)
    idx +=1
    
print(fixed_dict_37class) 
print(list_37class)  
#fo = open( Result_txt_path, "w" )
#fo.write( "Total candicates is " + str(len(fixed_dict_37class)) + "\n")
#fo.close()


#====== Compare features for test image and goaldan image ======#
MODEL_DIR = "./Src/"  # 
TESTMODEL_DIR = "./BW/"
fo = open( Result_txt_path, "w" )
fo.close()
fo_2 = open( Result2_txt_path, "w" )
fo_2.close()

for modelname in os.listdir( MODEL_DIR ):
    dict_37class = fixed_dict_37class.copy() # get goaldan model
    
    fo = open( Result_txt_path, "a" )
    global process_start_time
    process_start_time = time.time()
    fo.write( "Total candicates is " + str(len(dict_37class)) + "\n")
    
    # Check test image is png file
    if modelname.find('.jpg') == -1:
        print("Log: "+modelname+" not jpg")
        continue
    print("Log: "+modelname)
    
    modelpath = MODEL_DIR + modelname
    ori_img = io.imread( modelpath )
    
    img_model = PreProcessTestImg(ori_img) 
    io.imsave( TESTMODEL_DIR+modelname, img_model )
    
    #moment
    #img_moment = cv2.imread(img_model)
    #calculate zernike moment, Second parameter must be false
    img_moment = img_model.astype('uint8')
    shapeFeatures = describe_shapes(img_moment, False)
    #print(shapeFeatures)
    
    
    #=======  extract Test image feature vector =======#
    label_img = label( img_model )
    regions = regionprops( label_img )
    test_name = modelname
   
    # inter hole
    cnts = label_img.astype('uint8')
    imgbinary ,contours,hierarchy = cv2.findContours(cnts,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # Solidity,axis ratio, moment
    test_feature = (regions[0].solidity, (regions[0].major_axis_length/regions[0].minor_axis_length), int(len(contours)-1), shapeFeatures)
    fo.write( "FileName = "+ test_name +", features: "+ str(test_feature) +"\n")
    
    #======= Start using Compare feature (test and goldan) ######
    tmp_dict = {}
    '''
    ####### axis ratio #######
    for key,values in dict_37class.items():
        model_feature = list_37class[int(key)]
        error = abs(model_feature[0]-test_feature[0])  # absolute error
        tmp_dict[key] = error
    
    # sort and filter by error
    delet_num = 0
    for key,values in sorted(tmp_dict.items(), key=lambda item:item[1], reverse=True):
        print("# ", values , dict_37class[key])
        del dict_37class[key]
        if(delet_num+1>=(len(dict_37class))):  #  50% out
            break
        else:
            delet_num+=1  		
    tmp_dict.clear()
    fo.write( "After axisRatio candicates is " + str(len(dict_37class)) + "\n")
    '''
    ####### inter hole #######
    for key,values in dict_37class.items():
        model_feature = list_37class[int(key)]
        error = abs(model_feature[2]-test_feature[2])  # absolute error
        tmp_dict[key] = error
    
    # sort and filter by error
    delet_num = 0
    for key,values in sorted(tmp_dict.items(), key=lambda item:item[1], reverse=True):
        print("## ", values, dict_37class[key])    
        if(values<=3):  # delete error more than 3
            break
        else:
            del dict_37class[key]
    tmp_dict.clear()
    fo.write( "After internal holes candicates is " + str(len(dict_37class)) + "\n")    

    
    ####### Solidity #######
    for key,values in dict_37class.items():
        model_feature = list_37class[int(key)]
        error = abs(model_feature[1]-test_feature[1])  # absolute error
        tmp_dict[key] = error
    
    # sort and filter by error
    delet_num = len(dict_37class)*0.75
    delet_count = 0
    for key,values in sorted(tmp_dict.items(), key=lambda item:item[1], reverse=True):
        print("## ", values, dict_37class[key])    
        if(delet_count+1>=delet_num):  # 75% out
            break
        else:
            del dict_37class[key]
            delet_count+=1
    tmp_dict.clear()
    fo.write( "After solidity candicates is " + str(len(dict_37class)) + "\n")
    
    ####### Moment #######
    for key,values in dict_37class.items():
        model_feature = list_37class[int(key)]
        # Calculate Euclidean Distance
        D = dist.cdist(model_feature[3], test_feature[3])
        error = D[0] # error of euclidean distance
        tmp_dict[key] = error
        
    # sort and filter by error
    delet_num = len(dict_37class)*0.6
    delet_count = 0
    for key,values in sorted(tmp_dict.items(), key=lambda item:item[1], reverse=True):
        print("## ", values, dict_37class[key])
        if(delet_count+1>=delet_num):
            break
        else:
            del dict_37class[key]
            delet_count+=1
    tmp_dict.clear()
    fo.write( "After moment candicates is " + str(len(dict_37class)) + "\n")
    
    
        
    ####### Output candidates item #######    
    for key,values in dict_37class.items():    
        #fo.write( "candidate = "+ str(values) +", features: "+ str(list_37class[int(key)]) +";\n")
        if(str(values) == test_name):
            match_number += 1
        fo.write( "candidate = "+ str(values) +", features: "+ str(list_37class[int(key)]) +";\n")
    
    final_candidate_number = len(dict_37class.items())
    match_sum += 1
    
    fo.write( "**********************************************************\n")
    fo.close()
    
    '''
    ####### shape compare part #######
    output_img = feature_compare(modelpath, dict_37class)
    body, ext = os.path.splitext(modelname)    
    cv2.imwrite(Results_DIR+body+".png", ori_img)
    '''

print ("final candidate number: " + str(final_candidate_number))
print ("accuracy: " + str(match_number/match_sum))  
fo.close()

	




