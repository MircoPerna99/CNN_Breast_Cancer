import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def pre_process(img):
    bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    largest_cc = np.zeros_like(closing)
    largest_cc[labels == largest_label] = 255
    return cv2.bitwise_and(img, img, mask=largest_cc)

df = pd.read_csv("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/dataset.txt",sep=r'\s+')

if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered')

if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered/Benign')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered/Benign')
    
if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered/Malignant')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered/Malignant')

data = []
for index, row in df.iterrows():
    file_name = "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/"+row['image_id']+".pgm" 
    result_to_save =  1 if row["result"] == 'M' else 0
    
    dir_selected = "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered/Malignant/" if row["result"] == 'M' else "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Filtered/Benign/"
    
    img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
    img = pre_process(img)
    cv2.imwrite(dir_selected+row['image_id']+".png", img)
    data.append([row["image_id"]+".pgm", result_to_save])
    
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
    cv2.imwrite(dir_selected+row['image_id']+"_90.png", rotated_90)
    data.append([row["image_id"]+"_90.pgm", result_to_save])
    
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(dir_selected+row['image_id']+"_180.png", rotated_180)
    data.append([row["image_id"]+"_180.pgm", result_to_save])
    
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(dir_selected+row['image_id']+"_270.png", rotated_270)
    data.append([row["image_id"]+"_270.pgm", result_to_save])
    
    flip = cv2.flip(img, 1)
    cv2.imwrite(dir_selected+row['image_id']+"_flip.png", flip)
    data.append([row["image_id"]+"_flip.pgm", result_to_save])


# df_new = pd.DataFrame(data=data, columns=["image_id","result"])

# df_new.to_csv("DataSet.csv", index=False)