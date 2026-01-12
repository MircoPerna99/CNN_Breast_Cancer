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

def take_component(img, x, y, radius):
    h, w = img.shape
    radius = radius+50
    x1 = max(x - radius, 0)
    x2 = min(x + radius, w)
    y1 = max(y - radius, 0)
    y2 = min(y + radius, h)

    sub_img = img[y1:y2, x1:x2]
    return sub_img

df = pd.read_csv("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/dataset_component.txt",sep=r'\s+')

if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component')

if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component/Benign')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component/Benign')
    
if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component/Malignant')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component/Malignant')

data = []
for index, row in df.iterrows():
    file_name = "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/"+row['image_id']+".pgm" 
    result_to_save =  1 if row["result"] == 'M' else 0
    dir_selected = "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component/Malignant/" if row["result"] == 'M' else "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use_Component/Benign/"
    
    img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    img = take_component(img, row["x"], row["y"], row["radius"])
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
    img = pre_process(img)
    
    cv2.imwrite(dir_selected+row['image_id']+".pgm", img)
    data.append([row["image_id"]+".pgm", result_to_save])
    
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
    cv2.imwrite(dir_selected+row['image_id']+"_90.pgm", rotated_90)
    data.append([row["image_id"]+"_90.pgm", result_to_save])
    
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(dir_selected+row['image_id']+"_180.pgm", rotated_180)
    data.append([row["image_id"]+"_180.pgm", result_to_save])
    
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(dir_selected+row['image_id']+"_270.pgm", rotated_270)
    data.append([row["image_id"]+"_270.pgm", result_to_save])
    
    flip_h = cv2.flip(img, 1)
    cv2.imwrite(dir_selected+row['image_id']+"_flip_h.pgm", flip_h)
    data.append([row["image_id"]+"_flip_h.pgm", result_to_save])
    
    flip_v = cv2.flip(img, 0)
    cv2.imwrite(dir_selected+row['image_id']+"_flip_v.pgm", flip_v)
    data.append([row["image_id"]+"_flip_v.pgm", result_to_save])


# df_new = pd.DataFrame(data=data, columns=["image_id","result"])

# df_new.to_csv("DataSet.csv", index=False)