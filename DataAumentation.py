import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
df = pd.read_csv("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/dataset.txt",sep=r'\s+')

if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use')


data = []
for index, row in df.iterrows():
    file_name = "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/"+row['image_id']+".pgm" 
    
    img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/"+row['image_id']+".pgm", img)
    data.append([row["image_id"]+".pgm", row["result"]])
    
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
    cv2.imwrite("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/"+row['image_id']+"_90.pgm", rotated_90)
    data.append([row["image_id"]+"_90.pgm", row["result"]])
    
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/"+row['image_id']+"_180.pgm", rotated_180)
    data.append([row["image_id"]+"_180.pgm", row["result"]])
    
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/"+row['image_id']+"_270.pgm", rotated_270)
    data.append([row["image_id"]+"_270.pgm", row["result"]])
    
    flip = cv2.flip(img, 1)
    cv2.imwrite("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/"+row['image_id']+"_flip.pgm", flip)
    data.append([row["image_id"]+"_flip.pgm", row["result"]])


df_new = pd.DataFrame(data=data, columns=["image_id","result"])

df_new.to_csv("DataSet.csv", index=False)