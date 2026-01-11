import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
df = pd.read_csv("/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/dataset.txt",sep=r'\s+')

if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use')

if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/Bening')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/Bening')
    
if(not os.path.isdir('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/Malignant')):
    os.makedirs('/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/Malignant')

data = []
for index, row in df.iterrows():
    file_name = "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_images/"+row['image_id']+".pgm" 
    result_to_save =  1 if row["result"] == 'M' else 0
    
    dir_selected = "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/Malignant/" if row["result"] == 'M' else "/Users/mircoperna/Documents/Universita/Magistrale/DeepLearning/Code/CNN_Breast_Cancer/Data_To_Use/Bening/"
    
    img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
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
    
    flip = cv2.flip(img, 1)
    cv2.imwrite(dir_selected+row['image_id']+"_flip.pgm", flip)
    data.append([row["image_id"]+"_flip.pgm", result_to_save])


# df_new = pd.DataFrame(data=data, columns=["image_id","result"])

# df_new.to_csv("DataSet.csv", index=False)