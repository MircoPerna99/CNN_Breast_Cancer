import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from enum import Enum
from sklearn.model_selection import train_test_split

class DatasetTypes(Enum):
    Standard = 1
    Standard_PreTraining = 2
    Component = 3
    Component_PreTraining = 4
    Component_PreTraining_Flip = 5
    

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

def define_folder(dataset_type:DatasetTypes):
    if(dataset_type == DatasetTypes.Standard):
        return "DataSetStandard"
    if(dataset_type == DatasetTypes.Standard_PreTraining):
        return "DataSetStandard_Pretraining"
    if(dataset_type == DatasetTypes.Component):
        return "DataSetComponent"
    if(dataset_type == DatasetTypes.Component_PreTraining):
        return "DataSetComponent_Pretraining"
    if(dataset_type == DatasetTypes.Component_PreTraining_Flip):
        return "DataSetComponent_PreTraining_Flip"

def check_folders(folder):
    if(not os.path.isdir(folder)):
        os.makedirs(folder)
        return False
    return True

def create_dataset(dataset_type:DatasetTypes,test_size = 0.3):
    df = pd.read_csv("Data_images/dataset_component.txt",sep=r'\s+')
    X = list(df['image_id'])
    X_train, X_test = train_test_split(X, test_size=test_size)
    
    dir_name = define_folder(dataset_type)
    path_benign_training = f'{dir_name}/Training/Benign'
    path_malignant_training = f'{dir_name}/Training/Malignant'
    path_benign_test = f'{dir_name}/Test/Benign'
    path_malignant_test = f'{dir_name}/Test/Malignant'
    
    if(check_folders(dir_name)):
        return

    check_folders(path_benign_training)
    check_folders(path_malignant_training)
    check_folders(path_benign_test)
    check_folders(path_malignant_test)

    for index, row in df.iterrows():
        file_name = "Data_images/"+row['image_id']+".pgm" 
        
        img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)        
        
        dir_selected = path_benign_training if row["result"] == 'M' else path_malignant_training
        dir_selected = dir_selected + "/"
                
        if(dataset_type == DatasetTypes.Component or dataset_type == DatasetTypes.Component_PreTraining):
            img = take_component(img, row["x"], row["y"], row["radius"])
        
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        
        if(dataset_type == DatasetTypes.Component_PreTraining or dataset_type == DatasetTypes.Standard_PreTraining):
            img = pre_process(img)
        
        if(row['image_id'] in X_test):
            dir_selected_test = path_malignant_test if row["result"] == 'M' else path_benign_test
            cv2.imwrite(dir_selected_test+ "/"+row['image_id']+".pgm", img)
            continue
        
        cv2.imwrite(dir_selected+row['image_id']+".pgm", img)
        
        rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
        cv2.imwrite(dir_selected+row['image_id']+"_90.pgm", rotated_90)
        
        rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(dir_selected+row['image_id']+"_180.pgm", rotated_180)
        
        rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(dir_selected+row['image_id']+"_270.pgm", rotated_270)
        
        flip_h = cv2.flip(img, 1)
        cv2.imwrite(dir_selected+row['image_id']+"_flip_h.pgm", flip_h)
        
        if(dataset_type == DatasetTypes.Component_PreTraining_Flip):
            rotated_90 = cv2.rotate(flip_h, cv2.ROTATE_90_CLOCKWISE) 
            cv2.imwrite(dir_selected+row['image_id']+"_flip_h_90.pgm", rotated_90)
            
            rotated_180 = cv2.rotate(flip_h, cv2.ROTATE_180)
            cv2.imwrite(dir_selected+row['image_id']+"_flip_h_180.pgm", rotated_180)
            
            rotated_270 = cv2.rotate(flip_h, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(dir_selected+row['image_id']+"_flip_h_270.pgm", rotated_270)
        
        flip_v = cv2.flip(img, 0)
        cv2.imwrite(dir_selected+row['image_id']+"_flip_v.pgm", flip_v)
        if(dataset_type == DatasetTypes.Component_PreTraining_Flip):
            rotated_90 = cv2.rotate(flip_v, cv2.ROTATE_90_CLOCKWISE) 
            cv2.imwrite(dir_selected+row['image_id']+"_flip_v_90.pgm", rotated_90)
            
            rotated_180 = cv2.rotate(flip_v, cv2.ROTATE_180)
            cv2.imwrite(dir_selected+row['image_id']+"_flip_v_180.pgm", rotated_180)
            
            rotated_270 = cv2.rotate(flip_v, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(dir_selected+row['image_id']+"_flip_v_270.pgm", rotated_270)