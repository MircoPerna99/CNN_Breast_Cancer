import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from enum import Enum

class DatasetTypes(Enum):
    Standard = 1
    Standard_PreTraining = 2
    Component = 3
    Component_PreTraining = 4
    

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
    radius = radius+100
    y = h - 1 - y
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

def check_folders(folder):
    if(not os.path.isdir(folder)):
        os.makedirs(folder)
        return False
    return True

def create_dataset(dataset_type:DatasetTypes):
    df = pd.read_csv("Data_images/dataset_component.txt",sep=r'\s+')

    dir_name = define_folder(dataset_type)
    path_benign = f'{dir_name}/Benign'
    path_malignant = f'{dir_name}/Malignant'
    
    if(check_folders(dir_name)):
        return

    check_folders(path_benign)
    check_folders(path_malignant)

    for index, row in df.iterrows():
        file_name = "Data_images/"+row['image_id']+".pgm" 
        dir_selected = path_malignant if row["result"] == 'M' else path_benign
        dir_selected = dir_selected + "/"
        
        img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
        
        if(dataset_type == DatasetTypes.Component or dataset_type == DatasetTypes.Component_PreTraining):
            img = take_component(img, row["x"], row["y"], row["radius"])
        
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        
        if(dataset_type == DatasetTypes.Component_PreTraining or dataset_type == DatasetTypes.Standard_PreTraining):
            img = pre_process(img)
        
        cv2.imwrite(dir_selected+row['image_id']+".pgm", img)
        