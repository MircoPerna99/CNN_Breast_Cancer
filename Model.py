import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2



def load_data():
  images = []
  labels = []
  class_names = ['Benign', 'Malignant']
  path_base = "Data_To_Use/"

  for label, class_name in enumerate(class_names):
    class_dir = os.path.join(path_base, class_name)
    print(f"Load class {class_name}...")
    for img_name in os.listdir(class_dir):
      img_path = os.path.join(class_dir, img_name)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      if img is not None:
        images.append(img)
        labels.append(label)
    print(f"Load class {class_name} completed")
        
  X = np.array(images, dtype='float32').reshape(-1, 224, 224, 1) / 255.0
  y = np.array(labels, dtype='int32')
  
  
  return X, y
  
X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)




