import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D,Input,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold
from DataAugmentation import DatasetTypes,define_folder,create_dataset

def create_model(seeSummary = False):
  model = Sequential(
    [Input(shape=(224, 224, 1)),
    Conv2D(filters=4, kernel_size=(3,3)),
    Conv2D(filters=4, kernel_size=(3,3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(3,2)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(3,2)),
    MaxPooling2D(pool_size=(2, 1)),
    Conv2D(filters=80, kernel_size=(3,1)),
    Conv2D(filters=80, kernel_size=(3,1)),
    MaxPooling2D(pool_size=(2, 1)),
    Flatten(),
    Dense(1, activation='sigmoid')
    ]
  )
  
  optimizer = tf.keras.optimizers.SGD(
      learning_rate=0.01,
      momentum=0.7,
  )

  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  
  if(seeSummary):
    model.summary()
    
  return model

def apply_cross_validation(X, y, k=5):
  kf = KFold(n_splits=k, shuffle=True)
  accuracy_per_fold = []
  
  for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f'Fold {fold+1}')
    K.clear_session()
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    model = create_model()
    
    model.fit(X_train_fold, y_train_fold,epochs=10,batch_size=10)

    y_hat = model(X_val_fold, training=False)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]

    accuracy = accuracy_score(y_val_fold, y_hat)
    print(accuracy)
    accuracy_per_fold.append(accuracy)
    
  return accuracy_per_fold
  
def load_data(dataset_type:DatasetTypes):
  images = []
  labels = []
  class_names = ['Benign', 'Malignant']
  create_dataset(dataset_type)
  folder = define_folder(dataset_type)
  path_base = f"{folder}/"

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
        
  X = np.array(images, dtype='float32') / 255.0
  y = np.array(labels, dtype='int32')
  
  return X, y


devices = tf.config.list_physical_devices('GPU')

if len(devices) > 0:
    details = tf.config.experimental.get_device_details(devices[0])
    tf.config.set_visible_devices(devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    print("GPU non trovata.")

cross_validation_results=[]

for type in DatasetTypes:
  X, y = load_data(type)
  details = []
  details.append(type)
  details.append(apply_cross_validation(X, y, k=5))
  details.append(apply_cross_validation(X, y, k=10))
  cross_validation_results.append(details)

for result in cross_validation_results:
  print(f"Result for dataset {result[0]}")
  print(f"Result cross-validation with k=5")
  print(f"Mean {np.mean(result[1])}")
  print(f"Standard deviation {np.std(result[1])}")
  print(f"Result cross-validation with k=10")
  print(f"Mean {np.mean(result[2])}")
  print(f"Standard deviation {np.std(result[2])}")






