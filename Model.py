import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D,Input,ZeroPadding2D,RandomFlip,RandomRotation,RandomContrast
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold
from PreProcessing import DatasetTypes,define_folder,create_dataset
from enum import Enum

class ModelTypes(Enum):
    Standard = 1
    Standard_With_Augmentation = 2

def create_model(kernel_size, learning_rate, momentum,with_decay):
  model = Sequential(
    [Input(shape=(224, 224, 1)),
    ZeroPadding2D(padding=(3, 3)),
    Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), activation='relu'),
    Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    ZeroPadding2D(padding=(2, 2)),
    Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    ZeroPadding2D(padding=(1, 1)),
    Conv2D(filters=80, kernel_size=(kernel_size,kernel_size), activation='relu'),
    Conv2D(filters=80, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
    ]
  )
  
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum,)
  
  if(with_decay):
    lr_schedule = ExponentialDecay(learning_rate,decay_steps=500,decay_rate=0.2)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,momentum=momentum)

  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  
  return model

def create_model_with_augmentation(kernel_size,learning_rate, momentum, with_decay):
  model = Sequential(
    [Input(shape=(224, 224, 1)),
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.1),
    ZeroPadding2D(padding=(3, 3)),
    Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), activation='relu'),
    Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    ZeroPadding2D(padding=(2, 2)),
    Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    ZeroPadding2D(padding=(1, 1)),
    Conv2D(filters=80, kernel_size=(kernel_size,kernel_size), activation='relu'),
    Conv2D(filters=80, kernel_size=(kernel_size,kernel_size), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
    ]
  )
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum,)
  
  if(with_decay):
    lr_schedule = ExponentialDecay(learning_rate,decay_steps=500,decay_rate=0.2)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,momentum=momentum)

  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
  return model

def take_model(kernel_size,learning_rate, momentum,type:ModelTypes,with_decay):
  if(type == ModelTypes.Standard):
    return create_model(kernel_size,learning_rate, momentum,with_decay)
  if(type == ModelTypes.Standard_With_Augmentation):
    return create_model_with_augmentation(kernel_size,learning_rate, momentum,with_decay)


def apply_cross_validation(X, y, kernel_size,learning_rate, momentum,type_model:ModelTypes,epochs = 50,batch_size=5, k=5, with_decay = False):
  kf = KFold(n_splits=k, shuffle=True)
  accuracy_per_fold = []
  
  for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f'Fold {fold+1}')
    K.clear_session()
    
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    model = take_model(kernel_size,learning_rate, momentum,type_model,with_decay)
    model.fit(X_train_fold, y_train_fold,epochs=epochs,batch_size=batch_size)

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
  return X,y




    

