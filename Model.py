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

    y_hat = model.predict(X_val_fold)
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]

    accuracy = accuracy_score(y_val_fold, y_hat)
    print(accuracy)
    accuracy_per_fold.append(accuracy)
  
  average_accuracy = np.mean(accuracy_per_fold)
  print(f'The average accuracy is {average_accuracy}')


devices = tf.config.list_physical_devices('GPU')

if len(devices) > 0:
    print(f"Successo! Trovata GPU: {devices[0]}")
    # Verifica che sia effettivamente il supporto Metal
    details = tf.config.experimental.get_device_details(devices[0])
    print(f"Tipo dispositivo: {details.get('device_name')}")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tf.config.set_visible_devices(devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    print("GPU non trovata.")

def load_data():
  images = []
  labels = []
  class_names = ['Benign', 'Malignant']
  path_base = "Data_To_Use_Component/"

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
  
X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


apply_cross_validation(X_train, y_train)






