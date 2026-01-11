import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
import tensorflow as tf
import keras 
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
    print("GPU non trovata. Controlla l'installazione di tensorflow-metal.")

# tf.debugging.set_log_device_placement(True)
def load_data():
  images = []
  labels = []
  class_names = ['Benign', 'Malignant']
  path_base = "Data_To_Use_Filtered/"

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


#Define model
model = Sequential(
  [Input(shape=(224, 224, 1)),
   Conv2D(filters=10, kernel_size=(5,5),input_shape=(224, 224, 1)),
   MaxPooling2D(pool_size=(2, 2)),
   Conv2D(filters=16, kernel_size=(3,2)),
   MaxPooling2D(pool_size=(2, 2)),
   Conv2D(filters=16, kernel_size=(3,2)),
   MaxPooling2D(pool_size=(2, 1)),
   Conv2D(filters=80, kernel_size=(3,1)),
   Conv2D(filters=80, kernel_size=(3,1)),
   MaxPooling2D(pool_size=(2, 1)),
   Flatten(),
   Dense(16, activation='relu'),
   Dense(8, activation='relu'),
   Dense(16, activation='relu'),
   Dense(1, activation='sigmoid')
   ]
)

# optimizer = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,epochs=50)

y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

accuracy = accuracy_score(y_test, y_hat)
print(accuracy)


