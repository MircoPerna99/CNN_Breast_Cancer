import tensorflow as tf
import numpy as np
import pandas as pd 
from PreProcessing import DatasetTypes
from Model import ModelTypes, load_data,apply_cross_validation
from sklearn.model_selection import train_test_split
#MAIN
APPLY_DECAY = False
devices = tf.config.list_physical_devices('GPU')

if len(devices) > 0:
    details = tf.config.experimental.get_device_details(devices[0])
    tf.config.set_visible_devices(devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
else:
    print("GPU has not been found.")
    
cross_validation_results=[]
learning_rates = [0.01,0.001]
momentums = [0.9,0.5]
kernel_sizes = [2,3,5]

#TUNING
for dataset_type in DatasetTypes:
  X, y= load_data(dataset_type)
  X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
  for learning_rate in learning_rates:
    for momentum in momentums:
      for kernel_size in kernel_sizes:
        for model_type in ModelTypes:
          details = []
          details.append(model_type)
          details.append(dataset_type)
          details.append(learning_rate)
          details.append(momentum)
          details.append(kernel_size)
          results = apply_cross_validation(X, y,kernel_size,learning_rate, momentum,model_type,k=5,with_decay=APPLY_DECAY)
          details.append(np.mean(results))
          details.append(np.std(results))
          cross_validation_results.append(details)
         
df = pd.DataFrame(cross_validation_results, columns = ['model_type', 'dataset_type', 'learning_rate', 'momentum', 'kernel_size', 'accuracy_mean', 'standard_deviation']) 

#SAVE RESULTS
if(APPLY_DECAY):
  df.to_csv('result_with_decay.csv', index=False)  
else:
  df.to_csv('result_standard.csv', index=False)  