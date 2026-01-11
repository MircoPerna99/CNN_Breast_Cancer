import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds

df = pd.read_csv("DataSet.csv")

batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  "DataSet.csv",
  validation_split=0.2,
  subset="training",
  seed=123,
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "DataSet.csv",
  validation_split=0.2,
  subset="validation",
  seed=123,
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)