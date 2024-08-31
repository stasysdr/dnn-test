import tensorflow as tf
from tensorflow.keras import layers, models, utils, Sequential
import pathlib

dataset_url = "https://github.com/stasysdr/dnn-test/raw/main/digits.tgz"
data_dir = utils.get_file('digits.tar', origin=dataset_url, extract=True,
                          cache_dir="/content")
data_dir = pathlib.Path(data_dir).with_suffix('')

batch_size = 32
img_height, img_width = 18, 20

training_data_set, validation_data_set = utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="both",
  seed=123,
  color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)

model = Sequential([
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, activation='relu'),
  layers.Conv2D(32, 3, activation='relu'),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(10)
])

model.compile(optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(
  training_data_set,
  validation_data=validation_data_set,
  epochs=15
)
