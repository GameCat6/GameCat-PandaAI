import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16
# from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os
train_dir = 'animals/animals'

test_dir = 'Test'

img_size = 150

train_dataset = image_dataset_from_directory(train_dir,
                                             subset='training',
                                             seed=123,
                                             validation_split=0.2,
                                             batch_size=100,
                                             image_size=(img_size,img_size))
print('Cat')
validation_dataset = image_dataset_from_directory(train_dir,
                                             subset='validation',
                                             seed=123,
                                             validation_split=0.2,
                                             batch_size=100,
                                             image_size=(img_size,img_size))
print('Cat')

vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
vgg16_net.trainable = False

model = Sequential()

model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
history = model.fit(train_dataset, 
                    validation_data=validation_dataset,
                    epochs=3,
                    verbose=1)

model.save("SavedAI/my_model")