from keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image
data_augment = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
path = 'C:/Python Programs/PandaCat1/animals/animals/panda/'
for file in os.listdir(path):
    i = 0
    path1 = path+ file
    single_img = image.load_img(path1)
    image_array = image.img_to_array(single_img)
    image_array = image_array.reshape((1,) + image_array.shape)
    for batch in data_augment.flow(image_array, save_to_dir = path, save_prefix = 'panda+', save_format='jpg'):
        i += 1
        if i > 2:
            break