import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import os
def pred1(path,model):
    img_size = 150
    for file in os.listdir(path):
        # c1.append(file)
        path1 =  path + file 
        try:
            img = image.load_img(path1, target_size=(img_size, img_size))    
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            # print(file,':')
            # print(np.argmax(model.predict(img_array,verbose=0)))
            path2 = '"' + 'C:\Python Programs\PandaCat1\Test\Test' + '\\' + file + '"'
            if np.argmax(model.predict(img_array,verbose=0)) == 0:
                respath = 'copy ' + path2 + ' "C:\Python Programs\PandaCat1\Result\Cats\"'
            else:
                if np.argmax(model.predict(img_array,verbose=0)) == 1:
                    respath = 'copy ' + path2 + ' "C:\Python Programs\PandaCat1\Result\Dogs\"'
                else:
                    respath = 'copy ' +  path2 + ' "C:\Python Programs\PandaCat1\Result\Pandas\"'
            os.system(respath)
            print(respath)
        except OSError:
    	    pass
model = keras.models.load_model('SavedAI/my_model')
test_dir = 'C:/Python Programs/PandaCat1/Test/Test/'
print(pred1(test_dir,model))
