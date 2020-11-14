#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.metrics import categorical_accuracy, categorical_crossentropy


# In[7]:


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32 itu jumlah filter/baja, ukurannya harus ganjil 1~9. input_shape harus sama ukurannya
# Step 2 - Pooling
classifier.add(Conv2D(32, (5, 5), input_shape = (80, 80, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 21, activation = 'softmax'))
#units = 1000, softmax, binary->caregory, accuracy->categorical accuracy

classifier.summary()


# In[9]:


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = categorical_crossentropy, metrics = [categorical_accuracy])


# In[ ]:


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('E:/CNN/DATA CITRA/BARU/TRAIN', target_size = (80, 80), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('E:/CNN/DATA CITRA/BARU/TEST', target_size = (80, 80), batch_size = 32, class_mode = 'categorical')

for i in range(100):
    classifier.fit_generator(training_set, steps_per_epoch = 1600, epochs = 1, validation_data = test_set, validation_steps = 2000)
    classifier.save_weights("21 CLASS M 2.2 NONDIS " + str(i) + ".h5") #Menyimpan data epoch per step


# In[ ]:


#Menyimpan training data epoch paling terakhir
classifier.save_weights('train18feb.h5')


# In[8]:


#Memanggil list nama class yang terdapat pada dataset
def switch_dict_key_values(this_dict):
    return dict((v,k) for k,v in this_dict.items())

training_names = switch_dict_key_values(training_set.class_indices)
print(training_names)


import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import glob
c = glob.glob('E:/CNN/DATA CITRA/BARU/TEST/*')
angka = 0
for i in c:
    a = glob.glob(i + '/*.jpg')
    hasil = 0
    for b in a:
        test_image = image.load_img(b, target_size = (80, 80))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict_classes(test_image)


        hasil += int(result[0] == angka)
    print(hasil)
    angka += 1

