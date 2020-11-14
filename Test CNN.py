#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.metrics import categorical_accuracy, categorical_crossentropy

# Initialising the CNN
classifier = Sequential()
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

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = categorical_crossentropy, metrics = [categorical_accuracy])

classifier.load_weights("21 CLASS M 2.2 NONDIS 35.h5")

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('E:/CNN/DATA CITRA/BARU/TRAIN', target_size = (80, 80), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('E:/CNN/DATA CITRA/BARU/TEST', target_size = (80, 80), batch_size = 32, class_mode = 'categorical')


# In[5]:


def switch_dict_key_values(this_dict):
    return dict((v,k) for k,v in this_dict.items())

training_names = switch_dict_key_values(training_set.class_indices)
print(training_names)


# In[6]:


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import glob

kochenk = glob.glob('E:/CNN/DATA CITRA/BARU/TEST/YUSUF/*')
hasil = 0
for kucing in kochenk:
    test_image = image.load_img(kucing, target_size = (80, 80))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    #training_set.class_indices
    #/print(training_names[result[0][0]])
    
    hasil += int(training_names[result[0]] == "YUSUF")
    #if result[0][0] == 1:
    #    prediction = 'dog'
    #else:
    #    prediction = 'cat'
print(hasil)


# In[9]:


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import glob

c = glob.glob('D:/BACKUP PIC/LIANDA/*')
hasil = 0
for kucing in c:
    test_image = image.load_img(kucing, target_size = (80, 80))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    #training_set.class_indices
    #/print(training_names[result[0][0]])
    print(training_names[result[0]])
    #hasil += int(training_names[result[0]] == "LIANDA")
    
#print(hasil)


# In[7]:


import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import glob
citra = glob.glob('E:/CNN/DATA CITRA/BARU/TEST/*')
angka = 0
for ini in citra:
    adapicnew = glob.glob(ini + '/*.jpg')
    hasil = 0
    for barudah in adapicnew:
        test_image = image.load_img(barudah, target_size = (80, 80))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict_classes(test_image)


        hasil += int(result[0] == angka)
    print(hasil)
    angka += 1


# In[8]:


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import glob

kochenk = glob.glob('E:/CNN/DATA CITRA/BARU/TEST/WIWI/*.jpg')
hasil = 0
for kucing in kochenk:
    test_image = image.load_img(kucing, target_size = (80, 80))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict_classes(test_image)
    #training_set.class_indices
    print(training_names[result[0]], kucing)

    #hasil += int(training_names[result[0]] == "OLVY")
print(hasil)


# In[19]:


classifier.predict_classes(test_image)

