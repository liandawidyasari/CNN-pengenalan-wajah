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
# Step 1 - Convolution
classifier.add(Conv2D(128, (5, 5), input_shape = (80, 80, 3), activation = 'relu'))
# 32 itu jumlah filter/baja, ukurannya harus ganjil 1~9. input_shape harus sama ukurannya
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer 
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#3
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#4
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 21, activation = 'softmax'))
#units = 1000, softmax, binary->caregory, accuracy->categorical accuracy

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = categorical_crossentropy, metrics = [categorical_accuracy])

classifier.load_weights("TF_1_FAUZI1.h5")


# In[2]:


classifier.summary()


# In[3]:


#len(classifier.layers) 
#classifier.layers[0]
x = Sequential()
for i in range(10):
    if (i<8):
        classifier.layers[i].trainable = False
    x.add(classifier.layers[i])
    #if (i<8):
        #telor[-1].trainable = False
x.add(Dense(units = 26, activation = 'softmax'))
x.compile(optimizer = 'adam', loss = categorical_crossentropy, metrics = [categorical_accuracy])
x.summary()


# In[4]:


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('E:/CNN/PERCOBAAN/KODINGAN MODEL/TRANSFER LEARNING/1/TFL/Train', target_size = (80, 80), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('E:/CNN/PERCOBAAN/KODINGAN MODEL/TRANSFER LEARNING/1/TFL/Test', target_size = (80, 80), batch_size = 32, class_mode = 'categorical')


# In[5]:


def switch_dict_key_values(this_dict):
    return dict((v,k) for k,v in this_dict.items())

training_names = switch_dict_key_values(training_set.class_indices)
print(training_names)


# In[6]:


for i in range(25):
    telor.fit_generator(training_set, steps_per_epoch = 1600, epochs = 1, validation_data = test_set, validation_steps = 2000)
    telor.save_weights("TF_1_5ORNG" + str(i) + ".h5")


# In[7]:


#telor.save_weights('AKMAL TFL.h5')


# In[36]:


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import glob

kochenk = glob.glob('E:/CNN/PERCOBAAN/KODINGAN MODEL/TRANSFER LEARNING/1/TFL/Test/ROBI/*')
hasil = 0
for kucing in kochenk:
    test_image = image.load_img(kucing, target_size = (80, 80))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = telor.predict_classes(test_image)
    #training_set.class_indices
    #/print(training_names[result[0][0]])
    
    hasil += int(training_names[result[0]] == "ROBI")
    #if result[0][0] == 1:
    #    prediction = 'dog'
    #else:
    #    prediction = 'cat'
print(hasil)


# In[ ]:





# In[ ]:




