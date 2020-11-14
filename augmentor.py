#!/usr/bin/env python
# coding: utf-8

# In[18]:


import Augmentor
import glob
import cv2  # Import OpenCV library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

call = glob.glob('E:/CNN/FAUZI/*.jpg')


# In[19]:


def potongWajahTerusSimpan(ini):

    # Buat pendeteksi wajah
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #cap = cv2.VideoCapture(0)
    img = cv2.imread(show)  # Baca gambar dengan OpenCV

    h, w = img.shape[:2]

    # Menentukan Ukuran dan Resizing Image
    new_h, new_w = int(h/4), int(w/4)
    resizeImg = cv2.resize(img, (new_w, new_h))

    # Proses identifikasi dilakukan pada mode grayscale (hitam/putih) / abu-abu
    # Convert dari gambar warna ke grayscale
    grayImg = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2GRAY)

    # Mengimplementasikan face detection
    faces = faceCascade.detectMultiScale(
        grayImg,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Menggambar kotak pada bidang wajah yang terdeteksi. Dan warnai photo
    x = 0
    y = 0
    w = 0
    h = 0
    for (x, y, w, h) in faces:
        pass
        cv2.rectangle(resizeImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # plt.axis("off")
    # plt.imshow(resizeImg)
    # plt.show()
    roi = resizeImg[y:y+h, x:x+w]
    cv2.imwrite(ini, roi)


# In[21]:


for show in call:
    print(ini)
    potongWajahTerusSimpan(ini)


# In[ ]:


#foto = "E:/CNN/Data/60%~40%/test/16102167/16102167 (80).jpg"
print(kucing)


# In[5]:


roi = resizeImg[y:y+h, x:x+w]
cv2.imwrite("telor.jpg", roi)


# In[ ]:


get_ipython().system('pip install Augmentor')


# In[25]:


path_to_data = "E:/CNN/DATA CITRA/BARU/TEST/FAUZI"

# Membuat Pipeline
p = Augmentor.Pipeline(path_to_data)


# In[26]:


# Add some operations to an existing pipeline.
# Untuk citra menggunakan Random Distorsi
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)

# First, we add a horizontal flip operation to the pipeline:
# Untuk citra menggunakan flip kanan kiri
p.flip_left_right(probability=0.4)

# Now we add a vertical flip operation to the pipeline:
# p.flip_top_bottom(probability=0.8)

# Add a rotate90 operation to the pipeline:
# Untuk citra wajah menggunakan rotasi
p.rotate(probability=0.7, max_left_rotation=13, max_right_rotation=13)
# Untuk citra menggunakan zoom(perbesar)
p.zoom_random(probability=0.5, percentage_area=0.9)


# In[27]:


# Seringkali berguna untuk menggunakan notasi ilmiah untuk menentukan
# Jumlah besar dengan nol tambahan.
num_of_samples = int(430)

# menjalankan pipeline
p.sample(num_of_samples)
