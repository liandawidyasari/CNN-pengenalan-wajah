#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import confusion_matrix

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])


# In[1]:


#REGRESI LASSO
#PREDIKSI LAYER PADA CNN

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit([[32, 32, 64, 0], [32, 32, 32, 0], [128, 64, 32, 32], [32, 32, 64, 128], [32, 64, 128, 0], [128, 64, 32, 0], [16, 0, 0, 0] ], [0.48, 0.44, 0.46, 0.45, 0.51, 0.48, 0.42])

print(clf.coef_)

print(clf.intercept_)


# In[ ]:


clf.predict([[256, 512, 1024, 0]])


# In[2]:


f = open("hasil_lianda.txt", "w")

max = 0
max_tulis = ""

for i in range(7):
  for j in range(7):
    for k in range(7):
      for l in range(7):
        pred = clf.predict([[2**i *16, 2**j*16, 2**k*16, 2**l*16]])
        tulis = str(2**i*16) + "\t" + str(2**j*16) + "\t" + str(2**k*16) + "\t" + str(2**l*16) + "\t" + str(pred)
        print(tulis)
        f.write(tulis + "\n")
        if pred > max:
          max = pred
          max_tulis = tulis

print('MAX:', max_tulis)

f.close()


# In[ ]:




