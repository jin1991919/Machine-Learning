from sklearn.datasets import load_digits
digits=load_digits()
import cv2
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags=cv2.KMEANS_RANDOM_CENTERS
import numpy as np
digits.data=digits.data.astype(np.float32)
compactness,clusters,centers=cv2.kmeans(digits.data,10,None,criteria,10,flags)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
flg,ax=plt.subplots(2,5,figsize=(8,3))
centers=centers.reshape(10,8,8)
for axi,center in zip(ax.flat,centers):
    axi.set(xticks=[],yticks=[])
    axi.imshow(center,interpolation='nearest',cmap=plt.cm.binary)
plt.show()
from scipy.stats import mode
labels=np.zeros_like(clusters.ravel())
for i in range(10):
    mask=(clusters.ravel()==i)
    labels[mask]=mode(digits.target[mask])[0]
print(mask)
print(labels)
from sklearn.metrics import accuracy_score
print(accuracy_score(digits.target,labels))
