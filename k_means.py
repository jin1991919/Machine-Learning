import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.datasets.samples_generator import make_blobs
x,y_true=make_blobs(n_samples=300,centers=4,cluster_std=1.0,random_state=10)
#plt.scatter(x[:,0],x[:,1],s=100)

import cv2
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags=cv2.KMEANS_RANDOM_CENTERS
import numpy as np
compactness,labels,centers=cv2.kmeans(x.astype(np.float32),4,None,criteria,10,flags)
plt.scatter(x[:, 0], x[:, 1], c=labels[:,0], s=50, cmap='viridis')
#plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5);
plt.show()