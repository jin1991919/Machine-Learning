import cv2
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
#The syntax to create an MLP in OpenCV is the same as for all the other classifiers:
mlp=cv2.ml.ANN_MLP_create()

n_input=2
n_hidden=8
n_output=2
mlp.setLayerSizes(np.array([n_input,n_hidden,n_output]))
#we will use a proper sigmoid function that squashes the input values into the range [0, 1]. We do this by choosing α = 2.5 and β = 1.0:
mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM,2.5,1.0)

import matplotlib.pyplot as plt
#we will choose backpropagation:

mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
term_mode=cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS
term_max_iter=300
term_eps=0.01
mlp.setTermCriteria((term_mode,term_max_iter,term_eps))
x,y=make_blobs(n_samples=100,centers=2,cluster_std=5.2,random_state=42)
y=2*y-1
plt.scatter(x[:,0],x[:,1],s=100,c=y)
plt.show()
mlp.train(x,cv2.ml.ROW_SAMPLE,y)
_,y_hat=mlp.predict(x)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_hat.round(),y))

