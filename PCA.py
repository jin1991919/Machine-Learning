import numpy as np
import matplotlib.pyplot as plt
mean=[20,20]
cov=[[5,0],[25,25]]
x,y=np.random.multivariate_normal(mean,cov,1000).T
X=np.vstack((x,y)).T
import cv2
mu,eig=cv2.PCACompute(X,np.array([]))
plt.plot(x,y,'o',zorder=1)
plt.quiver(mean[0],mean[1],eig[:,0],eig[:,1],zorder=3,scale=0.2,units='xy')
plt.show()