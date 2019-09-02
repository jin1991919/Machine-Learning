import cv2
import numpy as np
pic=cv2.imread('01.png',cv2.IMREAD_COLOR)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rc('axes',**{'grid':False})
plt.imshow(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB))
img_data=pic/255.0
img_data=img_data.reshape((-1,3))
def plot_pixels(data,title,colors=None,N=10000)
    if colors is None:
        colors=data
        rng=np.random.RandomState(0)
        i=rng.permutation(data.shape[0])[:N]
        colors=colors[i]
        pixel=data[i].T
        R,G,B=pixel[0],pixel[1],pixel[2]
        fig,ax=plt.subplots(1,2,figsize=(16,6))
        
plt.show()