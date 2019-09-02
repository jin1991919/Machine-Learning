import cv2
import matplotlib.pyplot as plt
img_bgr=cv2.imread('01.png')
img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(18,6))
plt.subplot(131)
plt.imshow(img_bgr)
plt.subplot(132)
plt.imshow(img_rgb)
img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
plt.subplot(133)

corners=cv2.cornerHarris(img_gray,2,3,0.04)
plt.imshow(corners,cmap='gray')
plt.show()