
import cv2 
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import prep as prep

name='20150302053034Lh.png' # clear
#name="20150302025534Lh.png" # fair
#name="20150302065934Lh.png" # cloudy

img0 = prep.readim2gray(name)
img=cv2.medianBlur(img0,13) #give a smooth


def multi_show(imgs):
    fig = plt.figure(figsize=(8,8))
    for i in range(len(imgs)):
        img = imgs[i]
        title=str(i+1)
        #行，列，索引
        plt.subplot(2,2,i+1)
        plt.imshow(img,cmap='gray')
        plt.title(title,fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()
corect,center,rsun=prep.remove_limbdark(img0)
imgs = [img0,corect]
print(center)
multi_show(imgs)

print("all is done.")
