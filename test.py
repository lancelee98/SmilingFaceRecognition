import cv2 as cv
import matplotlib.pyplot as plt
import  os
import PIL.Image as Image
import numpy as np

PATH='C:/Users\Administrator\Desktop\design\select'
def matplotlib_multi_pic1(index):
    i = 0
    imglist = []
    for img_name in os.listdir(PATH):
        if i % 5 == 0 + index:
            imglist.append(PATH + '/' + img_name)
        i += 1
    print(imglist)
    for i in range(8):
        img=Image.open(imglist[i]).convert("L")
        plt.subplot(2,4,i+1)
        arr = np.asarray(img)
        plt.imshow(arr,cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()


matplotlib_multi_pic1(3)
