# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import rcParams
import matplotlib.font_manager as font_manager


def pro_array(Array):
    min_array = np.min(Array, axis=1)
    max_array = np.max(Array, axis=1)
    ave_array = np.average(Array, axis=1)
    return max_array, ave_array, min_array
    
legend_font = {
    'family': 'Arial',  # 字体
    'style': 'normal',
    'size': 10,  # 字号
    'weight': "normal",  # 是否加粗，不加粗
}    

#绘制图片:N-MNIST
NMNIST_Final = np.loadtxt("./data/FigEx5ef_NMNIST.txt")
Array = NMNIST_Final
plt.figure(dpi=256,figsize=(3,2))
max_array, ave_array, min_array = pro_array(Array)
plt.fill_between(np.arange(Array.shape[0]), min_array, max_array, color=['#90C9E7'], alpha=1)
plt.plot(np.arange(Array.shape[0]), ave_array, color='black', label='N-MNIST')
plt.yticks(np.arange(90,98,2), weight='normal')
plt.xticks(np.arange(0,110,20), weight='normal')
plt.ylabel("Accuracy (%)", weight='normal')
plt.xlabel("Training epochs", weight='normal')
plt.hlines(y=np.max(ave_array), xmin=0, xmax=100, color='red', linestyles='--')
plt.text(x=60, y=np.max(ave_array)-1, s='Max: '+str(round(np.max(ave_array), 2))+'%', color='red', weight='bold', size=9)
plt.show()
plt.close()

#绘制图片:DVS-Gesture
DVSGesture_Final = np.loadtxt("./data/FigEx5ef_DVSGesture.txt")
Array = DVSGesture_Final
plt.figure(dpi=256,figsize=(3,2))
max_array, ave_array, min_array = pro_array(Array)
plt.fill_between(np.arange(Array.shape[0]), min_array, max_array, color=['#90C9E7'], alpha=1)
plt.plot(np.arange(Array.shape[0]), ave_array, color='black', label='DVS-Gesture')
plt.yticks(np.arange(60,100,20), weight='normal')
plt.xticks(np.arange(0,110,20), weight='normal')
plt.ylabel("Accuracy (%)", weight='normal')
plt.xlabel("Training epochs", weight='normal')
plt.hlines(y=np.max(ave_array), xmin=0, xmax=100, color='red', linestyles='--')
plt.text(x=60, y=np.max(ave_array)-5, s='Max: '+str(round(np.max(ave_array), 2))+'%', color='red', weight='bold', size=9)
plt.show()
plt.close()