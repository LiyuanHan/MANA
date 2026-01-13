# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt

power = np.array([9.7, 28.2]) #N-MNIST. DVS-Gesture

plt.figure(figsize=(2,4),dpi=256)

plt.bar([0,2], power, width=0.5, color='#30aba6', edgecolor='black')

plt.xticks([0,2],['MNIST', 'DVSGesture'],size=12)
plt.yticks([i for i in range(0,60,10)],size=12)
plt.xlim(-0.5,2.5)

plt.ylabel('Cost (mW)',size=12)
plt.show()
plt.close()