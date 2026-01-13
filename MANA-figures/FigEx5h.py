# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt

spk_count = np.array([3553, 13834]) #N-MNIST. DVS-Gesture

plt.figure(figsize=(2,4),dpi=256)

plt.bar([0,2], spk_count, width=0.5, color='gray', edgecolor='black')

plt.xticks([0,2],['MNIST', 'DVSGesture'],size=12)
plt.yticks([i for i in range(0,16000,2000)],['0','2K','4K','6K','8K','10K','12K','14K'],size=12)
plt.xlim(-0.5,2.5)

plt.ylabel('Spike count / Sample',size=12)
plt.show()
plt.close()