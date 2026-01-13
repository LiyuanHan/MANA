import pickle
import numpy as np
from matplotlib import pyplot as plt

# if __name__ == "__main__":

with open('./data/FigEx5d.pkl', 'rb') as f:
    acc_np_list = pickle.load(f)

x = [i + 1 for i in range(15)]

plt.plot(x, acc_np_list[0], label='Real recording data', color='red')
plt.plot(x, acc_np_list[1], label='Deterioration (drift+rotate)', color='blue')

plt.scatter(x, acc_np_list[0], color='red')
plt.scatter(x, acc_np_list[1], color='blue')

plt.title('Periodic fluctuation reproduced by synthetic data', fontsize=12)
plt.xlabel('Test days (2 days per week)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

plt.xticks([])
plt.yticks(np.arange(0, 110, 10))

plt.legend()
plt.ylim([0, 110])

plt.show()
plt.close()
