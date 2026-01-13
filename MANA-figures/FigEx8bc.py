
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

fontSize = 8

days = [6,7,8,9,10,11,12,13,15,16,17,19,20]
R2score = [0.89,0.82,0.67,0.38,0.58,0.22,0.59,0.64,0.61,0.06,0.36,0.63,0.62]

fig, ax = plt.subplots(sharey=True,figsize=(7,3),dpi=512)
ax = plt.gca() 
ax.set_facecolor("none")

plt.plot(np.arange(len(R2score)), R2score, color='#f15398', marker='s', label='Without on-chip adapation')
plt.hlines(np.average(R2score), 0, len(R2score)-1, color='#f15398', linewidth=2, linestyle='--')

plt.xticks(np.arange(len(R2score)),days)
plt.xticks(fontsize=fontSize)
plt.yticks(np.arange(-1.0,1.2,0.5), fontsize=fontSize)
plt.ylim(0,1)

plt.xlabel('Decoding day', fontsize=fontSize)
plt.ylabel('R$^2$ score', fontsize=fontSize)


plt.savefig('figs/FigEx8b.pdf',bbox_inches='tight')


import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


fontSize = 8

days = [6,7,8,9,10,11,12,13,15,16,17,19,20]
power = [7.7, 7.7, 7.6, 7.7, 7.6, 7.7, 7.7, 7.7, 7.6, 7.6, 7.7, 7.6, 7.7]

fig, ax = plt.subplots(sharey=True,figsize=(7,3),dpi=512)
ax = plt.gca()

ax.set_facecolor("none")

plt.bar(np.arange(len(days)), power, color='#f15398', label='Without on-chip adapation', width=0.25)

plt.xticks(np.arange(len(R2score)),days)
plt.xticks(fontsize=fontSize)
plt.yticks(np.arange(0,60,10), fontsize=fontSize)
plt.ylim(0,50)

plt.xlabel('Decoding day', fontsize=fontSize)
plt.ylabel('Computational cost', fontsize=fontSize)

plt.savefig('figs/FigEx8c.pdf',bbox_inches='tight')
