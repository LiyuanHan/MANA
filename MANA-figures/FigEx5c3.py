
import matplotlib.pyplot as plt

name__dataset = 'rotate'

model_type_list = [		# plot in sequence
	"pureMLP", 
	"pureGRU", 
	"pureTransformer",
	"MACUDA-ANN", 
	"MACUDA-SNN", 
]

avg_list = []
plot_list = []

color_dict = {
	"MACUDA-ANN": '#da357eb3', 
	"MACUDA-SNN": '#00a29aa6',
	"pureMLP": '#33a9ff33', 
	"pureGRU": '#6baed680',
	"pureTransformer": '#4292c6CD',
}

fig, ax = plt.subplots(figsize=(6, 6))
labels = range(0, 15, 2)


import pickle


	

import pickle


with open('./data/FigEx5c3.pkl', 'rb') as f:
	acc_turn_dict = pickle.load(f)



for name__model_type in model_type_list:

	acc_turn_dict[name__model_type] = acc_turn_dict[name__model_type].mean(axis=1)	# (days)
	# print(acc_turn_dict[name__model_type])

	avg_list.append(sum(acc_turn_dict[name__model_type]) / len(acc_turn_dict[name__model_type]))
	plot_tmp, = plt.plot(labels, acc_turn_dict[name__model_type], marker='s', linewidth=2, linestyle='solid', color=color_dict[name__model_type])
	plot_list.append(plot_tmp)

	# ax.axhline(y=avg_list[-1], color=color_dict[name__model_type], linestyle='--', linewidth=0.5)


plt.title('Rotate')
plt.xticks(labels)
plt.ylim(20, 100)
plt.yticks(range(30, 110, 10))


plt.legend(plot_list, ["MLP", "GRU", "CEBRA", "Transformer", "MANA (ANN)", "MANA (SNN)"], loc='lower right', bbox_to_anchor=(1, 0))

plt.xlabel("Time interval (week)", fontsize=12)
plt.ylabel("Accuracy(%)", fontsize=12)

bwith = 2
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)

path__savefig = 'figs/FigEx5c3.pdf'
plt.savefig(path__savefig, format='pdf')

