import random

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np



import argparse
from model.digit5 import CNN, Classifier
from model.amazon import AmazonMLP, AmazonClassifier
from model.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
from model.domainnet import DomainNet_encoder, DomainNetClassifier
# from model.resnet import ResNetSNN_T
from datasets.DigitFive import digit5_dataset_read
from datasets.AmazonReview import amazon_dataset_read
from lib.utils.federated_utils import *
from train.train import train, test
from datasets.MiniDomainNet import get_mini_domainnet_dloader
from datasets.OfficeCaltech10 import get_office_caltech10_dloader
# from datasets.DomainNet import get_domainnet_dloader_new
from datasets.Office31 import get_office31_dloader
from datasets.DomainNet import *
import os
from os import path
import shutil
import yaml
import cebra
from cebra import CEBRA

import cebra.datasets

import pickle

import logging

# Default settings
parser = argparse.ArgumentParser(description='K3DA Official Implement')
# Dataset Parameters
parser.add_argument("--config", default="DomainNet.yaml")
parser.add_argument('-bp', '--base-path', default="")
parser.add_argument('--target-domain', default="target_domain", type=str, help="The target domain we want to perform domain adaptation")
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
# Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=str, metavar='N', help='the x-th time of training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# Optimizer Parameters
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
parser.add_argument('--wd', '--weight-decay', default=5e-3, type=float)    # -4 -》-3
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
parser.add_argument('--variable', default=5, type=int, help='A numerical parameter')
# parser.add_argument('--seed', type=int)
args = parser.parse_args()

# set the visible GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import pdb



### model and task relevant parameters

name__model_type = "MANA-ANN"
print(f"Model: {name__model_type}")

always_train = True					# always ignore saved model file and train anew
num_workers = 4						# specify dataloader num_workers


### import from dataset config file

# import dataset name and basic properties
name__dataset = "drift"
print(f"Dataset: {name__dataset}")	
num_classes = 10
sequence_length = 28
output_dimension = 20
print(f"	num_classes = {num_classes}")
print(f"	sequence_length = {sequence_length}")
print(f"	output_dimension = {output_dimension}")

# import task type
name__task_type = "crossday"
print(f"Task type: {name__task_type}")

# import variable and verify range
variable = args.variable
if variable not in [5, 6, 7, 8, 9, 10, 11, 12]:
	print(f"WARNING: variable = {variable} is not in the pre-specified range.")

# import day_from_to_dict and deduce train-test variables
day_from_to_dict = {
	'5': [1, 4],
	'6': [2, 5],
	'7': [3, 6],
	'8': [4, 7],
	'9': [5, 8],
	'10': [6, 9],
	'11': [7, 10],
	'12': [8, 11],
	'13': [9, 12],
	'14': [10, 13],
	'15': [11, 14],
	'16': [12, 15],
	'17': [13, 16],
	'18': [14, 17],
}
Day_from = day_from_to_dict[str(variable)][0]
Day_to = day_from_to_dict[str(variable)][1] 
test_Day = variable
print(f"Variables: train from {Day_from} to {Day_to}, test in {test_Day}.")

# import data and project path
path__all_data = "./AI-tasks/data"
path__all_projects = "./AI-tasks"
path__data = path__all_data + "/" + name__dataset
path__python = path__all_projects + "/" + name__dataset + "/" + name__model_type
path__res = path__all_projects + "/" + name__dataset + "/" + name__model_type + "/res"
model_savepath = path__all_projects + "/" + name__dataset + "/" + name__model_type + "/res/model"	# WARNING 1
print(f"Result path: {path__res}")

# model structure parameters
### MODEL FROZEN

# model learning parameters
batch_size = 16
learning_rate = 1e-3
num_epochs = 150
threshold__train_loss_exit = 1e-3

# model criterion parameter
usingCrossEntropy = True		# default: True  # MSEloss if False

data_from_to = '20221103_20230402_delete'

# import config files
with open(path__python + r"/config/{}".format(args.config)) as file:
	configs = yaml.full_load(file)


######

align_method = 2
print(f'align_method = {align_method}')

seed_set = np.array([31, 44, 46, 54, 59])
turns = int(seed_set.shape[0])

log_name = path__python + f'/log/log_{variable}.log'

# 创建logger对象
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler(log_name)
file_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理器添加到logger对象中
logger.addHandler(file_handler)

# 打印log信息
logger.info('This is a log message.')


def main(args=args, configs=configs):
	import numpy as np
	import torch
	data_from_to = '20221103_20230402_delete'

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def align_embedings_cross_days(cebra_pos_train, cebra_pos_test):
		if align_method == 4:
			return cebra_pos_test
		elif align_method == 5:
			Q_train, _ = torch.linalg.qr(cebra_pos_train, 'complete')
			cebra_pos_test_align = Q_train @ cebra_pos_test
			return cebra_pos_test_align

		cebra_pos_train_sample = cebra_pos_train

		# torch
		Q_train, R_train = torch.linalg.qr(cebra_pos_train_sample)
		Q_test, R_test = torch.linalg.qr(cebra_pos_test)
		U, S, V = torch.linalg.svd(Q_train.T @ Q_test)
		V = V.T

		if align_method == 1:
			cebra_pos_test_align = cebra_pos_test @ torch.linalg.pinv(R_test) @ V @ torch.linalg.inv(U) @ R_train
		elif align_method == 2:
			cebra_pos_test_align = Q_train @ U @ torch.linalg.pinv(V) @ R_test
		elif align_method == 3:
			cebra_pos_test_align = Q_train @ R_test
		elif align_method == 6:
			Q = torch.linalg.pinv(cebra_pos_test.T) @ cebra_pos_test.T @ cebra_pos_train @ torch.linalg.pinv(cebra_pos_test)
			cebra_pos_test_align = Q @ cebra_pos_test
			return cebra_pos_test_align

		return cebra_pos_test_align


	##
	# data loading
	bin_method = 'expect_bins'
	expect_num_bins = 50
	print('bin_method: ', bin_method)
	print('expect_num_bins: ', expect_num_bins)

	if bin_method == 'expect_bins':
		all_micro_spikes_concat = torch.load(path__data + '/x.pt')
		all_macro_conditions_concat = torch.load(path__data + '/y.pt')

		load_data_1 = torch.load(path__data + '/len_of_each_session.pt')		
		len_for_each_session_trial = load_data_1							

		load_data_2 = torch.load(path__data + '/target.pt')
		target_for_each_session_trial = load_data_2
		
		for i in range(len(target_for_each_session_trial)):
				target_for_each_session_trial[i] = (np.array(target_for_each_session_trial[i]).astype(int)).tolist()



	week_count_list = [1,1,1,1]

	if True:

		max_acc_list = []
		last_acc_list = []
		acc_of_week = []



		# import pdb; pdb.set_trace()
		for turn in range(turns):
			# set the dataloader list, model list, optimizer list, optimizer schedule list

			print("Turn", turn, "started ===================================================")

			train_dloaders = []
			test_dloaders = []
			models = []
			classifiers = []
			optimizers = []
			classifier_optimizers = []
			optimizer_schedulers = []
			classifier_optimizer_schedulers = []
			# build dataset
			if configs["DataConfig"]["dataset"] == "DomainNet":
				seeds = int(seed_set[turn])
				random.seed(seeds)
				np.random.seed(seeds)
				torch.manual_seed(seeds)
				torch.cuda.manual_seed_all(seeds)

				torch.backends.cudnn.deterministic = True

				print('turn: ', turn)
				# today = '2023-10-08-v7-turn_' + str(turn)
				today = '2023-10-245-v2-p2-delete-turn_align_KD3A_' + str(turn)
				# today = str(today_list[test_count]) + str(turn)
				## split_data and Load the model for training
				max_iterations = 10000  # default is 5000.

				print('output_dimension: ', output_dimension)

				if bin_method == 'expect_bins':
					train_data = 'days'
					print('train_data: ', train_data)
					if train_data == 'days':
						if data_from_to == '20221103_20230402_delete':
							day = { # day[i] = i, AI datasets have no session segmentations. # should be changed if using session-segmented datasets.
								str(i): i for i in range(200)
							}


							print('------------------Day_from_{}_to_{}------------------'.format(Day_from, Day_to))
							print('--------------------test_Day: {}------------------'.format(test_Day))


							def split_data(data_spike_train, data_label, len_for_each_session_trial, Day_from, Day_to):

								split_idx_start_beg = 0
								for i in range(day[str(Day_from - 1)]):
									split_idx_start_beg += int(sum(len_for_each_session_trial[i]))
								split_idx_start_end = 0
								for i in range(day[str(Day_to)]):
									split_idx_start_end += sum(len_for_each_session_trial[i])

								split_idx_start_end = int(split_idx_start_end)

								split_idx_start = 0
								for i in range(day[str(test_Day - 1)]):
									split_idx_start += int(sum(len_for_each_session_trial[i]))
								split_idx_end = 0  # 只预测第test_Day这一天的
								for i in range(day[str(test_Day)]):
									split_idx_end += int(sum(len_for_each_session_trial[i]))

								# import pdb; pdb.set_trace()
								neural_train = data_spike_train[split_idx_start_beg:split_idx_start_end, :]
								label_train = data_label[split_idx_start_beg:split_idx_start_end, :]

								neural_test = data_spike_train[split_idx_start:split_idx_end, :]
								label_test = data_label[split_idx_start:split_idx_end, :]
								return neural_train, neural_test, label_train, label_test

				# split data
				neural_train, neural_test, label_train, label_test = split_data(
					all_micro_spikes_concat,
					all_macro_conditions_concat,
					len_for_each_session_trial, 
					Day_from,
					Day_to
				)  # direction

				add_num = 0
				add_num_to = 0

				print('neural_train_length: ', len(neural_train))
				print('split data...finished!')

				distance = 'euclidean'
				# distance = 'cosine'
				decoder_method = 'gru_test_8_aligned'
				print('distance: ', distance)
				# model
				cl_dir_model = CEBRA(model_architecture='offset10-model',
									batch_size=512,
									learning_rate=3e-4,
									temperature=1,
									output_dimension=output_dimension,
									max_iterations=max_iterations,
									distance=distance,
									device='cuda_if_available',
									verbose=True)
				if bin_method == 'expect_bins':
					if train_data == 'days':

						path__cl_file = path__res + "/cl_dir" + '/cl_dir_model_dim_' + distance + '_' + str(output_dimension) + '_' + str(max_iterations) + '_' + name__dataset + '_' + name__task_type + '_expect_bin_' + str(expect_num_bins) + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_'+ str(test_Day)  + today + '_' + data_from_to + '.pt'

						if not os.path.exists(path__cl_file):

							cl_dir_model.fit(neural_train, label_train)
							cl_dir_model.save(path__cl_file)
							print('save model ...... finished!')

				## ## Load the models and get the corresponding embeddings
				if bin_method == 'expect_bins':
					if train_data == 'days':
						cl_dir_model = cebra.CEBRA.load(path__cl_file)
						cebra_dir_train = cl_dir_model.transform(neural_train)
						# pdb.set_trace()
						cebra_dir_test = cl_dir_model.transform(neural_test)
						cebra_dir_test=torch.tensor(cebra_dir_test)


						torch.save(cebra_dir_test.cpu().data, path__res + "/align" + '/without_align_dir_' + decoder_method + '_distance_' + distance + '_' + name__dataset + '_' + name__task_type + '_expect_bin_' + str(expect_num_bins) + '_data_aligned_test_method_' + str(align_method) + '_0' + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_'  + str(test_Day) + today + '_together_' + data_from_to + '.pt')


				print('Load the model ..... finished!')

				#
				decoder_method = 'gru_test_8_aligned'
				print('decoder_method: ', decoder_method)
				import torch
				import torch.nn as nn
				import torch.optim as optim
				import numpy as np
				import torch.utils.data as Data

				
				input_size = output_dimension * num_classes  # dim of embedding output

				weather_together = True

				if weather_together:
					if data_from_to == '20221103_20230402_delete':

						with open(path__data + '/stard_'+ str(Day_to) +'.pkl', 'rb') as f:
							cebra_dir_train_stard_embeddings = pickle.load(f)  # (2800, 20)  len = 10
							print('load standard embeddings ...... finished!')

					path__align_file = path__res + "/align" + '/align_dir_' + decoder_method + '_distance_' + distance + '_' + name__dataset + '_' + name__task_type + '_expect_bin_' + str(expect_num_bins) + '_data_aligned_method_' + str(align_method) + '_0' + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'

					# 训练部分对齐
					if not os.path.exists(path__align_file):
						print('staring data aligned......')
						id_target_align_train = [[] for _ in range(num_classes)]
						id_l, id_r = 0, 0
						images_train_align = torch.zeros((len(cebra_dir_train), input_size)).to(device)
						for i in range(day[str(Day_from - 1)], day[str(Day_to)]):
							id_target_align = [[] for _ in range(num_classes)]
							for j in range(len(len_for_each_session_trial[i])):
								id_r += len_for_each_session_trial[i][j]
								id_target_align[target_for_each_session_trial[i][j]].append(
									torch.arange(id_l, id_r)
								)
								id_l = id_r

							for h in range(len(id_target_align)):
								id_target_align[h] = torch.cat(id_target_align[h])
								id_target_align_train[h].append(id_target_align[h])
								for m in range(num_classes):


									len_train = len(cebra_dir_train_stard_embeddings['embeddings'][
													cebra_dir_train_stard_embeddings['id_target'][m].astype(int), :])  # 280
									len_test = len(id_target_align[h])  # 364
									# import pdb; pdb.set_trace()

									if len_train >= len_test:
										nums = len_train // len_test
										temp_align = []
										images_align = []

										for k in range(nums):
											idx = torch.arange(len_train)[k * len_test:(k + 1) * len_test]

											temp_align.append(align_embedings_cross_days(
												torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
															cebra_dir_train_stard_embeddings['id_target'][m].astype(int), :][idx, :]).double().to(device), torch.tensor(cebra_dir_train[id_target_align[h], :]).double().to(device)))

										images_align.append(torch.stack(temp_align, dim=0).mean(0))
										images_train_align[id_target_align[h], m * output_dimension:(m + 1) * output_dimension] = \
											torch.stack(images_align, dim=0).mean(0).float()
									else:
										nums = len_test // len_train
										temp_align = []
										images_align = []

										for k in range(nums):
											idx = torch.arange(len_test)[k * len_train:(k + 1) * len_train]

											# import pdb; pdb.set_trace()

											temp_align.append(align_embedings_cross_days(
												torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
															cebra_dir_train_stard_embeddings['id_target'][m].astype(int), :]).double().to(
													device),
												torch.tensor(cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
											))
										if len_test % len_train > 0:
											idx = torch.arange(len_test)[nums * len_train:]
											temp_align.append(align_embedings_cross_days(
												torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
															cebra_dir_train_stard_embeddings['id_target'][m].astype(int)[
															:len_test % len_train],
															:]).double(
												).to(device),
												torch.tensor(cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
											))
										images_align.append(torch.cat(temp_align, dim=0).mean(0))
										images_train_align[id_target_align[h], m * output_dimension:(m + 1) * output_dimension] = \
											torch.stack(images_align, dim=0).mean(0).float()

						images_train_align = images_train_align.cpu().data  # torch.Size([28000, 200])

						# import pdb; pdb.set_trace()

						torch.save(images_train_align.cpu().data, path__align_file)
					else:
						images_train_align = torch.load(path__align_file)

					path__align_test_file = path__res + "/align" + '/align_dir_' + decoder_method + '_distance_' + distance + '_' + name__dataset + '_' + name__task_type + '_expect_bin_' + str(expect_num_bins) + '_data_aligned_test_method_' + str(align_method) + '_0' + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'

					# print('train data aligned......finished!')
					# 测试部分对齐
					if not os.path.exists(path__align_test_file):

						# for idx in range(8):
						# 	id_target_align_train[idx] = torch.cat(id_target_align_train[idx])

						id_l, id_r = 0, 0
						images_test_align = torch.zeros((len(cebra_dir_test), input_size)).to(device)
						for i in range(day[str(test_Day - 1)], day[str(test_Day)]):
							id_target_align_test = [[] for _ in range(num_classes)]
							for j in range(len(len_for_each_session_trial[i])):
								id_r += len_for_each_session_trial[i][j]
								id_target_align_test[target_for_each_session_trial[i][j]].append(
									torch.arange(id_l, id_r)
								)
								id_l = id_r

							for h in range(len(id_target_align_test)):
								id_target_align_test[h] = torch.cat(id_target_align_test[h])
								for m in range(num_classes):
									# print(f"i={i},j={j},h={h},m={m}")
									# len_train = len(images_train_align[id_target_align_train[h],
									#                 m * output_dimension:(m + 1) * output_dimension])
									len_train = len(cebra_dir_train_stard_embeddings['embeddings'][
													cebra_dir_train_stard_embeddings['id_target'][m].astype(int), :])
									len_test = len(id_target_align_test[h])
									if len_train >= len_test:
										nums = len_train // len_test
										temp_align = []
										images_align = []

										if nums > 10:
											nums = 10

										for k in range(nums):
											idx = torch.arange(len_train)[k * len_test:(k + 1) * len_test]

											# temp_align.append(align_embedings_cross_days(
											# 	images_train_align[
											# 	id_target_align_train[h], m * output_dimension:(m + 1) * output_dimension][
											# 	idx, :].double().to(device),
											# 	torch.tensor(cebra_dir_test[id_target_align_test[h], :]).double().to(device)
											# ))
											temp_align.append(align_embedings_cross_days(
												torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
															cebra_dir_train_stard_embeddings['id_target'][m].astype(int), :][
															idx, :]).double().to(device),
												torch.tensor(cebra_dir_test[id_target_align_test[h], :]).double().to(device)
											))
										images_align.append(torch.stack(temp_align, dim=0).mean(0))
										images_test_align[id_target_align_test[h],
										m * output_dimension:(m + 1) * output_dimension] = \
											torch.stack(images_align, dim=0).mean(0).float()
									else:
										nums = len_test // len_train
										temp_align = []
										images_align = []
										for k in range(nums):
											idx = torch.arange(len_test)[k * len_train:(k + 1) * len_train]

											# temp_align.append(align_embedings_cross_days(
											# 	images_train_align[
											# 	id_target_align_train[h],
											# 	m * output_dimension:(m + 1) * output_dimension].double().to(device),
											# 	torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
											# ))
											temp_align.append(align_embedings_cross_days(
												torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
															cebra_dir_train_stard_embeddings['id_target'][m].astype(int), :]).double().to(device),
												torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
											))
										if len_test % len_train > 0:
											idx = torch.arange(len_test)[nums * len_train:]
											# temp_align.append(align_embedings_cross_days(
											# 	torch.tensor(images_train_align[
											# 	             id_target_align_train[h][:len_test % len_train],
											# 	             m * output_dimension:(m + 1) * output_dimension]).double().to(device),
											# 	torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
											# ))
											temp_align.append(align_embedings_cross_days(
												torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
															cebra_dir_train_stard_embeddings['id_target'][m].astype(int)[:len_test % len_train],
															:]).double().to(device),
												torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
											))
										images_align.append(torch.cat(temp_align, dim=0).mean(0))
										images_test_align[id_target_align_test[h],
										m * output_dimension:(m + 1) * output_dimension] = \
											torch.stack(images_align, dim=0).mean(0).float()

						images_test_align = images_test_align.cpu().data
						torch.save(images_test_align.cpu().data, path__align_test_file)

					else:
						images_test_align = torch.load(path__align_test_file)
					print('test data aligned......finished!')

				def split_data_for_day(data_spike_train, data_label, len_for_each_session_trial, Day_from):
					split_day_data = []
					split_day_label = []
					flag = 0
					count_left = 0
					for count in week_count_list:
						split_idx_start_beg = 0
						for i in range(day[str(Day_from - 1 + count_left)]):
							split_idx_start_beg += sum(len_for_each_session_trial[i])
						split_idx_start_end = 0
						for i in range(day[str(Day_from -1 + count_left + count)]):
							split_idx_start_end += sum(len_for_each_session_trial[i])
						day_data = data_spike_train[flag:flag+split_idx_start_end-split_idx_start_beg, :]
						day_label = data_label[flag:flag+split_idx_start_end-split_idx_start_beg, :]
						flag = flag + split_idx_start_end - split_idx_start_beg
						count_left = count + count_left

						split_day_data.append(day_data)
						split_day_label.append(day_label)
					return split_day_data, split_day_label
				qy_train_data, qy_train_label = split_data_for_day(images_train_align.float(), label_train,
																			len_for_each_session_trial, Day_from)

				qy_test_data = images_test_align.float()
				qy_test_label = label_test



				# BCI数据加载
				# domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

				# domains = torch.arange(0, 42, 1).tolist()
				# this_source_domain = [0, 2, 3, 4, 5]
				# this_target_domain = [1]
				args.source_domains = list(range(len(week_count_list)))

				# import pdb; pdb.set_trace()
				test_data = qy_test_data.reshape(-1,sequence_length,input_size)
				test_label = qy_test_label[::sequence_length].view(-1)
				# test_label = np.eye(8)[test_label].reshape(-1,8)

				### print shape

				# print(test_data.shape)
				# print(test_label.shape)

				target_train_dloader ,target_test_dloader = get_domainnet_dloader_test(None, test_data, test_label, configs["TrainingConfig"]["batch_size"], args.workers)

				# target_train_dloader, target_test_dloader = get_domainnet_dloader_test(args.base_path,
				# 																	   test_data, test_label,
				# 																	   1,
				# 																	   args.workers)

				train_dloaders.append(target_train_dloader)
				test_dloaders.append(target_test_dloader)
				test_model = DomainNet_encoder('resnet101', args.bn_momentum, False, False).cuda()
				# print(test_model.__dict__.keys())
				models.append(test_model)
				classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], num_classes, args.data_parallel).cuda())
				# domains.remove(this_target_domain)

				for num in range(len(week_count_list)):
					# import pdb; pdb.set_trace()
					domain_data = np.array(qy_train_data[num]).reshape(-1,sequence_length,input_size)
					domain_label = qy_train_label[num][::sequence_length].view(-1)
					# domain_label = np.eye(8)[domain_label].reshape[-1,8]

					### print shape

					# print(domain_data.shape)
					# print(domain_label.shape)

					source_train_dloader,source_test_dloader = get_domainnet_dloader_train(None, domain_data, domain_label, configs["TrainingConfig"]["batch_size"], args.workers)


					train_dloaders.append(source_train_dloader)
					test_dloaders.append(source_test_dloader)
					models.append(DomainNet_encoder(configs["ModelConfig"]["backbone"], args.bn_momentum, False, False).cuda())
					classifiers.append(
						DomainNetClassifier(configs["ModelConfig"]["backbone"], num_classes, args.data_parallel).cuda())
			else:
				raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))
			# federated learning step 1: initialize model with the same parameter (use target as standard)
			for model in models[1:]:
				for source_weight, target_weight in zip(model.named_parameters(), models[0].named_parameters()):
					# consistent parameters
					source_weight[1].data = target_weight[1].data.clone()
			# create the optimizer for each model
			for model in models:
				optimizers.append(
					torch.optim.SGD(model.parameters(), momentum=args.momentum,
					                lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
			for classifier in classifiers:
				classifier_optimizers.append(
					torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
					                lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
			# create the optimizer scheduler with cosine annealing schedule
			for optimizer in optimizers:
				optimizer_schedulers.append(
					CosineAnnealingLR(optimizer, configs["TrainingConfig"]["total_epochs"],
					                  eta_min=configs["TrainingConfig"]["learning_rate_end"]))
			for classifier_optimizer in classifier_optimizers:
				classifier_optimizer_schedulers.append(
					CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
					                  eta_min=configs["TrainingConfig"]["learning_rate_end"]))
			# create the event to save log info
			# args.train_time = 1
			# writer_log_dir = path.join(args.base_path, configs["DataConfig"]["dataset"], "runs")
			# print("create writer in {}".format(writer_log_dir))
			# if os.path.exists(writer_log_dir):
			# 	shutil.rmtree(writer_log_dir, ignore_errors=True)
			# writer = SummaryWriter(log_dir=writer_log_dir)
			# begin train
			print("Begin")

			# create the initialized domain weight
			domain_weight = create_domain_weight(len(args.source_domains))
			# adjust training strategy with communication round
			batch_per_epoch, total_epochs = decentralized_training_strategy(
				communication_rounds=configs["UMDAConfig"]["communication_rounds"],
				epoch_samples=configs["TrainingConfig"]["epoch_samples"],
				batch_size=configs["TrainingConfig"]["batch_size"],
				total_epochs=configs["TrainingConfig"]["total_epochs"])
			# train model
			acc_list = []
			train_loss_list = []
			test_loss_list = []
			domain1_loss_list = []
			domain2_loss_list = []
			domain3_loss_list = []
			domain4_loss_list = []
			domain_weight_list = []




			for epoch in range(args.start_epoch, total_epochs):

				# print("EPOCH =", epoch)

				domain_weight, train_loss, target_linear_outputs, s1_linear_outputs, s2_linear_outputs, s3_linear_outputs, s4_linear_outputs = train(
					train_dloaders, models, classifiers, optimizers,
					classifier_optimizers, epoch, num_classes=num_classes,
					model_savepath=model_savepath, turn=turn, testDay=test_Day,
					domain_weight=domain_weight, source_domains=args.source_domains,
					batch_per_epoch=batch_per_epoch, total_epochs=total_epochs,
					batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
					communication_rounds=configs["UMDAConfig"]["communication_rounds"],
					confidence_gate_begin=configs["UMDAConfig"]["confidence_gate_begin"],
					confidence_gate_end=configs["UMDAConfig"]["confidence_gate_end"],
					malicious_domain=configs["UMDAConfig"]["malicious"]["attack_domain"],
					attack_level=configs["UMDAConfig"]["malicious"]["attack_level"],
					mix_aug=(configs["DataConfig"]["dataset"] != "AmazonReview"))

				acc_top1, acc_top2, acc_top3, test_loss, test_linear_outputs = test(args.target_domain, args.source_domains, test_dloaders,
															   models, classifiers, epoch, test_Day=test_Day, total_epochs=total_epochs, today=today,
															   num_classes=num_classes, top_5_accuracy=True)

				# print(target_linear_outputs.shape,
				# 	  s1_linear_outputs.shape,
				# 	  s2_linear_outputs.shape,
				# 	  s3_linear_outputs.shape,
				# 	  s4_linear_outputs.shape,
				# 	  test_linear_outputs.shape)


				acc_list.append([acc_top1 * 100, acc_top2 * 100, acc_top3 * 100, seeds])

				train_loss_list.append(train_loss)
				# domain1_loss_list.append(domain1_loss)
				# domain2_loss_list.append(domain2_loss)
				# domain3_loss_list.append(domain3_loss)
				# domain4_loss_list.append(domain4_loss)

				test_loss_list.append(test_loss)

				# 记录每轮的domain  weight
				domain_weight_list.append(domain_weight)
				# 存储更新后的target_weight
				torch.save(models[0],
						   model_savepath + f'/target_model_state_testday_{test_Day}_turn_{turn}_epoch_{epoch}.pth')  # 存储模型状态文件
				torch.save(classifiers[0],
						   model_savepath + f'/target_model_classifier_state_testday_{test_Day}_turn_{turn}_epoch_{epoch}.pth')  # 存储分类器状态

				for scheduler in optimizer_schedulers:
					scheduler.step(epoch)
				for scheduler in classifier_optimizer_schedulers:
					scheduler.step(epoch)
				# save models every 10 epochs
				if (epoch + 1) % 10 == 0:
					# save target model with epoch, domain, model, optimizer
					save_checkpoint(
						{
							"epoch": epoch + 1,
							"domain": args.target_domain,
							"backbone": models[0].state_dict(),
							"classifier": classifiers[0].state_dict(),
							"optimizer": optimizers[0].state_dict(),
							"classifier_optimizer": classifier_optimizers[0].state_dict()
						}, 
						filename="{}.pth.tar".format(args.target_domain),
					)

            # 只存最后一轮的
			average_target_linear_outputs = target_linear_outputs.mean(dim=0).unsqueeze(0)
			average_s1_linear_outputs = s1_linear_outputs.mean(dim=0).unsqueeze(0)
			average_s2_linear_outputs = s2_linear_outputs.mean(dim=0).unsqueeze(0)
			average_s3_linear_outputs = s3_linear_outputs.mean(dim=0).unsqueeze(0)
			average_s4_linear_outputs = s4_linear_outputs.mean(dim=0).unsqueeze(0)
			average_test_linear_outputs=test_linear_outputs.mean(dim=0).unsqueeze(0)

			all_averages = [
				average_target_linear_outputs,
				average_s1_linear_outputs,
				average_s2_linear_outputs,
				average_s3_linear_outputs,
				average_s4_linear_outputs,
				average_test_linear_outputs,
			]

			# 在batch维度组合这些张量
			all_linear_outputs = torch.cat(all_averages, dim=0)

			plt.plot(train_loss_list, label='Train Loss')
			plt.plot(test_loss_list, label='Test Loss')
			plt.title('Training and Test Losses Over Epochs')
			plt.xlabel('Epochs')
			plt.ylabel('Loss')
			plt.legend()
			plt.savefig(path__res + '/loss/' + str(
				Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + f'_turn_{turn}.png')
			plt.close()

			last_acc = acc_list[-1]
			max_acc = np.max(np.array(acc_list), axis=0)
			print("Maximum accuracy", max_acc)
			print("Last epoch accuracy", last_acc)
			max_acc_list.append(max_acc)
			last_acc_list.append(last_acc)

			np.save(path__res + '/domain_weight/domain_weight'+'_testDay_' + str(test_Day)+f'_turn_{turn}.npy', domain_weight_list)

			torch.save(all_linear_outputs, path__res + '/linear_output/all_linear_outputs'+'_testDay_' + str(test_Day)+f'_turn_{turn}.pt')

		np.save(path__res + '/outputs_acc_model_' + name__model_type + '_data_' + name__dataset + '_type_' + name__task_type + '_bin_' + str(expect_num_bins) + '_align_method_' + str(align_method) + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_test_' + str(test_Day) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '.npy', max_acc_list)
		np.save(path__res + '/outputs_acc_model_' + name__model_type + '_data_' + name__dataset + '_type_' + name__task_type + '_bin_' + str(expect_num_bins) + '_align_method_' + str(align_method) + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_test_' + str(test_Day) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_last.npy', last_acc_list)


		# print accuracy results
		print("=== MAXIMUM TEST ACCURACY ===")
		print(np.stack(max_acc_list))
		all_top_1 = np.stack(max_acc_list)[:, 0]
		print("Sorted Accuracies:")
		print(np.sort(all_top_1))
		print("mean = ", np.sort(all_top_1)[-5:].mean())

		print("=== LAST EPOCH ACCURACY ===")
		print(np.stack(last_acc_list))
		all_top_1 = np.stack(last_acc_list)[:, 0]
		print("Sorted Accuracies:")
		print(np.sort(all_top_1))
		print("mean = ", np.sort(all_top_1)[-5:].mean())

		acc_of_seeds = np.mean(np.array(max_acc_list), axis=0)[:-1]
		print(acc_of_seeds)
		logger.info(acc_of_seeds)
		acc_of_week.append(acc_of_seeds)
		
	print(acc_of_week)
	logger.info(acc_of_week)


def save_checkpoint(state, filename):
	# args.train_time = 1
	# args.train_time = 1
	# filefolder = args.base_path + "/" + name__dataset + "/" + name__model_type + "/DomainNet/parameter"
	# if not path.exists(filefolder):
	# 	os.makedirs(filefolder)
	# torch.save(state, path.join(filefolder, filename))
	pass


if __name__ == "__main__":
	print('start training')
	main()
	print('end training')
