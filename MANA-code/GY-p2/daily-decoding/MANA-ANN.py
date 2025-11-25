import random

import numpy as np

import argparse
from model.MANAANN.digit5 import CNN, Classifier
from model.MANAANN.amazon import AmazonMLP, AmazonClassifier
from model.MANAANN.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
from model.MANAANN.domainnet import DomainNet_encoder, DomainNetClassifier
from datasets.DigitFive import digit5_dataset_read
from datasets.AmazonReview import amazon_dataset_read
from lib.utils.federated_utils import *
from train.ANN.train import train, test
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 创建logger对象
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# 创建文件处理器
# file_handler = logging.FileHandler('./test_day_34.log')
# file_handler.setLevel(logging.INFO)

# 创建格式化器
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# 将文件处理器添加到logger对象中
# logger.addHandler(file_handler)

# # 打印log信息
# logger.info('This is a log message.')

# Default settings
parser = argparse.ArgumentParser(description='K3DA Official Implement')
# Dataset Parameters
parser.add_argument("--config", default="DomainNet.yaml")
parser.add_argument('-bp', '--base-path', default="")
parser.add_argument('--target-domain', type=str, help="The target domain we want to perform domain adaptation")
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
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')

# parser.add_argument('--variable_num', type=int, help='A numerical parameter')
parser.add_argument('--mix_length', default=28, type=int)
parser.add_argument('--time_window', default=10, type=int)

args = parser.parse_args()
variable_num = 24
mix_length = args.mix_length
time_window = args.time_window

args.base_path = ''

always_rereducedim = True
always_realign = True

print(" ")
if always_rereducedim:
    print("Dimension reduction procedure is always executed regardless of existent dim_reduced files.")
if always_realign:
    print("Alignment procedure is always executed regardless of existent aligned files.")
print(" ")

# import config files
with open(r"./GY-p2/daily-decoding/config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn



path__loadData = './GY-p2/data'

base_path = "./GY-p2/daily-decoding"

path__dimReduced = base_path + '/data/dim_reduced'
path__aligned = base_path + '/data/aligned'
path__decodedResults = base_path + '/results'

path_subpath__modelName = f'/MANA-ANN'

print(path_subpath__modelName)
print(" ")

os.makedirs(path__dimReduced, exist_ok=True)
os.makedirs(path__aligned, exist_ok=True)
os.makedirs(path__decodedResults, exist_ok=True)

if not os.path.exists(path__decodedResults + path_subpath__modelName):
    os.makedirs(path__decodedResults + path_subpath__modelName)
if not os.path.exists(path__decodedResults + path_subpath__modelName + '/acc'):
    os.makedirs(path__decodedResults + path_subpath__modelName + '/acc')
if not os.path.exists(path__decodedResults + path_subpath__modelName + '/loss'):
    os.makedirs(path__decodedResults + path_subpath__modelName + '/loss')



def main(args=args, configs=configs):
    import numpy as np
    import torch
    # data_from_to = '20221103_20230402_delete'
    # data_from_to = '20221103_20230630' # all days
    data_from_to = '20221103_20230911' # all days
    
    device = torch.device("cuda:0")
    
    align_method = 2
    print('align_method= ', align_method)

    # filepath = ''
    
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
    
    # if bin_method == 'expect_bins':
        
    # 	all_micro_spikes_concat = torch.load(path__loadData + '/all_micro_spikes_expect_bin_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
    # 	all_macro_conditions_concat = torch.load(path__loadData + '/all_macro_conditions_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
        
    # 	len_for_each_session_trial = torch.load(path__loadData + '/len_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
    # 	target_for_each_session_trial = torch.load(path__loadData + '/target_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
        
    # 	# 1-8 ==> 0-7
    # 	all_macro_conditions_concat = all_macro_conditions_concat - torch.tensor(1)
    # 	for i in range(len(target_for_each_session_trial)):
    # 		target_for_each_session_trial[i] = (np.array(target_for_each_session_trial[i]) - 1).tolist()
    
    

    # daily-decoding setup
    seed_set = np.array([31, 44, 46, 54, 59])
    Day_from = variable_num
    Day_to = Day_from
    test_Day_list = [Day_from]
    week_count_list = [1, 1, 1, 1]


    acc_of_week = []

    for test_count in range(len(test_Day_list)):
        test_Day = test_Day_list[test_count]
        max_acc_list = []
        last_acc_list = []
        unsupervised_acc_list = []
        for turn in range(5):
            # set the dataloader list, model list, optimizer list, optimizer schedule list
            
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
                
                print('\nturn: ', turn)
                # today = '2023-10-08-v7-turn_' + str(turn)
                # today = '2024-01-22-v1-p2-delete-turn_34_' + str(turn)
                # today = '2024-01-25-v1-p2-delete_alignKD3A_single_turn_' + str(turn)  # train:80%, test: 20%; standard embedding: Day_from + all training, (shuffled), test_align- line 439, target sourch: 32
                # today = '2024-01-25-v1-p2-delete_alignKD3A_single_TarSou-30_turn_' + str(turn)  # train:80%, test: 20%; standard embedding: Day_from + all training, (shuffled), test_align- line 439, target sourch: 8

                today = '2024-07-28-v1-p2-delete_alignKD3A_single_TarSou-32_turn_' + str(turn)  # train:80%, test: 20%; standard embedding: Day_from + all training, (shuffled), test_align- line 439, target sourch: 8
                
                # # split_data and Load the model for training
                max_iterations = 10000  # default is 5000.
                output_dimension = 36  # here, we set as a variable for hypothesis testing below.
                num_hidden_units = 64
                print('output_dimension: ', output_dimension)
                
                if bin_method == 'expect_bins':
                    train_data = 'days'
                    print('train_data: ', train_data)
                    if train_data == 'days':

                        day = {
                            '0': 0, 
                            '1': 5, '2': 10, 
                            '3': 13, '4': 18, '5': 24, '6': 30, 
                            '7': 36, '8': 40, '9': 42, '10': 46, '11': 48, 
                            '12': 53, 
                            '13': 55, '14': 58, '15': 62, 
                            '16': 63, '17': 66, '18': 68, 
                            '19': 70, '20': 71, '21': 74, 
                            '22': 81, 
                            '23': 86, 
                            '24': 88, '25': 90, '26': 92, '27': 94, 
                            '28': 96, '29': 98, '30': 100, '31': 101, 
                            '32': 103, 
                            '33': 106, '34': 110, '35': 114, 
                            '36': 118, '37': 120, '38': 122, 
                            '39': 124, '40': 128, 
                            '41': 132, '42': 134, '43': 136, 
                            '44': 138, '45': 140, 
                            '46': 142, 
                            '47': 143, '48': 146, '49': 149, 
                            '50': 152, '51': 154, '52': 158, '53': 161, 
                            '54': 165, '55': 168, '56': 173, 
                            '57': 175, '58': 177, '59': 179, '60': 182, 
                            '61': 184, '62': 189, '63': 195, 
                            '64': 197, '65': 202, 
                            '66': 205, '67': 208, '68': 210, '69': 212, '70': 215, 
                            '71': 217, '72': 221, 
                            '73': 223, '74': 227, '75': 229, 
                            '76': 232, '77': 238, 
                            '78': 241, '79': 245, '80': 251, '81': 255, '82': 259, 
                            '83': 262, '84': 267, '85': 268, '86': 273, 
                            '87': 277, '88': 281, 
                            '89': 283, '90': 285, '91': 288, '92': 292, 
                            '93': 295, '94': 298, '95': 300, '96': 304, '97': 307, 
                            '98': 310, '99': 312, '100': 314, 
                            '101': 315, '102': 317, '103': 321, 
                            '104': 323, '105': 327, '106': 329, 
                            '107': 332, '108': 335, '109': 338, '110': 342, 
                            '111': 345, '112': 349, '113': 353, '114': 357, 
                            '115': 360, '116': 363, '117': 366, '118': 369, 
                            '119': 372,
                        }
                        
                        # Day_from = 1
                        # Day_to = 58
                        # test_Day = Day_to + 1
                        ratio = 0.8
                        print('------------------Day_from_{}_to_{}------------------'.format(Day_from, Day_to))
                        print('--------------------test_Day: {}------------------'.format(test_Day))
                        
                        # with shuffle
                        # def split_data(data_spike_train, data_label, len_for_each_session_trial, Day_from, Day_to):
                            
                        #     split_idx_start_beg = 0
                        #     for i in range(day[str(Day_from - 1)]):
                        #         split_idx_start_beg += sum(len_for_each_session_trial[i])
                        #     split_idx_start_end = 0
                        #     for i in range(day[str(Day_to)]):
                        #         split_idx_start_end += sum(len_for_each_session_trial[i])
                            
                        #     data_neural_train = data_spike_train[split_idx_start_beg:split_idx_start_end, :]
                        #     data_label_train = data_label[split_idx_start_beg:split_idx_start_end, :]
                            
                        #     random_indices = np.arange(0, len(data_label_train), expect_num_bins)
                        #     np.random.shuffle(random_indices)
                            
                        #     len_train = (int(len(data_neural_train) * ratio) // expect_num_bins)
                            
                        #     random_indices_augmented_train = []
                        #     for num in random_indices[:len_train]:
                        #         augmented_indices = [num + i for i in range(expect_num_bins)]
                        #         random_indices_augmented_train.extend(augmented_indices)
                            
                        #     random_indices_augmented_test = []
                        #     for num in random_indices[len_train:]:
                        #         augmented_indices = [num + i for i in range(expect_num_bins)]
                        #         random_indices_augmented_test.extend(augmented_indices)
                            
                        #     neural_train = data_neural_train[np.stack(random_indices_augmented_train), :]
                        #     label_train = data_label_train[np.stack(random_indices_augmented_train), :]
                            
                        #     neural_test = data_neural_train[np.stack(random_indices_augmented_test), :]
                        #     label_test = data_label_train[np.stack(random_indices_augmented_test), :]
                            
                        #     return neural_train, neural_test, label_train, label_test, random_indices
                            
                # split data
                with open('./GY-p2/data/data--daily-decoding.pkl', 'rb') as f:
                    len_for_each_session_trial, target_for_each_session_trial, neural_train, neural_test, label_train, label_test, random_indices= pickle.load(f)
                
                add_num = 0
                add_num_to = 0
                # neural_train_add, _, label_train_add, _ = split_data(all_micro_spikes_concat, all_macro_conditions_concat,
                #                                                      len_for_each_session_trial, Day_from + add_num,
                #                                                      Day_to + add_num_to)  # direction
                print(f"add_num={add_num},add_num_to={add_num_to}")
                
                # neural_train = torch.vstack([neural_train, neural_train_add])
                # label_train = torch.vstack([label_train, label_train_add])
                print('neural_train_length: ', len(neural_train))
                print('split data...finished!')
                
                # num_hidden_units = 64
                distance = 'euclidean'
                # distance = 'cosine'
                print('distance: ', distance)
                # model
                cl_dir_model = CEBRA(
                    model_architecture='offset10-model',
                    batch_size=512,
                    learning_rate=3e-4,
                    temperature=1,
                    output_dimension=output_dimension,
                    num_hidden_units=num_hidden_units,
                    max_iterations=max_iterations,
                    distance=distance,
                    device='cuda_if_available',
                    verbose=True
                )

                path__dimReduceModel = path__dimReduced + '/cl_dir_model_dim_' + distance + '_' + str(output_dimension) + '_' + str(max_iterations) + '_GY-p2_singleday_expect_bin_' + str(expect_num_bins) + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + str(test_Day) + today + '_' + data_from_to + '.pt'

                if bin_method == 'expect_bins':
                    if train_data == 'days':
                        if (not os.path.exists(path__dimReduceModel)) or always_rereducedim:
                            cl_dir_model.fit(neural_train, label_train)
                            cl_dir_model.save(path__dimReduceModel)
                            print('save model ...... finished!')
                
                ## ## Load the models and get the corresponding embeddings
                if bin_method == 'expect_bins':
                    if train_data == 'days':
                        cl_dir_model = cebra.CEBRA.load(path__dimReduceModel)
                        cebra_dir_train = cl_dir_model.transform(neural_train)
                        cebra_dir_test = cl_dir_model.transform(neural_test)
                print('Load the model ..... finished!')
                
                ##
                decoder_method = 'gru_test_8_aligned'
                print('decoder_method: ', decoder_method)
                import torch
                import torch.nn as nn
                import torch.optim as optim
                import numpy as np
                import torch.utils.data as Data
                
                # 定义超参数
                sequence_length = 50
                num_classes = 8
                input_size = output_dimension * num_classes  # dim of embedding output
                
                weather_together = True
                
                if weather_together:

                    week_all =  {
                        '1': 20221103, '2': 20221104, 
                        '3': 20221108, '4': 20221109, '5': 20221110, '6': 20221111, 
                        '7': 20221114, '8': 20221115, '9': 20221116, '10': 20221117, '11': 20221118, 
                        '12': 20221125, 
                        '13': 20221128, '14': 20221129, '15': 20221202, 
                        '16': 20221205, '17': 20221208, '18': 20221209, 
                        '19': 20221213, '20': 20221214, '21': 20221215, 
                        '22': 20221219, 
                        '23': 20221230, 
                        '24': 20230103, '25': 20230104, '26': 20230105, '27': 20230106, 
                        '28': 20230109, '29': 20230111, '30': 20230112, '31': 20230113, 
                        '32': 20230116, 
                        '33': 20230208, '34': 20230209, '35': 20230210, 
                        '36': 20230213, '37': 20230215, '38': 20230217, 
                        '39': 20230227, '40': 20230303, 
                        '41': 20230306, '42': 20230308, '43': 20230310, 
                        '44': 20230313, '45': 20230316, 
                        '46': 20230320, 
                        '47': 20230327, '48': 20230328, '49': 20230329, 
                        '50': 20230402, '51': 20230404, '52': 20230406, '53': 20230407, 
                        '54': 20230410, '55': 20230411, '56': 20230414, 
                        '57': 20230418, '58': 20230419, '59': 20230420, '60': 20230421, 
                        '61': 20230423, '62': 20230425, '63': 20230428, 
                        '64': 20230504, '65': 20230506, 
                        '66': 20230508, '67': 20230509, '68': 20230510, '69': 20230511, '70': 20230512, 
                        '71': 20230516, '72': 20230517, 
                        '73': 20230612, '74': 20230614, '75': 20230616, 
                        '76': 20230619, '77': 20230621, 
                        '78': 20230625, '79': 20230626, '80': 20230628, '81': 20230629, '82': 20230630, 
                        '83': 20230703, '84': 20230705, '85': 20230706, '86': 20230707, 
                        '87': 20230710, '88': 20230713, 
                        '89': 20230717, '90': 20230719, '91': 20230720, '92': 20230721, 
                        '93': 20230724, '94': 20230725, '95': 20230726, '96': 20230727, '97': 20230728, 
                        '98': 20230731, '99': 20230801, '100': 20230803, 
                        '101': 20230807, '102': 20230808, '103': 20230811, 
                        '104': 20230814, '105': 20230817, '106': 20230818, 
                        '107': 20230821, '108': 20230822, '109': 20230824, '110': 20230825, 
                        '111': 20230828, '112': 20230829, '113': 20230831, '114': 20230901, 
                        '115': 20230904, '116': 20230905, '117': 20230906, '118': 20230908, 
                        '119': 20230911,
                    }
                    week_name_from_to = str(week_all[str(Day_to)])
                    
                    print(f'0{week_name_from_to}')
                    
                    # Save embeddings in current folder
                    cebra_dir_train_stard_embeddings = dict()
                    
                    len_stand = int(len(label_train) * 0.5 // expect_num_bins) * expect_num_bins # 取训练的部分作为标准
                    # len_stand = len(label_train)  # 取训练的所有作为标准
                    cebra_dir_train_stard_embeddings['embeddings'] = cebra_dir_train[-len_stand:, :]
                    
                    # label_index = np.arange(len(label_train) - len_stand, len(label_train))
                    label_index = np.arange(0, len_stand)
                    id_target = [[] for _ in range(8)]
                    for i, value in enumerate(label_train[-len_stand:, :]):
                        id_target[int(value)].append(label_index[i])
                    for id in range(len(id_target)):
                        id_target[id] = np.stack(id_target[id])
                    cebra_dir_train_stard_embeddings['id_target'] = id_target
                    
                    # 训练部分对齐

                    path__alignedData_train = path__aligned + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_singleday_expect_bin_' + str(expect_num_bins) + '_data_aligned_method_' + str(align_method) + '_0' + week_name_from_to + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'

                    flag_align_target = 0
                    label_train_align = label_train[::expect_num_bins]
                    if (not os.path.exists(path__alignedData_train)) or always_realign:
                        print('staring data aligned......')
                        id_target_align_train = [[] for _ in range(8)]
                        id_l, id_r = 0, 0
                        images_train_align = torch.zeros((len(cebra_dir_train), input_size)).to(device)
                        for i in range(day[str(Day_from - 1)], day[str(Day_to)]):
                            id_target_align = [[] for _ in range(8)]
                            for j in range(len(len_for_each_session_trial[i])):
                                id_r += len_for_each_session_trial[i][j]
                                if id_r > len(cebra_dir_train):
                                    break
                                id_target_align[label_train_align[flag_align_target]].append(
                                    torch.arange(id_l, id_r)
                                )
                                flag_align_target += 1
                                id_l = id_r
                            
                            for h in range(len(id_target_align)):
                                if len(id_target_align[h]) == 0:
                                    continue
                                id_target_align[h] = torch.cat(id_target_align[h])
                                id_target_align_train[h].append(id_target_align[h])
                                for m in range(8):
                                    
                                    # print(f"i={i},j={j},h={h},m={m}")
                                    len_train = len(cebra_dir_train_stard_embeddings['embeddings'][cebra_dir_train_stard_embeddings['id_target'][m], :])
                                    len_test = len(id_target_align[h])
                                    if len_train >= len_test:
                                        nums = len_train // len_test
                                        temp_align = []
                                        images_align = []
                                        
                                        for k in range(nums):
                                            idx = torch.arange(len_train)[k * len_test:(k + 1) * len_test]
                                            
                                            temp_align.append(align_embedings_cross_days(
                                                torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][cebra_dir_train_stard_embeddings['id_target'][m], :][
                                                             idx, :]).double().to(device),
                                                torch.tensor(cebra_dir_train[id_target_align[h], :]).double().to(device)
                                            ))
                                        images_align.append(torch.stack(temp_align, dim=0).mean(0))
                                        images_train_align[id_target_align[h], m * output_dimension:(m + 1) * output_dimension] = \
                                            torch.stack(images_align, dim=0).mean(0).float()
                                    else:
                                        nums = len_test // len_train
                                        temp_align = []
                                        images_align = []
                                        
                                        for k in range(nums):
                                            idx = torch.arange(len_test)[k * len_train:(k + 1) * len_train]
                                            
                                            temp_align.append(align_embedings_cross_days(
                                                torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
                                                             cebra_dir_train_stard_embeddings['id_target'][m],
                                                             :]).double().to(
                                                    device),
                                                torch.tensor(
                                                    cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
                                            ))
                                        if len_test % len_train > 0:
                                            idx = torch.arange(len_test)[nums * len_train:]
                                            temp_align.append(align_embedings_cross_days(
                                                torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
                                                             cebra_dir_train_stard_embeddings['id_target'][m][
                                                             :len_test % len_train],
                                                             :]).double(
                                                ).to(device),
                                                torch.tensor(
                                                    cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
                                            ))
                                        images_align.append(torch.cat(temp_align, dim=0).mean(0))
                                        images_train_align[id_target_align[h],
                                        m * output_dimension:(m + 1) * output_dimension] = \
                                            torch.stack(images_align, dim=0).mean(0).float()
                        
                        images_train_align = images_train_align.cpu().data
                        torch.save(images_train_align.cpu().data, path__alignedData_train)
                    else:
                        images_train_align = torch.load(path__alignedData_train)
                    
                    print('train data aligned......finished!')

                    # 测试部分对齐

                    path__alignedData_test = path__aligned + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_singleday_expect_bin_' + str(expect_num_bins) + '_data_aligned_test_method_' + str(align_method) + '_0' + week_name_from_to + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'

                    if (not os.path.exists(path__alignedData_test)) or always_realign:
                        
                        # for idx in range(8):
                        # 	id_target_align_train[idx] = torch.cat(id_target_align_train[idx])
                        
                        id_l, id_r = len(neural_train), 0
                        s2 = 0
                        flag_test, flag_align_test_target = 0, 0
                        label_test_align = label_test[::expect_num_bins]
                        images_test_align = torch.zeros((len(cebra_dir_test), input_size)).to(device)
                        # id_target_align_test = [[] for _ in range(8)]
                        for i in range(day[str(test_Day - 1)], day[str(test_Day)]):
                            id_target_align_test = [[] for _ in range(8)]
                            for j in range(len(len_for_each_session_trial[i])):
                                id_r += len_for_each_session_trial[i][j]
                                if id_r > len(neural_train):
                                    flag_test = 1
                                    id_target_align_test[label_test_align[flag_align_test_target]].append(
                                        torch.arange(id_l, id_r)
                                    )
                                    flag_align_test_target += 1
                                    id_l = id_r
                            if flag_test == 0:
                                continue
                            for h in range(len(id_target_align_test)):
                                if len(id_target_align_test[h]) == 0:
                                    continue
                                # s2 += len(id_target_align_test[h])
                                id_target_align_test[h] = torch.cat(id_target_align_test[h]) - len(neural_train)
                                # id_target_align_test[h] = torch.cat(id_target_align_test[h])
                                for m in range(8):
                                    # print(f"i={i},j={j},h={h},m={m}")
                                    # len_train = len(images_train_align[id_target_align_train[h],
                                    #                 m * output_dimension:(m + 1) * output_dimension])
                                    len_train = len(cebra_dir_train_stard_embeddings['embeddings'][
                                                    cebra_dir_train_stard_embeddings['id_target'][m], :])
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
                                                             cebra_dir_train_stard_embeddings['id_target'][m], :][
                                                             idx, :]).double().to(device),
                                                torch.tensor(cebra_dir_test[id_target_align_test[h], :]).double().to(
                                                    device)
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
                                            temp_align.append(align_embedings_cross_days(torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][cebra_dir_train_stard_embeddings['id_target'][m], :]).double().to(device), torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)))
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
                                                             cebra_dir_train_stard_embeddings['id_target'][m][
                                                             :len_test % len_train],
                                                             :]).double().to(device),
                                                torch.tensor(
                                                    cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(
                                                    device)
                                            ))
                                        images_align.append(torch.cat(temp_align, dim=0).mean(0))
                                        images_test_align[id_target_align_test[h],
                                        m * output_dimension:(m + 1) * output_dimension] = \
                                            torch.stack(images_align, dim=0).mean(0).float()
                        
                        images_test_align = images_test_align.cpu().data
                        torch.save(images_test_align.cpu().data, path__alignedData_test)
                    
                    else:
                        images_test_align = torch.load(path__alignedData_test)
                    print('test data aligned......finished!')
                
                def split_data_for_day(data_spike_train, data_label):
                    split_day_data = []
                    split_day_label = []
                    # Save embeddings in current folder
                    len_stand = int(len(label_train) * 0.25 // expect_num_bins) * expect_num_bins  
                    for count in range(len(week_count_list) - 1):
                        day_data = data_spike_train[count * len_stand:(count + 1) * len_stand, :]
                        day_label = data_label[count * len_stand:(count + 1) * len_stand, :]
                        
                        split_day_data.append(day_data)
                        split_day_label.append(day_label)
                    
                    day_data = data_spike_train[(count + 1) * len_stand:, :]
                    day_label = data_label[(count + 1) * len_stand:, :]
                    split_day_data.append(day_data)
                    split_day_label.append(day_label)
                    
            
                    return split_day_data, split_day_label
                
                qy_train_data, qy_train_label = split_data_for_day(images_train_align.float(), label_train)
                
                qy_test_data = images_test_align.float()
                qy_test_label = label_test
                
                # 闹点数据加载
                # domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                
                # domains = torch.arange(0, 42, 1).tolist()
                # this_source_domain = [0, 2, 3, 4, 5]
                # this_target_domain = [1]
                args.source_domains = list(range(len(week_count_list)))
                
                # import pdb; pdb.set_trace()
                test_data = qy_test_data.reshape(-1, sequence_length, input_size)
                test_label = qy_test_label[::50].view(-1)
                # test_label = np.eye(8)[test_label].reshape(-1,8)
                print(test_data.shape)
                print(test_label.shape)
                
                target_train_dloader, target_test_dloader = get_domainnet_dloader_test(args.base_path, test_data, test_label, configs["TrainingConfig"]["batch_size"], args.workers, mix_length)
                
                train_dloaders.append(target_train_dloader)
                test_dloaders.append(target_test_dloader)
                test_model = DomainNet_encoder(time_window, 'resnet101', args.bn_momentum, False, False).cuda()
                print(count_parameters(test_model))
                print(count_parameters(
                    DomainNetClassifier(configs["ModelConfig"]["backbone"], 8, args.data_parallel).cuda()))
                # print(test_model.__dict__.keys())
                models.append(test_model)
                classifiers.append(
                    DomainNetClassifier(configs["ModelConfig"]["backbone"], 8, args.data_parallel).cuda())
                # domains.remove(this_target_domain)
                
                for num in range(len(week_count_list)):
                    # import pdb; pdb.set_trace()
                    domain_data = np.array(qy_train_data[num]).reshape(-1, sequence_length, input_size)
                    domain_label = qy_train_label[num][::50].view(-1)
                    # domain_label = np.eye(8)[domain_label].reshape[-1,8]
                    print(domain_data.shape)
                    print(domain_label.shape)
                    
                    source_train_dloader, source_test_dloader = get_domainnet_dloader_train(args.base_path, domain_data, domain_label, configs["TrainingConfig"]["batch_size"], args.workers)
                    train_dloaders.append(source_train_dloader)
                    test_dloaders.append(source_test_dloader)
                    models.append(
                        DomainNet_encoder(time_window, configs["ModelConfig"]["backbone"], args.bn_momentum, False, False).cuda())
                    classifiers.append(
                        DomainNetClassifier(configs["ModelConfig"]["backbone"], 8, args.data_parallel).cuda())
                num_classes = 8
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
#			writer_log_dir = path.join(args.base_path, configs["DataConfig"]["dataset"], "runs")
#			print("create writer in {}".format(writer_log_dir))
#			if os.path.exists(writer_log_dir):
#				shutil.rmtree(writer_log_dir, ignore_errors=True)
#			writer = SummaryWriter(log_dir=writer_log_dir)

            # begin train
            print(f"Begin turn {turn}")
            
            # create the initialized domain weight
            domain_weight = create_domain_weight(len(args.source_domains))
            # adjust training strategy with communication round
            batch_per_epoch, total_epochs = decentralized_training_strategy(
                communication_rounds=configs["UMDAConfig"]["communication_rounds"],
                epoch_samples=configs["TrainingConfig"]["epoch_samples"],
                batch_size=configs["TrainingConfig"]["batch_size"],
                total_epochs=configs["TrainingConfig"]["total_epochs"],
            )

            # train model

            train_acc_list = []
            acc_list = []
            train_loss_list = []
            test_loss_list = []

            for epoch in range(args.start_epoch, total_epochs):
                domain_weight, train_loss, _, _, _, _, _ = train(
                    train_dloaders, models, classifiers, optimizers, classifier_optimizers, epoch, 
                    num_classes=num_classes, 
                    model_savepath=None, 
                    turn=turn, 
                    testDay=test_Day,
                    domain_weight=domain_weight, 
                    source_domains=args.source_domains,
                    batch_per_epoch=batch_per_epoch, 
                    total_epochs=total_epochs,
                    batchnorm_mmd=configs["UMDAConfig"]["batchnorm_mmd"],
                    communication_rounds=configs["UMDAConfig"]["communication_rounds"],
                    confidence_gate_begin=configs["UMDAConfig"]["confidence_gate_begin"],
                    confidence_gate_end=configs["UMDAConfig"]["confidence_gate_end"],
                    malicious_domain=configs["UMDAConfig"]["malicious"]["attack_domain"],
                    attack_level=configs["UMDAConfig"]["malicious"]["attack_level"],
                    mix_aug=(configs["DataConfig"]["dataset"] != "AmazonReview"),
                )

                
                train_acc_list.append(0.0)		# maintainance substitution

                acc_top1, acc_top2, acc_top3, test_loss, _ = test(
                    args.target_domain, args.source_domains, test_dloaders, models, classifiers, epoch, total_epochs, today, test_Day, 
                    num_classes=num_classes, 
                    top_5_accuracy=True
                )
                
                acc_list.append([acc_top1 * 100, acc_top2 * 100, acc_top3 * 100, seeds])
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)

                for scheduler in optimizer_schedulers:
                    scheduler.step(epoch)
                for scheduler in classifier_optimizer_schedulers:
                    scheduler.step(epoch)
                # save models every 10 epochs
                if (epoch + 1) % 10 == 0:
                    # save target model with epoch, domain, model, optimizer
                    save_checkpoint(
                        {"epoch": epoch + 1,
                         "domain": args.target_domain,
                         "backbone": models[0].state_dict(),
                         "classifier": classifiers[0].state_dict(),
                         "optimizer": optimizers[0].state_dict(),
                         "classifier_optimizer": classifier_optimizers[0].state_dict()
                         },
                        filename="{}.pth.tar".format(args.target_domain))
                # if acc_top1 >= 1:
                # 	break

            ## graph acc curves

            import matplotlib.pyplot as plt

            path__decodedResults_acc_plt = path__decodedResults + path_subpath__modelName + '/acc' + '/outputs_acc_align_data_align_method_' + str(align_method) + '_0' + week_name_from_to + '_GY-p2_singleday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A.png'

            plt.plot(train_acc_list, label='Train Accuracy')
            plt.plot(np.array(acc_list)[:, 0], label='Test Accuracy')
            plt.title(f'Training and Test Accuracies Over Epochs, test day {test_Day}, turn {turn}, daily-decoding')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracies')
            plt.legend()
            plt.savefig(path__decodedResults_acc_plt)
            plt.close()

            ## find best result using unsupervised method

            # unsupervised_epoch = np.argmax(train_acc_list, axis=0)
            # unsupervised_acc = acc_list[unsupervised_epoch]


            last_acc = acc_list[-1]
            max_acc = np.max(np.array(acc_list), axis=0)
            print(path_subpath__modelName)
            print("Maximum accuracy", max_acc)
            print("Last epoch accuracy", last_acc)
            max_acc_list.append(max_acc)
            last_acc_list.append(last_acc)
            # unsupervised_acc_list.append(unsupervised_acc)

            ## graph loss curves

            import matplotlib.pyplot as plt

            path__decodedResults_loss = path__decodedResults + path_subpath__modelName + '/loss' + '/outputs_loss_align_data_align_method_' + str(align_method) + '_0' + week_name_from_to + '_GY-p2_singleday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A.png'

            plt.plot(train_loss_list, label='Train Loss')  # 绘制训练损失曲线
            plt.plot(test_loss_list, label='Test Loss')  # 绘制测试损失曲
            plt.title('Training and Test Losses Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(path__decodedResults_loss)
            plt.close()

        # # TODO: choose a method (and seperate folder)

        path__decodedResults_accMax = path__decodedResults + path_subpath__modelName + '/acc' + '/outputs_acc_align_data_align_method_' + str(align_method) + '_0' + week_name_from_to + '_GY-p2_singleday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A.npy'
        np.save(path__decodedResults_accMax, max_acc_list)

        path__decodedResults_accLast = path__decodedResults + path_subpath__modelName + '/acc' + '/outputs_acc_align_data_align_method_' + str(align_method) + '_0' + week_name_from_to + '_GY-p2_singleday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A_last.npy'
        np.save(path__decodedResults_accLast, last_acc_list)

        # path__decodedResults_accUnspv = path__decodedResults + path_subpath__modelName + '/acc' + '/outputs_acc_align_data_align_method_' + str(align_method) + '_0' + week_name_from_to + '_GY-p2_singleday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A_unspv.npy'
        # np.save(path__decodedResults_accUnspv, unsupervised_acc_list)
        
        print("\ntest day is:",test_Day)

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

        # print("=== UNSUPERVISED BEST ACCURACY ===")
        # print(np.stack(unsupervised_acc_list))
        # all_top_1 = np.stack(unsupervised_acc_list)[:, 0]
        # print("Sorted Accuracies:")
        # print(np.sort(all_top_1))
        # print("mean = ", np.sort(all_top_1)[-5:].mean())

        acc_of_seeds = np.mean(np.array(max_acc_list), axis=0)[:-1]
        print(acc_of_seeds)
        # logger.info(acc_of_seeds)
        acc_of_week.append(acc_of_seeds)

    print(path_subpath__modelName)
    print(acc_of_week)
    # logger.info(acc_of_week)

def save_checkpoint(state, filename):
    # args.train_time = 1
    # filefolder = "{}/{}/parameter".format(args.base_path, configs["DataConfig"]["dataset"])
    # if not path.exists(filefolder):
    #     os.makedirs(filefolder)
    # torch.save(state, path.join(filefolder, filename))
    pass


if __name__ == "__main__":
    print('start training')
    main()
    print('end training')
