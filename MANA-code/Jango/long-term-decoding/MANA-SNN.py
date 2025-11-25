import random

import numpy as np

import argparse
from model.MANASNN_Jango.domainnet import DomainNet_encoder, DomainNetClassifier
from train.train_SNN_Jango import *
from datasets.DomainNet import *
import os
import yaml
import cebra
from cebra import CEBRA

import pickle


def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parser = argparse.ArgumentParser(description=None)
# Dataset Parameters
parser.add_argument("--config", default="DomainNet_Jango.yaml")
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N', help='number of data loading workers (default: 8)')
# Train Strategy Parameters
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
# Optimizer Parameters
parser.add_argument('--optimizer', type=str, default="SGD", metavar="Optimizer Name")
parser.add_argument('-m', '--momentum', type=float, default=0.9, metavar='M', help='Momentum in SGD')
parser.add_argument('--wd', '--weight-decay', type=float, default=5e-4)
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument("--gpu", default="0", type=str, metavar='GPU plans to use', help='The GPU id plans to use')

parser.add_argument('--variable_num', type=int, default=6, help='number to test, from 6 to 20')
parser.add_argument('--mix_length', type=int, default=28)
parser.add_argument('--time_window', type=int, default=10)
parser.add_argument('--cross_week_num', type=float, default=0)
args = parser.parse_args()


target_domain = None
domainnet_base_path_obsolete = None


variable_num = args.variable_num
mix_length = args.mix_length
time_window = args.time_window
cross_week_num = args.cross_week_num
if cross_week_num == int(cross_week_num):
    cross_week_num = int(cross_week_num)


# import config files
with open(r"Jango/long-term-decoding/config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
print(" ")
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# cross week setup
from lib.utils.CrossMultiWeekDictionaries import *
# save decoding intermediates setup
from lib.utils.save_weight_hidden_state_condition import *


name__model_type = f'MANA-SNN'

print("Saving results to:", name__model_type)


path__loadData = 'Jango/data'
base_path = "Jango/long-term-decoding/"

path__intermediate_data = base_path + f"intermediate_data/{name__model_type}"              # 存放中间数据的路径
path__dimReduced = path__intermediate_data + '/dim_reduced'
path__aligned = path__intermediate_data + '/multi_aligned'
path__aligned_unaligned = path__intermediate_data + '/aligned_unaligned'

path__results = base_path + f'res/{name__model_type}'                                      # 存放最终结果的路径
path__save_loss = path__results + '/loss'

path__save_w_hs = base_path + save_weight_hidden_state__rel_path + f"/{name__model_type}"   # 存放weight和hidden state的路径


always_rereducedim = True
always_realign = True

for path in [path__dimReduced, path__aligned, path__aligned_unaligned, path__results, path__save_loss, path__save_w_hs]:
    os.makedirs(path, exist_ok=True)



seed_set = np.array([31, 44, 46, 50, 54])

def main(args=args, configs=configs):
    import numpy as np
    import torch
    
    data_from_to = '20150730_20151102' # all days

    device = torch.device("cuda:0")

    align_method = 2
    

    def align_embedings_cross_days(cebra_pos_train, cebra_pos_test):
        
        cebra_pos_train_sample = cebra_pos_train
        
        # torch
        Q_train, R_train = torch.linalg.qr(cebra_pos_train_sample)
        Q_test, R_test = torch.linalg.qr(cebra_pos_test)
        U, S, V = torch.linalg.svd(Q_train.T @ Q_test)
        V = V.T
        cebra_pos_test_align = Q_train @ U @ torch.linalg.pinv(V) @ R_test
        
        return cebra_pos_test_align


    ##
    bin_method = 'expect_bins'
    expect_num_bins = 30
        
    all_micro_spikes_concat = torch.load(path__loadData + '/all_micro_spikes_expect_bin_' + str(expect_num_bins) + '_Jango_only_' + data_from_to + '.pt')

    all_macro_positions_concat = torch.load(path__loadData + '/all_macro_positions_expect_' + str(expect_num_bins) + '_Jango_only_' + data_from_to + '.pt')

    all_macro_conditions_concat = torch.load(path__loadData + '/all_macro_conditions_expect_' + str(expect_num_bins) + '_Jango_only_' + data_from_to + '.pt')

    len_for_each_session_trial = torch.load(path__loadData + '/len_for_each_session_trial_expect_' + str(expect_num_bins) + '_Jango_only_' + data_from_to + '.pt')

    target_for_each_session_trial = torch.load(path__loadData + '/target_for_each_session_trial_expect_' + str(expect_num_bins) + '_Jango_only_' + data_from_to + '.pt')


    if cross_week_num == 0: # 固定训练前5天
        Day_from, Day_to = 1, 5
        test_Day_list = [variable_num]  
        week_count_list = [1, 1, 1, 1]
    elif cross_week_num == 1: # 跨周训练
        pass
    else:
        raise ValueError(f"cross_week_num {cross_week_num} not implemented.")

    day = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
    }
    week_all = {
        '1': 20150730, '2': 20150731, '3': 20150801, '4': 20150805, '5': 20150806, 
        '6': 20150807, '7': 20150808, '8': 20150809, '9': 20150820, '10': 20150824,
		'11': 20150825, '12': 20150826, '13': 20150827, '14': 20150828, '15': 20150831, 
        '16': 20150905, '17': 20150906, '18': 20150908, '19': 20151029, '20': 20151102,
    }

    for test_count in range(len(test_Day_list)):
        test_Day = test_Day_list[test_count]

        condition = save_weight_hidden_state__condition()
        if condition.check_test_day(test_Day):
            print(f"test day {test_Day} will be saved")


        max_acc_list = []
        last_acc_list = []


        for turn in range(len(seed_set)):

            os.makedirs(path__aligned_unaligned + f'/turn_{turn}', exist_ok=True)

            # build dataset
            if configs["DataConfig"]["dataset"] == "DomainNet":
                seeds = int(seed_set[turn])
                random.seed(seeds)
                np.random.seed(seeds)
                torch.manual_seed(seeds)
                torch.cuda.manual_seed_all(seeds)
                
                torch.backends.cudnn.deterministic = True
                
                print(' ')
                print('turn: ', turn)
                print(' ')
                today = f'20251028_turn_{turn}'
                
                max_iterations = 10000 
                output_dimension, num_hidden_units = 36, 64

                train_data = 'days'
                
                print('------------------Day_from_{}_to_{}------------------'.format(Day_from, Day_to))
                print('--------------------test_Day: {}------------------'.format(test_Day))
                
                
                def split_data(data_spike_train, data_label, data_position, len_for_each_session_trial, Day_from, Day_to):
                    
                    split_idx_start_beg = 0
                    for i in range(day[str(Day_from - 1)]):
                        split_idx_start_beg += sum(len_for_each_session_trial[i])
                    split_idx_start_end = 0
                    for i in range(day[str(Day_to)]):
                        split_idx_start_end += sum(len_for_each_session_trial[i])
                    
                    split_idx_start = 0
                    for i in range(day[str(test_Day - 1)]):
                        split_idx_start += sum(len_for_each_session_trial[i])
                    split_idx_end = 0  # 只预测第test_Day这一天的
                    for i in range(day[str(test_Day)]):
                        split_idx_end += sum(len_for_each_session_trial[i])
                    
                    neural_train = data_spike_train[split_idx_start_beg:split_idx_start_end, :]
                    label_train = data_label[split_idx_start_beg:split_idx_start_end, :]
                    pos_train = data_position[split_idx_start_beg:split_idx_start_end, :]
                    
                    neural_test = data_spike_train[split_idx_start:split_idx_end, :]
                    label_test = data_label[split_idx_start:split_idx_end, :]
                    pos_test = data_position[split_idx_start:split_idx_end, :]
                    return neural_train, neural_test, label_train, label_test, pos_train, pos_test
                
                # split data
                neural_train, neural_test, label_train, label_test, pos_train, pos_test = split_data(
                                                                                all_micro_spikes_concat,
                                                                                all_macro_conditions_concat,
                                                                                all_macro_positions_concat,
                                                                                len_for_each_session_trial, Day_from,
                                                                                Day_to)  # direction
                
                # normalization
                pos_train_min = pos_train.min()
                pos_train_max = pos_train.max()
                pos_train_range = pos_train_max - pos_train_min

                pos_train = 2 * (pos_train - pos_train_min) / (pos_train_range + 1e-8) - 1
                pos_test = 2 * (pos_test - pos_train_min) / (pos_train_range + 1e-8) - 1

                norm_params = {
                    'min': pos_train_min,
                    'max': pos_train_max,
                    'range': pos_train_range
                }
                torch.save(norm_params, path__results + f'/norm_params_Day_{Day_from}_to_{Day_to}.pt')

                add_num = 0
                add_num_to = 0
                
                distance = 'euclidean'
                cl_dir_model = CEBRA(model_architecture='offset10-model',
                                    batch_size=512,
                                    learning_rate=3e-4,
                                    temperature=1,
                                    output_dimension=output_dimension,
                                    num_hidden_units=num_hidden_units,
                                    max_iterations=max_iterations,
                                    distance=distance,
                                    device='cuda_if_available',
                                    verbose=True,
                                    )

                path__dimReduceModel = path__dimReduced + '/cl_dir_model_dim_' + distance + '_' + str(output_dimension) + '_' + str(max_iterations) + '_Jango_crossday_expect_bin_' + str(expect_num_bins) + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + str(test_Day) + '_' + today + '_' + data_from_to + '.pt'

                if bin_method == 'expect_bins':
                    if train_data == 'days':
                        if not (os.path.exists(path__dimReduceModel)) or always_rereducedim:
                            cl_dir_model.fit(neural_train, pos_train)
                            
                            cl_dir_model.save(path__dimReduceModel)
                            print('save model ...... finished!')
                
                ## ## Load the models and get the corresponding embeddings
                if bin_method == 'expect_bins':
                    if train_data == 'days':
                        cl_dir_model = cebra.CEBRA.load(path__dimReduceModel)
                        cebra_dir_train = cl_dir_model.transform(neural_train)
                        cebra_dir_test = cl_dir_model.transform(neural_test)
                print('Load the model ..... finished!')

                sequence_length = 30

                print(cebra_dir_test.shape, label_test.shape, pos_test.shape)

                unaligned_data_dict = {
                    0: (
                        torch.tensor(cebra_dir_test).reshape(-1, sequence_length, output_dimension).clone(), 
                        label_test.reshape(-1, sequence_length, 1).clone(), 
                        pos_test.reshape(-1, sequence_length, 2).clone(),
                    ),
                }
                
                with open(path__aligned_unaligned + f'/turn_{turn}/unaligned_Jango-test_{variable_num}.pkl', 'wb') as f:
                    pickle.dump(unaligned_data_dict, f)
                
                ##
                decoder_method = 'gru_test_8_aligned'
                print('decoder_method: ', decoder_method)
                import torch
                import torch.nn as nn
                import torch.optim as optim
                import numpy as np
                import torch.utils.data as Data
                		
                sequence_length = 30
                num_classes = 8
                input_size = output_dimension * num_classes  # dim of embedding output
                
                weather_together = True
                
                if weather_together:

                    week_name_from_to = str(week_all[str(Day_from)]) + '_' + str(week_all[str(Day_to)]) # 选择某一段时间作为文件名标识

                    with open(path__loadData + '/cl_dir_train_stard_embeddings_' + week_name_from_to + '_Jango.pkl', 'rb') as f:
                        cebra_dir_train_stard_embeddings = pickle.load(f)
                        print('load standard embeddings ...... finished!')

                    path__alignedData_train = path__aligned + '/align_dir_' + decoder_method + '_distance_' + distance + '_Jango_crossday_expect_bin_' + str(expect_num_bins) + '_data_aligned_method_' + str(align_method) + '_' + week_name_from_to + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + today + '_' + data_from_to + '.pt'

                    if (not os.path.exists(path__alignedData_train)) or always_realign:
                        print('staring data aligned......')
                        id_target_align_train = [[] for _ in range(8)]
                        id_l, id_r = 0, 0
                        images_train_align = torch.zeros((len(cebra_dir_train), input_size)).to(device)
                        for i in range(day[str(Day_from - 1)], day[str(Day_to)]):
                            id_target_align = [[] for _ in range(8)]
                            for j in range(len(len_for_each_session_trial[i])):
                                id_r += len_for_each_session_trial[i][j]
                                id_target_align[target_for_each_session_trial[i][j]].append(
                                    torch.arange(id_l, id_r)
                                )
                                id_l = id_r
                            
                            for h in range(len(id_target_align)):
                                id_target_align[h] = torch.cat(id_target_align[h])
                                id_target_align_train[h].append(id_target_align[h])
                                for m in range(8):
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
                                                            cebra_dir_train_stard_embeddings['id_target'][m], :]).double().to(
                                                    device),
                                                torch.tensor(cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
                                            ))
                                        if len_test % len_train > 0:
                                            idx = torch.arange(len_test)[nums * len_train:]
                                            temp_align.append(align_embedings_cross_days(
                                                torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
                                                            cebra_dir_train_stard_embeddings['id_target'][m][
                                                            :len_test % len_train],
                                                            :]).double(
                                                ).to(device),
                                                torch.tensor(cebra_dir_train[id_target_align[h], :][idx, :]).double().to(device)
                                            ))
                                        images_align.append(torch.cat(temp_align, dim=0).mean(0))
                                        images_train_align[id_target_align[h], m * output_dimension:(m + 1) * output_dimension] = \
                                            torch.stack(images_align, dim=0).mean(0).float()
                        
                        images_train_align = images_train_align.cpu().data
                        torch.save(images_train_align.cpu().data, path__alignedData_train)
                    else:
                        images_train_align = torch.load(path__alignedData_train)
                    
                    print('train data aligned......finished!')


                    path__alignedData_test = path__aligned + '/align_dir_' + decoder_method + '_distance_' + distance + '_Jango_crossday_expect_bin_' + str(expect_num_bins) + '_data_aligned_test_method_' + str(align_method) + '_' + week_name_from_to + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + today + '_' + data_from_to + '.pt'

                    if (not os.path.exists(path__alignedData_test)) or always_realign:
                        
                        id_l, id_r = 0, 0
                        images_test_align = torch.zeros((len(cebra_dir_test), input_size)).to(device)
                        for i in range(day[str(test_Day - 1)], day[str(test_Day)]):
                            id_target_align_test = [[] for _ in range(8)]
                            for j in range(len(len_for_each_session_trial[i])):
                                id_r += len_for_each_session_trial[i][j]
                                id_target_align_test[target_for_each_session_trial[i][j]].append(
                                    torch.arange(id_l, id_r)
                                )
                                id_l = id_r
                            
                            for h in range(len(id_target_align_test)):
                                id_target_align_test[h] = torch.cat(id_target_align_test[h])
                                for m in range(8):
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
                                            
                                            temp_align.append(align_embedings_cross_days(
                                                torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
                                                            cebra_dir_train_stard_embeddings['id_target'][m], :][
                                                            idx, :]).double().to(device),
                                                torch.tensor(cebra_dir_test[id_target_align_test[h], :]).double().to(device)
                                            ))
                                        images_align.append(torch.stack(temp_align, dim=0).mean(0))
                                        images_test_align[id_target_align_test[h], m * output_dimension:(m + 1) * output_dimension] = \
                                            torch.stack(images_align, dim=0).mean(0).float()
                                    else:
                                        nums = len_test // len_train
                                        temp_align = []
                                        images_align = []
                                        for k in range(nums):
                                            idx = torch.arange(len_test)[k * len_train:(k + 1) * len_train]
                                            
                                            temp_align.append(align_embedings_cross_days(
                                                torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
                                                            cebra_dir_train_stard_embeddings['id_target'][m], :]).double().to(device),
                                                torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
                                            ))
                                        if len_test % len_train > 0:
                                            idx = torch.arange(len_test)[nums * len_train:]
                                            
                                            temp_align.append(align_embedings_cross_days(
                                                torch.tensor(cebra_dir_train_stard_embeddings['embeddings'][
                                                            cebra_dir_train_stard_embeddings['id_target'][m][:len_test % len_train],
                                                            :]).double().to(device),
                                                torch.tensor(cebra_dir_test[id_target_align_test[h], :][idx, :]).double().to(device)
                                            ))
                                        images_align.append(torch.cat(temp_align, dim=0).mean(0))
                                        images_test_align[id_target_align_test[h], m * output_dimension:(m + 1) * output_dimension] = \
                                            torch.stack(images_align, dim=0).mean(0).float()
                        
                        images_test_align = images_test_align.cpu().data
                        torch.save(images_test_align.cpu().data, path__alignedData_test)
                    
                    else:
                        images_test_align = torch.load(path__alignedData_test)
                    print('test data aligned......finished!')
                
                def split_data_for_day(data_spike_train, data_label, data_position, len_for_each_session_trial=None, Day_from=None):
                    # evenly split for each model
                    split_day_data = []
                    split_day_label = []
                    split_day_pos = []
                    # Save embeddings in current folder
                    len_stand = int(len(label_train) * 0.25 // expect_num_bins) * expect_num_bins  
                    for count in range(len(week_count_list) - 1):
                        day_data = data_spike_train[count * len_stand:(count + 1) * len_stand, :]
                        day_label = data_label[count * len_stand:(count + 1) * len_stand, :]
                        day_pos = data_position[count * len_stand:(count + 1) * len_stand, :]
                        
                        split_day_data.append(day_data)
                        split_day_label.append(day_label)
                        split_day_pos.append(day_pos)
                    
                    day_data = data_spike_train[(count + 1) * len_stand:, :]
                    day_label = data_label[(count + 1) * len_stand:, :]
                    day_pos = data_position[(count + 1) * len_stand:, :]

                    split_day_data.append(day_data)
                    split_day_label.append(day_label)
                    split_day_pos.append(day_pos)
                    
                    return split_day_data, split_day_label, split_day_pos
                    
                qy_train_data, qy_train_label, qy_train_pos = split_data_for_day(images_train_align.float(), label_train, pos_train)
                
                qy_test_data = images_test_align.float()
                qy_test_label = label_test
                qy_test_pos = pos_test

                args.source_domains = list(range(len(week_count_list)))

                aligned_data_dict = {
                    0: (
                        qy_test_data.reshape(-1, sequence_length, input_size).clone(), 
                        qy_test_label.reshape(-1, sequence_length, 1).clone(), 
                        qy_test_pos.reshape(-1, sequence_length, 2).clone(),
                    ),
                }
                
                with open(path__aligned_unaligned + f'/turn_{turn}/aligned_Jango-test_{variable_num}-turn_{turn}.pkl', 'wb') as f:
                    pickle.dump(aligned_data_dict, f)

                test_data = qy_test_data.reshape(-1, sequence_length, input_size)
                test_label = qy_test_label[::sequence_length].view(-1)
                test_pos = qy_test_pos.reshape(-1, sequence_length, 2)


                # combine mix_length data into train set, with correct labels and positions
                mix_length_tuple, test_tuple = get_domainnet_dloader_test(domainnet_base_path_obsolete, test_data, test_label, test_pos, configs["TrainingConfig"]["batch_size"], args.workers, mix_length, output_data=True)

                qy_train_data.append(mix_length_tuple[0].reshape(-1, input_size))
                qy_train_label.append(mix_length_tuple[1].reshape(-1, 1).repeat(sequence_length, 1).reshape(-1, 1))
                qy_train_pos.append(mix_length_tuple[2].reshape(-1, 2))

                train_data = np.concatenate(qy_train_data, axis=0).reshape(-1, sequence_length, input_size)
                train_label = np.concatenate(qy_train_label, axis=0)[::sequence_length].reshape(-1, 1)
                train_pos = np.concatenate(qy_train_pos, axis=0).reshape(-1, sequence_length, 2)

                test_data = test_tuple[0].reshape(-1, sequence_length, input_size)
                test_label = test_tuple[1].reshape(-1, 1)
                test_pos = test_tuple[2].reshape(-1, sequence_length, 2)

                Transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                train_dataset = CustomDataset(train_data, train_label, train_pos, transform=Transform)
                test_dataset = CustomDataset(test_data, test_label, test_pos, transform=Transform)
                
                train_dloader = DataLoader(train_dataset, batch_size=configs["TrainingConfig"]["batch_size"], num_workers=args.workers, pin_memory=True, shuffle=True)
                test_dloader = DataLoader(test_dataset, batch_size=1, num_workers=args.workers, pin_memory=True, shuffle=False)
    
            else:
                raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))
            
                                      
            print("Begin")
            print(" ")

            unique_model = DomainNet_encoder(time_window, 'resnet101', args.bn_momentum, False, False).cuda()
            unique_classifier = DomainNetClassifier(configs["ModelConfig"]["backbone"], 60, args.data_parallel).cuda()
            model_optimizer = torch.optim.SGD(unique_model.parameters(), momentum=args.momentum, lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
            classifier_optimizer = torch.optim.SGD(unique_classifier.parameters(), momentum=args.momentum, lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd)
            model_optimizer_scheduler = CosineAnnealingLR(model_optimizer, configs["TrainingConfig"]["total_epochs"], eta_min=configs["TrainingConfig"]["learning_rate_end"])
            classifier_optimizer_scheduler = CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"], eta_min=configs["TrainingConfig"]["learning_rate_end"])
            total_epochs = total_epochs=configs["TrainingConfig"]["total_epochs"]
            

            # train model

            train_acc_list = []
            test_acc_list = []
            train_loss_list = []
            test_loss_list = []

            current_max_r2 = 0 

            for epoch in range(args.start_epoch, total_epochs):

                train_loss, unique_model, unique_classifier, model_optimizer, classifier_optimizer = train(
                    train_dloader, unique_model, unique_classifier, model_optimizer, classifier_optimizer, 
                    epoch=epoch, total_epochs=total_epochs,
                    turn=turn, 
                    Day_from=Day_from, Day_to=Day_to, testDay=test_Day,
                    save_whs__path = path__save_w_hs,
                    save_whs__condition = condition,
                )

                train_loss_list.append(train_loss)
                train_acc_list.append(0.0)		        # false train acc placeholder
                

                import time

                time__test_start = time.time()
                r2_avg, r2_x, r2_y, test_loss, _  = test(
                    test_dloader, unique_model, unique_classifier, epoch,
                    turn=turn, 
                    Day_from=Day_from, Day_to=Day_to, testDay=test_Day,
                    save_whs__path = path__save_w_hs,
                    save_whs__condition = condition,
                    current_max_acc = current_max_r2,
                )
                time__test_end = time.time()
                time__test = time__test_end - time__test_start
                
                current_max_r2 = max(r2_avg, current_max_r2)
                test_acc_list.append([r2_avg, r2_x, r2_y, seeds])  # 不乘100，R²范围是[0,1]

                test_loss_list.append(test_loss)

                model_optimizer_scheduler.step(epoch)
                classifier_optimizer_scheduler.step(epoch)

                # save models every 10 epochs
                if (epoch + 1) % 10 == 0:
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "domain": target_domain,
                            "backbone": unique_model.state_dict(),
                            "classifier": unique_classifier.state_dict(),
                            "optimizer": model_optimizer.state_dict(),
                            "classifier_optimizer": classifier_optimizer.state_dict()
                        },
                        filename="{}.pth.tar".format(target_domain)
                    )

            last_acc = test_acc_list[-1]
            max_acc = np.max(np.array(test_acc_list), axis=0)
            print(name__model_type)
            print("Maximum accuracy", max_acc)
            print("Last epoch accuracy", last_acc)
            max_acc_list.append(max_acc)
            last_acc_list.append(last_acc)

        # save loss
        loss_file_name = f'/outputs_loss_Jango_crossweek_{cross_week_num}__Day_from_{Day_from}_to_{Day_to}_testDay_{test_Day}'

        path__res_loss_train = path__save_loss + loss_file_name + '_train.npy'
        np.save(path__res_loss_train, train_loss_list)
        path__res_loss_test = path__save_loss + loss_file_name + '_test.npy'
        np.save(path__res_loss_test, test_loss_list)

        acc_file_name = f'/outputs_r2_Jango_crossweek_{cross_week_num}__Day_from_{Day_from}_to_{Day_to}_testDay_{test_Day}'

        path__res_accMax = path__results + acc_file_name + '.npy'
        np.save(path__res_accMax, max_acc_list)
        path__res_accLast = path__results + acc_file_name + '_last.npy'
        np.save(path__res_accLast, last_acc_list)
        
        print(" ")

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(" ")

        print(f"cross {cross_week_num} week")
        print(f"train {Day_from} - {Day_to} ==> test {test_Day}")
        print(" ")

        seed_selection = list(range(len(seed_set)))

        print("=== MAXIMUM R² SCORE ===")
        selected_max_acc = np.stack(max_acc_list)[seed_selection, :]
        print("R²(avg) | R²(x) | R²(y) | Seed")
        print(selected_max_acc)
        print("mean R²(avg) =", selected_max_acc[:, 0].mean())
        print("mean R²(x) =", selected_max_acc[:, 1].mean())
        print("mean R²(y) =", selected_max_acc[:, 2].mean())

        print(" ")

        print("=== LAST EPOCH R² SCORE ===")
        selected_last_acc = np.stack(last_acc_list)[seed_selection, :]
        print("R²(avg) | R²(x) | R²(y) | Seed")
        print(selected_last_acc)
        print("mean R²(avg) =", selected_last_acc[:, 0].mean())
        print("mean R²(x) =", selected_last_acc[:, 1].mean())
        print("mean R²(y) =", selected_last_acc[:, 2].mean())

    print("Result folder:", path__results)

def save_checkpoint(state, filename):
    pass


if __name__ == "__main__":
    print('start training')
    main()
    print('end training')