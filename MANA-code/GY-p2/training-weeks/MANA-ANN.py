import random

import numpy as np

import argparse
from model.MANAANN.domainnet import DomainNet_encoder, DomainNetClassifier
from lib.utils.federated_utils import *
from train.train_ANN import train, test
from datasets.DomainNet import *
import os
import yaml
import cebra
from cebra import CEBRA

import pickle


def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) \

parser = argparse.ArgumentParser(description=None)
# Dataset Parameters
parser.add_argument("--config", default="DomainNet.yaml")
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

parser.add_argument('--variable_num', type=int, default=24)
parser.add_argument('--mix_length', type=int, default=28)
parser.add_argument('--time_window', type=int, default=10)
parser.add_argument('--cross_week_num', type=float, default=1)
parser.add_argument('--train_weeks', type=int, default=4, help='number of weeks used for training')
args = parser.parse_args()

target_domain = None
domainnet_base_path_obsolete = None

variable_num = args.variable_num
mix_length = args.mix_length
time_window = args.time_window
cross_week_num = args.cross_week_num
train_weeks = args.train_weeks
if cross_week_num == int(cross_week_num):
    cross_week_num = int(cross_week_num)

# import config files
with open(r"GY-p2/training-weeks/config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# cross week setup
from lib.utils.CrossMultiWeekDictionaries import *
# save decoding intermediates setup
from lib.utils.save_weight_hidden_state_condition import *

save_domain = save_weight_hidden_state__domain


name__model_type = f'MANA-ANN-train_{train_weeks}_weeks'


path__loadData = 'GY-p2/data'
base_path = "GY-p2/training-weeks"

path__intermediate_data = base_path + f"/intermediate_data/{name__model_type}"
path__dimReduced = path__intermediate_data + '/dim_reduced'
path__aligned = path__intermediate_data + '/multi_aligned'

path__results = base_path + f'/res/{name__model_type}'
path__save_domain_weight = path__results + '/domain_weight'
path__save_loss = path__results + '/loss'

path__save_w_hs = base_path + save_weight_hidden_state__rel_path


always_rereducedim = True
always_realign = True

for path in [path__dimReduced, path__aligned, path__results, path__save_domain_weight, path__save_loss, path__save_w_hs]:
    os.makedirs(path, exist_ok=True)

seed_set = np.array([31, 44, 46, 54, 59])

def main(args=args, configs=configs):
    import numpy as np
    import torch
    
    data_from_to = '20221103_20230911'
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
    expect_num_bins = 50

    # if bin_method == 'expect_bins':

    #     all_micro_spikes_concat = torch.load('/home/hanly/Multiscale_SNN_Code/cerba_proc/GY-p2_20221101_20230911/all_micro_spikes_expect_bin_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
    #     all_macro_conditions_concat = torch.load('/home/hanly/Multiscale_SNN_Code/cerba_proc/GY-p2_20221101_20230911/all_macro_conditions_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
        
    #     len_for_each_session_trial = torch.load('/home/hanly/Multiscale_SNN_Code/cerba_proc/GY-p2_20221101_20230911/len_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
    #     target_for_each_session_trial = torch.load('/home/hanly/Multiscale_SNN_Code/cerba_proc/GY-p2_20221101_20230911/target_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
        
    #     # 1-8 ==> 0-7
    #     all_macro_conditions_concat = all_macro_conditions_concat - torch.tensor(1)
    #     for i in range(len(target_for_each_session_trial)):
    #         target_for_each_session_trial[i] = (np.array(target_for_each_session_trial[i]) - 1).tolist()

    verify_day_in_test_range(variable_num, cross_week_num, train_weeks)
    domain_dic, day_from_to_dic = generate_domain_dict_and_day_from_to_dict(cross_week_num, train_weeks, method="domains")


    Day_from, Day_to = day_from_to_dic[str(variable_num)]

    test_Day_list = [variable_num]  
    week_count_list = domain_dic[str(variable_num)]

    acc_of_week = []

    for test_count in range(len(test_Day_list)):
        test_Day = test_Day_list[test_count]

        condition = save_weight_hidden_state__condition()
        if condition.check_test_day(test_Day):
            print(f"test day {test_Day} will be saved")


        max_acc_list = []
        last_acc_list = []

        for turn in range(5):

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
                
                today = f'20240312-v2-p2-crossday_h1-64_h2-3072_turn_{turn}' # v1, v2
                
                max_iterations = 10000 
                output_dimension, num_hidden_units = 36, 64

                train_data = 'days'
                
                print('------------------Day_from_{}_to_{}------------------'.format(Day_from, Day_to))
                print('--------------------test_Day: {}------------------'.format(test_Day))
                
                
                # def split_data(data_spike_train, data_label, len_for_each_session_trial, Day_from, Day_to):
                    
                #     split_idx_start_beg = 0
                #     for i in range(day[str(Day_from - 1)]):
                #         split_idx_start_beg += sum(len_for_each_session_trial[i])
                #     split_idx_start_end = 0
                #     for i in range(day[str(Day_to)]):
                #         split_idx_start_end += sum(len_for_each_session_trial[i])
                    
                #     split_idx_start = 0
                #     for i in range(day[str(test_Day - 1)]):
                #         split_idx_start += sum(len_for_each_session_trial[i])
                #     split_idx_end = 0  # 只预测第test_Day这一天的
                #     for i in range(day[str(test_Day)]):
                #         split_idx_end += sum(len_for_each_session_trial[i])
                    
                #     neural_train = data_spike_train[split_idx_start_beg:split_idx_start_end, :]
                #     label_train = data_label[split_idx_start_beg:split_idx_start_end, :]
                    
                #     neural_test = data_spike_train[split_idx_start:split_idx_end, :]
                #     label_test = data_label[split_idx_start:split_idx_end, :]
                #     return neural_train, neural_test, label_train, label_test
                
                # # split data
                # neural_train, neural_test, label_train, label_test = split_data(all_micro_spikes_concat,
                #                                                                 all_macro_conditions_concat,
                #                                                                 len_for_each_session_trial, Day_from,
                #                                                                 Day_to)  # direction

                # with open(f'GY-p2/data/data--training-weeks-{train_weeks}.pkl', 'wb') as f:
                #     pickle.dump((len_for_each_session_trial, target_for_each_session_trial, neural_train.clone(), neural_test.clone(), label_train.clone(), label_test.clone()), f)


                # TODO: 如果是4，读long-term-decoding的数据

                with open(f'GY-p2/data/data--training-weeks-{train_weeks}.pkl', 'rb') as f:
                    len_for_each_session_trial, target_for_each_session_trial, neural_train, neural_test, label_train, label_test = pickle.load(f)
                
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

                path__dimReduceModel = path__dimReduced + '/cl_dir_model_dim_' + distance + '_' + str(output_dimension) + '_' + str(max_iterations) + '_GY-p2_crossday_expect_bin_' + str(expect_num_bins) + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + str(test_Day) + today + '_' + data_from_to + '.pt'

                if bin_method == 'expect_bins':
                    if train_data == 'days':
                        if not (os.path.exists(path__dimReduceModel)) or always_rereducedim:
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
                
                sequence_length = 50
                num_classes = 8
                input_size = output_dimension * num_classes  # dim of embedding output
                
                weather_together = True
                
                if weather_together:

                    week_name_from_to = str(week_all[str(Day_to)])

                    with open(path__loadData + '/cl_dir_train_stard_embeddings_' + week_name_from_to + '_gy-p2.pkl', 'rb') as f:
                        cebra_dir_train_stard_embeddings = pickle.load(f)
                        print('load standard embeddings ...... finished!')
                    
                    path__alignedData_train = path__aligned + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_crossday_expect_bin_' + str(expect_num_bins) + '_data_aligned_method_' + str(align_method) + '_0' + week_name_from_to + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'

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

                    path__alignedData_test = path__aligned + '/align_dir_' + decoder_method + '_distance_' + distance + '_GY-p2_crossday_expect_bin_' + str(expect_num_bins) + '_data_aligned_test_method_' + str(align_method) + '_0' + week_name_from_to + '_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + today + '_together_' + data_from_to + '.pt'

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
                                        images_test_align[id_target_align_test[h],
                                        m * output_dimension:(m + 1) * output_dimension] = \
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
                                        images_test_align[id_target_align_test[h],
                                        m * output_dimension:(m + 1) * output_dimension] = \
                                            torch.stack(images_align, dim=0).mean(0).float()
                        
                        images_test_align = images_test_align.cpu().data
                        torch.save(images_test_align.cpu().data, path__alignedData_test)
                    
                    else:
                        images_test_align = torch.load(path__alignedData_test)
                    print('test data aligned......finished!')
                
                def split_data_for_day(data_spike_train, data_label, len_for_each_session_trial, Day_from):
                    split_day_data = []
                    split_day_label = []
                    flag = 0
                    count_left = 0
                    print(f"week_count_list = {week_count_list}")
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
                    
                qy_train_data, qy_train_label = split_data_for_day(images_train_align.float(), label_train, len_for_each_session_trial, Day_from)
                
                qy_test_data = images_test_align.float()
                qy_test_label = label_test

                args.source_domains = list(range(len(week_count_list)))

                test_data = qy_test_data.reshape(-1,sequence_length,input_size)
                test_label = qy_test_label[::50].view(-1)

                target_train_dloader ,target_test_dloader = get_domainnet_dloader_test(domainnet_base_path_obsolete, test_data, test_label, configs["TrainingConfig"]["batch_size"], args.workers, mix_length)

                train_dloaders.append(target_train_dloader)
                test_dloaders.append(target_test_dloader)
                test_model = DomainNet_encoder(time_window, 'resnet101', args.bn_momentum, False, False).cuda()
                print(count_parameters(test_model))
                print(count_parameters(DomainNetClassifier(configs["ModelConfig"]["backbone"], 8, args.data_parallel).cuda()))
                models.append(test_model)
                classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 8, args.data_parallel).cuda())

                for num in range(len(week_count_list)):
                    domain_data = np.array(qy_train_data[num]).reshape(-1,sequence_length,input_size)
                    domain_label = qy_train_label[num][::50].view(-1)

                    source_train_dloader,source_test_dloader = get_domainnet_dloader_train(domainnet_base_path_obsolete, domain_data, domain_label,
                                                                       configs["TrainingConfig"]["batch_size"],
                                                                       args.workers)
                    train_dloaders.append(source_train_dloader)
                    test_dloaders.append(source_test_dloader)
                    models.append(DomainNet_encoder(time_window, configs["ModelConfig"]["backbone"], args.bn_momentum, False, False).cuda())
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
                                      
            print("Begin")
            print(" ")

            # create the initialized domain weight
            domain_weight = create_domain_weight(len(args.source_domains))
            # adjust training strategy with communication round
            batch_per_epoch, total_epochs = decentralized_training_strategy(
                communication_rounds=configs["UMDAConfig"]["communication_rounds"],
                epoch_samples=configs["TrainingConfig"]["epoch_samples"],
                batch_size=configs["TrainingConfig"]["batch_size"],
                total_epochs=configs["TrainingConfig"]["total_epochs"])
            
            # train model

            domain_weight_list = []
            train_acc_list = []
            acc_list = []
            train_loss_list = []
            test_loss_list = []

            current_max_acc = 0
            
            length__test = len(test_dloaders[0])

            for epoch in range(args.start_epoch, total_epochs):

                domain_weight, train_loss, _, _, _, _, _ = train(
                    train_dloaders, models, classifiers, optimizers, classifier_optimizers, 
                    epoch=epoch, 
                    num_classes=num_classes,
                    model_savepath=None, 
                    turn=turn, 
                    Day_from=Day_from, Day_to=Day_to, testDay=test_Day,
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
                    save_whs__path = path__save_w_hs + f"/{name__model_type}",
                    save_whs__condition = condition,
                    save_whs__domain = save_domain,
                )

                domain_weight_list.append(domain_weight)
                train_acc_list.append(0.0)	

                import time

                time__test_start = time.time()
                acc_top1, acc_top2, acc_top3, test_loss, _ = test(
                    target_domain, args.source_domains, test_dloaders, models, classifiers, epoch, total_epochs, today, 
                    turn=turn, 
                    Day_from=Day_from, Day_to=Day_to, testDay=test_Day,
                    num_classes=num_classes, 
                    top_5_accuracy=True,
                    save_whs__path = path__save_w_hs + f"/{name__model_type}",
                    save_whs__condition = condition,
                    save_whs__domain = save_domain,
                    current_max_acc = current_max_acc,
                )
                time__test_end = time.time()
                time__test = time__test_end - time__test_start

                current_max_acc = max(acc_top1, current_max_acc)

                acc_list.append([acc_top1*100,acc_top2*100,acc_top3*100,seeds])
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
                        {
                            "epoch": epoch + 1,
                            "domain": target_domain,
                            "backbone": models[0].state_dict(),
                            "classifier": classifiers[0].state_dict(),
                            "optimizer": optimizers[0].state_dict(),
                            "classifier_optimizer": classifier_optimizers[0].state_dict()
                        },
                        filename="{}.pth.tar".format(target_domain)
                    )


            last_acc = acc_list[-1]
            max_acc = np.max(np.array(acc_list), axis=0)
            print(name__model_type)
            print("Maximum accuracy", max_acc)
            print("Last epoch accuracy", last_acc)
            max_acc_list.append(max_acc)
            last_acc_list.append(last_acc)

            # save domain weight
            path__res_domain_weight = path__save_domain_weight + '/domain_weight_' + week_name_from_to + '_GY-p2_crossweek_' + str(cross_week_num) + '__Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '.npy'
            np.save(path__res_domain_weight, domain_weight_list)

            # save loss
            loss_file_name = '/outputs_loss_' + week_name_from_to + '_GY-p2_crossweek_' + str(cross_week_num) + '__Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to

            path__res_loss_train = path__save_loss + loss_file_name + '_train.npy'
            np.save(path__res_loss_train, train_loss_list)
            path__res_loss_test = path__save_loss + loss_file_name + '_test.npy'
            np.save(path__res_loss_test, test_loss_list)

        acc_file_name = f'/outputs_acc_GY-p2_crossweek_{cross_week_num}__Day_from_{Day_from}_to_{Day_to}_testDay_{test_Day}'

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


        print("=== MAXIMUM TEST ACCURACY ===")
        selected_max_acc = np.stack(max_acc_list)[:, :]
        print(selected_max_acc[:, [0, 3]])
        print("mean =", selected_max_acc[:, 0].mean())

        print(" ")

        print("=== LAST EPOCH ACCURACY ===")
        selected_last_acc = np.stack(last_acc_list)[:, :]
        print(selected_last_acc[:, [0, 3]])
        print("mean =", selected_last_acc[:, 0].mean())

    print("Result folder:", path__results)

def save_checkpoint(state, filename):
    pass


if __name__ == "__main__":
    print('start training')
    main()
    print('end training')