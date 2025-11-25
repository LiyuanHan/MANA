import random

import numpy as np



import argparse
from model.digit5 import CNN, Classifier
from model.amazon import AmazonMLP, AmazonClassifier
from model.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
from model.domainnet import DomainNet_encoder, DomainNetClassifier
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

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
    

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

# set the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn



# cross week setup
from lib.utils.CrossMultiWeekDictionaries import *


path__loadData = './GY-p2/data'
path__project = "./GY-p2/CL-MA-AD/ANN/AD"

path__dimReduced = path__project + '/data/dim_reduced'
path__aligned = path__project + '/data/aligned'
path__decodedResults = path__project + '/results'
path__stdEmbeddings = path__project + '/standard_embedding'

path_subpath__modelName = f'/KD3AwoCL_crossday_mix-{mix_length}'

path__save_acc = path__decodedResults + path_subpath__modelName + '/acc'
path__save_loss = path__decodedResults + path_subpath__modelName + '/loss'
path__save_linout = path__decodedResults + path_subpath__modelName + '/linear_output'
path__save_model = path__decodedResults + path_subpath__modelName + '/model'
path__save_domain_weight = path__decodedResults + path_subpath__modelName + '/domain_weight'

check_paths = [path__save_acc, path__save_loss, path__save_linout, path__save_model, path__save_domain_weight]





print(path_subpath__modelName)
os.makedirs(path__decodedResults + path_subpath__modelName, exist_ok=True)
for folder in check_paths:
    os.makedirs(folder, exist_ok=True)

# import config files
with open(path__project + r"/config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)


def main(args=args, configs=configs):
    import numpy as np
    import torch

    data_from_to = '20221103_20230911' # all days

    device = torch.device("cuda:0")

    align_method = 2
    print('align_method= ', align_method)
    # today = datetime.date.today()
    # today = str(today)
    # today = '2023-07-11'
    # filepath = ''

    def align_embedings_cross_days(cebra_pos_train, cebra_pos_test):
        
        cebra_pos_train_sample = cebra_pos_train
        
        Q_train, R_train = torch.linalg.qr(cebra_pos_train_sample)
        Q_test, R_test = torch.linalg.qr(cebra_pos_test)
        U, S, V = torch.linalg.svd(Q_train.T @ Q_test)
        V = V.T
        
        cebra_pos_test_align = Q_train @ U @ torch.linalg.pinv(V) @ R_test
        
        return cebra_pos_test_align


    ##
    # data loading
    bin_method = 'expect_bins'
    expect_num_bins = 50
    print('bin_method: ', bin_method)
    print('expect_num_bins: ', expect_num_bins)

    # if bin_method == 'expect_bins':

    #     all_micro_spikes_concat = torch.load(path__loadData + '/all_micro_spikes_expect_bin_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
    #     all_macro_conditions_concat = torch.load(path__loadData + '/all_macro_conditions_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
        
    #     len_for_each_session_trial = torch.load(path__loadData + '/len_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
    #     target_for_each_session_trial = torch.load(path__loadData + '/target_for_each_session_trial_expect_' + str(expect_num_bins) + '_concat_gy-p2_' + data_from_to + '.pt')
        
    #     # 1-8 ==> 0-7
    #     all_macro_conditions_concat = all_macro_conditions_concat - torch.tensor(1)
    #     for i in range(len(target_for_each_session_trial)):
    #         target_for_each_session_trial[i] = (np.array(target_for_each_session_trial[i]) - 1).tolist()

    
    cross_week_num = 1

    # cross-week setup
    verify_day_in_test_range(variable_num, cross_week_num)
    domain_dic, day_from_to_dic = generate_domain_dict_and_day_from_to_dict(cross_week_num)


    
    seed_set = np.array([31, 44, 46, 54, 59])
    Day_from = day_from_to_dic[str(variable_num)][0] 
    Day_to =  day_from_to_dic[str(variable_num)][1]  
    test_Day_list = [variable_num]  
    week_count_list = domain_dic[str(variable_num)]   
    week_name_from_to = str(week_all[str(Day_to)])
    print("week_name_from_to =", week_name_from_to)


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
                
                print('turn: ', turn)
                # today = '2023-10-08-v7-turn_' + str(turn)
                today = '20240312-v2-p2-crossday_h1-'+str(64)+'_h2-3072'+'_turn_' + str(turn) # v1, v2
                # today = str(today_list[test_count]) + str(turn)
                
                
                if bin_method == 'expect_bins':
                    train_data = 'days'
                    print('train_data: ', train_data)
                    if train_data == 'days':
                        
                        print('------------------Day_from_{}_to_{}------------------'.format(Day_from, Day_to))
                        print('--------------------test_Day: {}------------------'.format(test_Day))
                        
                        
                #         def split_data(data_spike_train, data_label, len_for_each_session_trial, Day_from, Day_to):
                            
                #             split_idx_start_beg = 0
                #             for i in range(day[str(Day_from - 1)]):
                #                 split_idx_start_beg += sum(len_for_each_session_trial[i])
                #             split_idx_start_end = 0
                #             for i in range(day[str(Day_to)]):
                #                 split_idx_start_end += sum(len_for_each_session_trial[i])
                            
                #             split_idx_start = 0
                #             for i in range(day[str(test_Day - 1)]):
                #                 split_idx_start += sum(len_for_each_session_trial[i])
                #             split_idx_end = 0  # 只预测第test_Day这一天的
                #             for i in range(day[str(test_Day)]):
                #                 split_idx_end += sum(len_for_each_session_trial[i])
                            
                #             neural_train = data_spike_train[split_idx_start_beg:split_idx_start_end, :]
                #             label_train = data_label[split_idx_start_beg:split_idx_start_end, :]
                            
                #             neural_test = data_spike_train[split_idx_start:split_idx_end, :]
                #             label_test = data_label[split_idx_start:split_idx_end, :]
                #             return neural_train, neural_test, label_train, label_test
                
                # # split data
                # neural_train, neural_test, label_train, label_test = split_data(all_micro_spikes_concat,
                #                                                                 all_macro_conditions_concat,
                #                                                                 len_for_each_session_trial, Day_from,
                #                                                                 Day_to)  # direction
                
                with open('./GY-p2/data/data--long-term-decoding.pkl', 'rb') as f:
                    len_for_each_session_trial, target_for_each_session_trial, neural_train, neural_test, label_train, label_test = pickle.load(f)

                
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
                input_size = 232  # dim of embedding output
                
                
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
                
                qy_train_data, qy_train_label = split_data_for_day(neural_train, label_train, len_for_each_session_trial, Day_from)
                
                qy_test_data = neural_test
                qy_test_label = label_test




                # 闹点数据加载
                # domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

                # domains = torch.arange(0, 42, 1).tolist()
                # this_source_domain = [0, 2, 3, 4, 5]
                # this_target_domain = [1]
                args.source_domains = list(range(len(week_count_list)))

                # import pdb; pdb.set_trace()
                test_data = qy_test_data.reshape(-1,sequence_length,input_size)
                test_label = qy_test_label[::50].view(-1)
                # test_label = np.eye(8)[test_label].reshape(-1,8)
                print(test_data.shape)
                print(test_label.shape)

                target_train_dloader ,target_test_dloader = get_domainnet_dloader_test(args.base_path, test_data, test_label, configs["TrainingConfig"]["batch_size"], args.workers, mix_length)

                train_dloaders.append(target_train_dloader)
                test_dloaders.append(target_test_dloader)
                test_model = DomainNet_encoder(time_window, 'resnet101', args.bn_momentum, False, False).cuda()
                print(count_parameters(test_model))
                print(count_parameters(DomainNetClassifier(configs["ModelConfig"]["backbone"], 8, args.data_parallel).cuda()))
                # print(test_model.__dict__.keys())
                models.append(test_model)
                classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 8, args.data_parallel).cuda())
                # domains.remove(this_target_domain)

                for num in range(len(week_count_list)):
                    # import pdb; pdb.set_trace()
                    domain_data = np.array(qy_train_data[num]).reshape(-1,sequence_length,input_size)
                    domain_label = qy_train_label[num][::50].view(-1)
                    # domain_label = np.eye(8)[domain_label].reshape[-1,8]
                    print(domain_data.shape)
                    print(domain_label.shape)

                    source_train_dloader,source_test_dloader = get_domainnet_dloader_train(args.base_path, domain_data, domain_label,
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

            domain_weight_list = []
            train_acc_list = []
            acc_list = []
            train_loss_list = []
            test_loss_list = []

            for epoch in range(args.start_epoch, total_epochs):
                domain_weight, train_loss, domain1_loss, domain2_loss, domain3_loss, domain4_loss, target_linear_outputs = train(
                    train_dloaders, models, classifiers, optimizers, classifier_optimizers, epoch, 
                    num_classes=num_classes,
                    model_savepath=path__save_model, 
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
                    mix_aug=(configs["DataConfig"]["dataset"] != "AmazonReview")
                )

                domain_weight_list.append(domain_weight)

                # # TODO: record training accuracies
                # # currently under maintainance. need to adjust train() for additional return value of training accuracy.
                # train_acc_list.append(train_acc)
                train_acc_list.append(0.0)		# maintainance substitution

                acc_top1, acc_top2, acc_top3, test_loss = test(
                    args.target_domain, args.source_domains, test_dloaders, models, classifiers, epoch, total_epochs, today, test_Day,
                    num_classes=num_classes, 
                    top_5_accuracy=True,
                )

                acc_list.append([acc_top1*100,acc_top2*100,acc_top3*100,seeds])
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)

                # torch.save(models[0], path__save_model + f'/target_model_state_testday_{test_Day}_turn_{turn}_epoch_{epoch}.pth')

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
                        filename="{}.pth.tar".format(args.target_domain)
                    )

                    # 保存每10个epoch的linout

                    # average_target_linear_outputs = target_linear_outputs.mean(dim=0).unsqueeze(0)
                    # print(f'epoch {epoch} average_target_linear_outputs shape:', average_target_linear_outputs.shape)

                # 每个epoch保存
                if (epoch + 1) % 1 == 0:

                    average_target_linear_outputs = target_linear_outputs.mean(dim=0).unsqueeze(0)
                    torch.save(average_target_linear_outputs, path__save_linout + '/all_linear_outputs' + '_testDay_' + str(test_Day) + f'_turn_{turn}_epoch_{epoch}.pt')
                    torch.save(models[0], path__save_model + f'/target_model_state_testday_{test_Day}_turn_{turn}_epoch_{epoch}.pth') 

                # if acc_top1 >= 1:
                # 	break

                # if epoch == 49: # stop after 50 epochs
                # 	break

            ## graph acc curves

            # import matplotlib.pyplot as plt

            # path__decodedResults_acc_plt = path__save_acc + '/outputs_acc_align_data_align_method_' + str(align_method) + '_0' + week_name_from_to + '_GY-p2_crossday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A.png'

            # plt.plot(train_acc_list, label='Train Accuracy')
            # plt.plot(np.array(acc_list)[:, 0], label='Test Accuracy')
            # plt.title(f'Training and Test Accuracies Over Epochs, test day {test_Day}, turn {turn}, cross-day')
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracies')
            # plt.legend()
            # plt.savefig(path__decodedResults_acc_plt)
            # plt.close()


            last_acc = acc_list[-1]
            max_acc = np.max(np.array(acc_list), axis=0)
            print(path_subpath__modelName)
            print("Maximum accuracy", max_acc)
            print("Last epoch accuracy", last_acc)
            max_acc_list.append(max_acc)
            last_acc_list.append(last_acc)

            # save domain weight
            path__decodedResults_domain_weight = path__save_domain_weight + '/domain_weight_align_data_align_method_' + str(align_method) + '_0' + '_GY-p2_crossday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from) + '_' + str(Day_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A_train.npy'
            np.save(path__decodedResults_domain_weight, domain_weight_list)


            # save loss
            loss_file_name = '/outputs_loss_' + week_name_from_to + '_GY-p2_KD3AwoCL__Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to

            path__decodedResults_loss_train = path__save_loss + loss_file_name + '_train.npy'
            np.save(path__decodedResults_loss_train, train_loss_list)
            path__decodedResults_loss_test = path__save_loss + loss_file_name + '_test.npy'
            np.save(path__decodedResults_loss_test, test_loss_list)

            # ## graph loss curves

            # import matplotlib.pyplot as plt

            # path__decodedResults_loss_plt = path__save_loss + '/outputs_loss_align_data_align_method_' + str(align_method) + '_0' + '_GY-p2_crossday_Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_' + 'add_' + str(Day_from + add_num) + '_' + str(Day_to + add_num_to) + '_' + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to + '_KD3A.png'

            # plt.plot(train_loss_list, label='Train Loss')  # 绘制训练损失曲线
            # plt.plot(test_loss_list, label='Test Loss')  # 绘制测试损失曲
            # plt.title('Training and Test Losses Over Epochs')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.savefig(path__decodedResults_loss_plt)
            # plt.close()


        # save acc
        acc_file_name = '/outputs_acc_' + week_name_from_to + '_GY-p2_KD3AwoCL__Day_from_' + str(Day_from) + '_to_' + str(Day_to) + '_testDay_' + str(test_Day) + '_batch_' + str(configs["TrainingConfig"]["batch_size"]) + '_' + today + '_' + data_from_to

        path__decodedResults_accMax = path__save_acc + acc_file_name + '.npy'
        np.save(path__decodedResults_accMax, max_acc_list)
        path__decodedResults_accLast = path__save_acc + acc_file_name + '_last.npy'
        np.save(path__decodedResults_accLast, last_acc_list)
        

        print(" ")
        
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(" ")

        print(f"train {Day_from} - {Day_to} ==> test {test_Day}")
        print(" ")

        seed_selection = [0, 1, 2, 5, 7]

        print("=== MAXIMUM TEST ACCURACY ===")
        selected_max_acc = np.stack(max_acc_list)[seed_selection, :]
        print(selected_max_acc[:, [0, 3]])
        # all_top_1 = np.stack(max_acc_list)[:, 0]
        # print("Sorted Accuracies:")
        # print(np.sort(all_top_1))
        print("mean =", selected_max_acc[:, 0].mean())

        print(" ")

        print("=== LAST EPOCH ACCURACY ===")
        selected_last_acc = np.stack(last_acc_list)[seed_selection, :]
        print(selected_last_acc[:, [0, 3]])
        # all_top_1 = np.stack(last_acc_list)[:, 0]
        # print("Sorted Accuracies:")
        # print(np.sort(all_top_1))
        print("mean =", selected_last_acc[:, 0].mean())

        # acc_of_seeds = np.mean(np.array(max_acc_list), axis=0)[:-1]
        # print(acc_of_seeds)
        # logger.info(acc_of_seeds)
        # acc_of_week.append(acc_of_seeds)

    print(path_subpath__modelName)
    # print(acc_of_week)
    # logger.info(acc_of_week)

def save_checkpoint(state, filename):
    # args.train_time = 1
    filefolder = "{}/{}/parameter".format(args.base_path, configs["DataConfig"]["dataset"])
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == "__main__":
    print('start training')
    main()
    print('end training')
