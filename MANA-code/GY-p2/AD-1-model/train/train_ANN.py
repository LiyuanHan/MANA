import torch
import torch.nn as nn
import numpy as np
import os
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

import pickle


def create_domain_weight(source_domain_num):
    return create_domain_weight_(source_domain_num)

def decentralized_training_strategy(communication_rounds, epoch_samples, batch_size, total_epochs):
    return decentralized_training_strategy_(communication_rounds, epoch_samples, batch_size, total_epochs)


def train(
        train_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, num_classes, model_savepath, turn, Day_from, Day_to, testDay, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin, confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, mix_aug=True, regularization_weight=0, 
        save_whs__path="", save_whs__condition=None, save_whs__domain=[],
    ):

    if save_whs__condition.check_test_day(testDay) and save_whs__condition.check_turn(turn) and (epoch == 0):
        # save initial model
        os.makedirs(save_whs__path, exist_ok=True)
        os.makedirs(save_whs__path + "/model_params/resnet", exist_ok=True)
        os.makedirs(save_whs__path + "/model_params/classifier", exist_ok=True)
        if save_whs__condition:
            for i in save_whs__domain: 
                if i in range(len(train_dloader_list)):
                    file_name = f"train_{Day_from}_{Day_to}-test_{testDay}-turn_{turn}-epoch_init-domain_{i}"

                    torch.save(model_list[i], save_whs__path + "/model_params/resnet/resnet-" + file_name + ".pth")
                    torch.save(classifier_list[i], save_whs__path + "/model_params/classifier/classifier-" + file_name + ".pth")
                    
                    print(f"model_params saved for {file_name}")

    task_criterion = nn.CrossEntropyLoss().cuda()
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1

    domain_idex = 0
    for f in range(model_aggregation_frequency):
        current_domain_index = 0

        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:], model_list[1:], classifier_list[1:], optimizer_list[1:], classifier_optimizer_list[1:]):

            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False

            for i, (image_s, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break
                image_s = image_s.cuda()
                label_s = label_s.long().cuda()
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # each source domain do optimize
                feature_s = model(image_s) 

                domain_key = f"domain{domain_idex}"
                conv_outputs = model.get_conv_outputs()
                linear_outputs = model.get_linear_outputs()

                output_s = classifier(feature_s) 

                final_outputs = output_s.clone().detach()
                labels = label_s.clone().detach().unsqueeze(-1)
                final_outputs_labels = torch.cat((final_outputs, labels), dim=1)

                task_loss_s = task_criterion(output_s, label_s.view(label_s.shape[0]))

                L1_reg_loss = 0
                for param in model.parameters():
                    L1_reg_loss += torch.sum(torch.abs(param))
                task_loss_s += regularization_weight * L1_reg_loss

                task_loss_s.backward()
                optimizer.step()
                classifier_optimizer.step()
            domain_idex = domain_idex + 1

    # Domain adaptation on target domain
    confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    # We use I(n_i>=1)/(N_T) to adjust the weight for knowledge distillation domain
    target_weight = [0, 0]
    consensus_focus_dict = {}
    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0


    train_domain_loss = AverageMeter()
    target_conv_outputs_list = []
    target_linear_outputs_list = []
    target_final_outputs_labels_list = []

    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        if i >= batch_per_epoch:
            break
        optimizer_list[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()
        image_t = image_t.cuda()
        # Knowledge Vote
        with torch.no_grad():
            knowledge_list = [torch.softmax(classifier_list[i](model_list[i](image_t)), dim=1).unsqueeze(1) for i in range(1, source_domain_num + 1)]
            knowledge_list = torch.cat(knowledge_list, 1)
        _, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate, num_classes=num_classes)
        target_weight[0] += torch.sum(consensus_weight).item()
        target_weight[1] += consensus_weight.size(0)
        # Perform data augmentation with mixup
        if mix_aug:
            lam = np.random.beta(2, 2)
        else:
            lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
        feature_t = model_list[0](mixed_image)

        conv_outputs = model_list[0].get_conv_outputs()
        target_conv_outputs_list.append(conv_outputs)
        linear_outputs = model_list[0].get_linear_outputs()
        target_linear_outputs_list.append(linear_outputs)

        output_t = classifier_list[0](feature_t)

        final_outputs = output_t.clone().detach()
        labels = label_t.clone().detach().unsqueeze(-1).cuda()
        concensuses = mixed_consensus.clone().detach()
        final_outputs_labels = torch.cat((final_outputs, concensuses, labels), dim=1)
        target_final_outputs_labels_list.append(final_outputs_labels)

        output_t = torch.log_softmax(output_t, dim=1)
        task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))
        
        L1_reg_loss = 0
        for param in model_list[0].parameters():
            L1_reg_loss += torch.sum(torch.abs(param))
        task_loss_t += regularization_weight * L1_reg_loss

        task_loss_t.backward()
        optimizer_list[0].step()
        classifier_optimizer_list[0].step()

        train_domain_loss.update(task_loss_t.item(), image_t.size(0))

        # Calculate consensus focus
        consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate, source_domain_num, num_classes)

    target_conv_outputs = torch.cat(target_conv_outputs_list, dim=0)
    target_linear_outputs = torch.cat(target_linear_outputs_list, dim=0)    # 40,512
    target_final_outputs_labels = torch.cat(target_final_outputs_labels_list, dim=0)

    # Consensus Focus Re-weighting
    target_parameter_alpha = target_weight[0] / target_weight[1]
    target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    epoch_domain_weight = []
    source_total_weight = 1 - target_weight
    for i in range(1, source_domain_num + 1):
        epoch_domain_weight.append(consensus_focus_dict[i])
    if sum(epoch_domain_weight) == 0:
        epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
    epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in epoch_domain_weight]
    epoch_domain_weight.insert(0, target_weight)

    # Update domain weight with moving average
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)
        
    # Model aggregation and Batchnorm MMD
    federated_average(model_list, domain_weight, batchnorm_mmd=batchnorm_mmd)

    print("epoch: {} training == Loss: {:.5f}, Source Domains: {}, Domain Weight: {}".format(epoch, train_domain_loss.avg, source_domains, domain_weight[1:]))

    return domain_weight, train_domain_loss.avg, target_linear_outputs, None, None, None, None


def test(
    target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, total_epochs, today, turn, Day_from, Day_to, testDay, num_classes=126, top_5_accuracy=True,
    save_whs__path="", save_whs__condition=None, save_whs__domain=[],
    current_max_acc=0,
    ):
    source_domain_losses = [AverageMeter() for i in source_domains]
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()

    test_conv_outputs_list = []
    test_linear_outputs_list = []
    test_pred_output_list = []
    test_real_label_list = []

    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]

    for i, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            output_t = classifier_list[0](model_list[0](image_t))
            test_conv_outputs_list.append(model_list[0].get_conv_outputs())
            test_linear_outputs_list.append(model_list[0].get_linear_outputs())

            final_outputs = output_t.clone().detach()
            labels = label_t.clone().detach().unsqueeze(-1).cuda()
            test_pred_output_list.append(final_outputs)
            test_real_label_list.append(labels)

        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        task_loss_t = task_criterion(output_t, label_t.view(label_t.shape[0]))
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)

    test_conv_outputs = torch.cat(test_conv_outputs_list, dim=0)
    test_linear_outputs = torch.cat(test_linear_outputs_list, dim=0)
    test_pred_outputs = torch.cat(test_pred_output_list, dim=0)
    test_real_label = torch.cat(test_real_label_list, dim=0)

    hidden_stats = {
        "conv_outputs": test_conv_outputs,
        "linear_outputs": test_linear_outputs,
        "predicted_outputs": test_pred_outputs,
        "real_labels": test_real_label,
    }

    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    if top_5_accuracy:
        _, y_pred_1 = torch.topk(tmp_score, k=1, dim=1)
        _, y_pred_2 = torch.topk(tmp_score, k=2, dim=1)
        _, y_pred_3 = torch.topk(tmp_score, k=3, dim=1)
    else:
        _, y_pred = torch.topk(tmp_score, k=1, dim=1)

    top_1_accuracy_t = float(torch.sum(y_true == y_pred_1[:,:1]).item()) / y_true.size(0)
    top_2_accuracy_t = float(torch.sum(torch.tensor([1 if y in top2 else 0 for y, top2 in zip(y_true, y_pred_2)])).item()) / y_true.size(0)
    top_3_accuracy_t = float(torch.sum(torch.tensor([1 if y in top3 else 0 for y, top3 in zip(y_true, y_pred_3)])).item()) / y_true.size(0)

    print("epoch: {} testing === Loss: {:.5f}, Target Domain: {}, Accuracy Top1: {:.3f}".format(epoch, target_domain_losses.avg, target_domain, top_1_accuracy_t))

    # save hidden states
    os.makedirs(save_whs__path, exist_ok=True)
    os.makedirs(save_whs__path + "/hidden_states", exist_ok=True)
    
    if save_whs__condition.check_test_day(testDay) and save_whs__condition.check_turn(turn):
        if 5 in save_whs__domain: 
            if current_max_acc < top_1_accuracy_t:
                file_name = f"train_{Day_from}_{Day_to}-test_{testDay}-turn_{turn}-epoch_best-domain_test"

                with open(save_whs__path + "/hidden_states/" + file_name + ".pkl", "wb") as f:
                    pickle.dump(hidden_stats, f)

                print(f"New best acc = {top_1_accuracy_t:.4f} !!!")
                print(f"    saved to {file_name + '.pkl'}")

            if save_whs__condition.check_epoch(epoch):
                file_name = f"train_{Day_from}_{Day_to}-test_{testDay}-turn_{turn}-epoch_{epoch}-domain_test"

                with open(save_whs__path + "/hidden_states/" + file_name + ".pkl", "wb") as f:
                    pickle.dump(hidden_stats, f)

                print(f"    saved to {file_name + '.pkl'}")

    print(" ")
        
    return top_1_accuracy_t, top_2_accuracy_t, top_3_accuracy_t, target_domain_losses.avg, test_linear_outputs

