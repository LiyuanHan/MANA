import torch
import torch.nn as nn
import numpy as np
import os
from lib.utils.avgmeter import AverageMeter

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

import pickle


def train(
        train_dloader, model, classifier, optimizer, classifier_optimizer, epoch, total_epochs, turn, Day_from, Day_to, testDay, regularization_weight=0, 
        save_whs__path="", save_whs__condition=None,
    ):

    if save_whs__condition.check_test_day(testDay) and save_whs__condition.check_turn(turn) and (epoch == 0):
        # save initial model
        os.makedirs(save_whs__path, exist_ok=True)
        os.makedirs(save_whs__path + "/model_params/resnet", exist_ok=True)
        os.makedirs(save_whs__path + "/model_params/classifier", exist_ok=True)
        if save_whs__condition:
            file_name = f"train_{Day_from}_{Day_to}-test_{testDay}-turn_{turn}-epoch_init--train"

            torch.save(model, save_whs__path + "/model_params/resnet/resnet-" + file_name + ".pth")
            torch.save(classifier, save_whs__path + "/model_params/classifier/classifier-" + file_name + ".pth")
            
            print(f"model_params saved for {file_name}")

    task_criterion = nn.MSELoss().cuda()
    
    model.train()
    classifier.train()

    train_loss = AverageMeter()

    for i, (image_s, label_s, pos_s) in enumerate(train_dloader):
        
        image_s = image_s.cuda()            # [B, 1, S, F]
        label_s = label_s.long().cuda()     # [B]
        pos_s = pos_s.cuda()                # [B, S, 2]

        optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        feature_s = model(image_s)          # [B, 512, 1, 1]
        conv_outputs = model.get_conv_outputs()
        linear_outputs = model.get_linear_outputs()

        output_s = classifier(feature_s)    # [B, S, 2]

        final_outputs = output_s.clone().detach()       # [B, S, 2]
        labels = label_s.clone().detach().unsqueeze(-1) # [B, 1]
        positions = pos_s.clone().detach()              # [B, S, 2]

        task_loss_s = task_criterion(output_s, pos_s) 

        L1_reg_loss = 0
        for param in model.parameters():
            L1_reg_loss += torch.sum(torch.abs(param))
        task_loss_s += regularization_weight * L1_reg_loss

        task_loss_s.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()
        classifier_optimizer.step()

        train_loss.update(task_loss_s.item(), image_s.size(0))

    print("epoch: {} training == Loss: {:.5f}".format(epoch, task_loss_s.item()))

    if save_whs__condition.check_test_day(testDay) and save_whs__condition.check_turn(turn) and (epoch == total_epochs - 1):
        # save initial model
        os.makedirs(save_whs__path, exist_ok=True)
        os.makedirs(save_whs__path + "/model_params/resnet", exist_ok=True)
        os.makedirs(save_whs__path + "/model_params/classifier", exist_ok=True)
        if save_whs__condition:
            file_name = f"train_{Day_from}_{Day_to}-test_{testDay}-turn_{turn}-epoch_{epoch}--train"

            torch.save(model, save_whs__path + "/model_params/resnet/resnet-" + file_name + ".pth")
            torch.save(classifier, save_whs__path + "/model_params/classifier/classifier-" + file_name + ".pth")
            
            print(f"model_params saved for {file_name}")

    return train_loss.avg, model, classifier, optimizer, classifier_optimizer


def test(
    test_dloader, model, classifier, epoch, turn, Day_from, Day_to, testDay,
    save_whs__path="", save_whs__condition=None,
    current_max_acc=0,
    ):
    
    test_loss = AverageMeter()
    task_criterion = nn.MSELoss().cuda()  

    test_conv_outputs_list = []
    test_linear_outputs_list = []
    test_pred_pos_list = []
    test_real_pos_list = []
    test_real_label_list = []

    model.eval()
    classifier.eval()
    
    all_predictions = []
    all_targets = []

    for i, (image_t, label_t, pos_t) in enumerate(test_dloader):
        image_t = image_t.cuda()        # [B, 1, 30, 288]
        label_t = label_t.long().cuda() # [B]
        pos_t = pos_t.cuda()            # [B, 30, 2]
        
        with torch.no_grad():
            output_t = classifier(model(image_t))   # [B, 30, 2]
            test_conv_outputs_list.append(model.get_conv_outputs())
            test_linear_outputs_list.append(model.get_linear_outputs())

            final_outputs = output_t.clone().detach()
            labels = label_t.clone().detach().unsqueeze(-1).cuda()
            positions = pos_t.clone().detach()
            test_pred_pos_list.append(final_outputs)
            test_real_pos_list.append(positions)
            test_real_label_list.append(labels)

        task_loss_t = task_criterion(output_t, pos_t)
        test_loss.update(float(task_loss_t.item()), image_t.size(0))
        
        all_predictions.append(output_t)
        all_targets.append(pos_t)

    test_conv_outputs = torch.cat(test_conv_outputs_list, dim=0)
    test_linear_outputs = torch.cat(test_linear_outputs_list, dim=0)
    test_pred_pos = torch.cat(test_pred_pos_list, dim=0)
    test_real_pos = torch.cat(test_real_pos_list, dim=0)
    test_real_label = torch.cat(test_real_label_list, dim=0)

    hidden_stats = {
        "conv_outputs": test_conv_outputs,
        "linear_outputs": test_linear_outputs,
        "predicted_positions": test_pred_pos,
        "real_positions": test_real_pos,
        "real_labels": test_real_label,
    }

    all_predictions = torch.cat(all_predictions, dim=0)  # [N, 30, 2]
    all_targets = torch.cat(all_targets, dim=0)  # [N, 30, 2]
    
    ss_res = torch.sum((all_targets - all_predictions) ** 2, dim=(0, 1)) 
    ss_tot = torch.sum((all_targets - all_targets.mean(dim=(0, 1), keepdim=True)) ** 2, dim=(0, 1))
    
    r2_x = 1 - (ss_res[0] / (ss_tot[0] + 1e-8))
    r2_y = 1 - (ss_res[1] / (ss_tot[1] + 1e-8))
    r2_avg = (r2_x + r2_y) / 2
    
    rmse = torch.sqrt(torch.mean((all_targets - all_predictions) ** 2))
    rmse_x = torch.sqrt(torch.mean((all_targets[:, :, 0] - all_predictions[:, :, 0]) ** 2))
    rmse_y = torch.sqrt(torch.mean((all_targets[:, :, 1] - all_predictions[:, :, 1]) ** 2))

    print("epoch: {} testing === Loss: {:.5f}, ✅ R²(avg): {:.4f}, R²(x): {:.4f}, R²(y): {:.4f}, RMSE: {:.4f}".format(
        epoch, test_loss.avg, r2_avg.item(), r2_x.item(), r2_y.item(), rmse.item()
    ))

    os.makedirs(save_whs__path, exist_ok=True)
    os.makedirs(save_whs__path + "/hidden_states", exist_ok=True)
    
    if save_whs__condition.check_test_day(testDay) and save_whs__condition.check_turn(turn):
        if current_max_acc < r2_avg.item():
            file_name = f"train_{Day_from}_{Day_to}-test_{testDay}-turn_{turn}-epoch_best--test"

            with open(save_whs__path + "/hidden_states/" + file_name + ".pkl", "wb") as f:
                pickle.dump(hidden_stats, f)

            print(f"New best R² = {r2_avg.item():.4f} !!!")
            print(f"    saved to {file_name + '.pkl'}")

        if save_whs__condition.check_epoch(epoch):
            file_name = f"train_{Day_from}_{Day_to}-test_{testDay}-turn_{turn}-epoch_{epoch}--test"

            with open(save_whs__path + "/hidden_states/" + file_name + ".pkl", "wb") as f:
                pickle.dump(hidden_stats, f)
            print(f"    saved to {file_name + '.pkl'}")

    print(" ")
    
    return r2_avg.item(), r2_x.item(), r2_y.item(), test_loss.avg, test_linear_outputs
