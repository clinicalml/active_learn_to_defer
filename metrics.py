import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


cifar_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def metrics_print_2step(net_mod, net_exp, expert_fn, n_classes, loader):
    '''
    Computes metrics for the confidence score method: on each example compare net_mod (classifier) and net_exp (expert) accuracies and defer 
    net_mod: classifier model
    net_exp: expert error model (pytorch)
    expert_fn: actual synth expert
    n_classes: number of classes
    loader: data loader
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    with torch.no_grad():
        for data in loader:
            images, labels, expert_preds, _, _ = data
            images, labels, expert_preds = images.to(device), labels.to(device), expert_preds.to(device)
            outputs_mod = net_mod(images)
            outputs_exp = net_exp(images)
            _, predicted = torch.max(outputs_mod.data, 1)
            _, predicted_exp = torch.max(outputs_exp.data, 1)
            batch_size = outputs_mod.size()[0]  # batch_size
            exp_prediction = expert_fn(images, labels)
            for i in range(0, batch_size):
                r_score =  outputs_mod.data[i][predicted[i].item()].item()
                r_score = outputs_exp.data[i][1].item() - r_score
                r = 0
                if r_score >= 0:
                    r = 1
                else:
                    r = 0
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001)}
    return to_print
    print(to_print)
    
def metrics_print_2step_linear(net_mod, net_exp, expert_fn, n_classes, loader):
    '''
    Computes metrics for the confidence score method with linear representations
    net_mod: classifier model
    net_exp: expert error model (pytorch)
    expert_fn: actual synth expert
    n_classes: number of classes
    loader: data loader
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    with torch.no_grad():
        for data in loader:
            images, labels, expert_preds, _, images_orig = data
            images, labels, expert_preds, images_orig = images.to(device), labels.to(device), expert_preds.to(device), images_orig.to(device)
            outputs_mod = net_mod(images_orig)
            outputs_exp = net_exp(images)
            _, predicted = torch.max(outputs_mod.data, 1)
            _, predicted_exp = torch.max(outputs_exp.data, 1)
            batch_size = outputs_mod.size()[0]  # batch_size
            exp_prediction = expert_fn(images, labels)
            for i in range(0, batch_size):
                r_score =  outputs_mod.data[i][predicted[i].item()].item()
                r_score = outputs_exp.data[i][1].item() - r_score
                r = 0
                if r_score >= 0:
                    r = 1
                else:
                    r = 0
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001)}
    return to_print
    print(to_print)

def metrics_print(net, expert_fn, n_classes, loader):
    '''
    Computes metrics for deferal (L_{CE} loss method)
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    correct_pred = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    with torch.no_grad():
        for data in loader:
            images, labels, _, _ ,_ = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = expert_fn(images, labels)
            for i in range(0, batch_size):
                r = (predicted[i].item() == n_classes)
                prediction = predicted[i]
                final_pred = 0
                if predicted[i] == n_classes:
                    max_idx = 0
                    # get second max
                    for j in range(0, n_classes):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    final_pred = predicted[i]
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    final_pred = exp_prediction[i]
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
                if labels[i].item() == final_pred:
                    correct_pred[cifar_classes[labels[i].item()]] += 1
                total_pred[cifar_classes[labels[i].item()]] += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    print(to_print)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.3f} %".format(classname,
                                                    accuracy))
    return to_print
def metrics_print_oracle(net_class, expert_fn, expert_k, n_classes, loader):
        '''
    Computes metrics for Oracle method (defer when expert is correct) 
    net_mod: classifier model
    expert_fn: actual synth expert
    expert_k: number of classes expert can predict
    n_classes: number of classes
    loader: data loader
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    correct_pred = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    with torch.no_grad():
        for data in loader:
            images, labels, _, _ ,_ = data
            images, labels = images.to(device), labels.to(device)
            outputs_class = net_class(images)
            _, predicted = torch.max(outputs_class.data, 1)
            batch_size = outputs_class.size()[0]  # batch_size

            exp_prediction = expert_fn(images, labels)
            for i in range(0, batch_size):
                r = (expert_k >= labels[i].item()) 
                final_pred = 0
                #r = (exp_prediction[i] == labels[i].item()), this has noise
                if r == 0:
                    total += 1
                    prediction = predicted[i]
                    if predicted[i] == n_classes:
                        max_idx = 0
                        for j in range(0, n_classes):
                            if outputs_class.data[i][j] >= outputs_class.data[i][max_idx]:
                                max_idx = j
                        prediction = max_idx
                    else:
                        prediction = predicted[i]
                    final_pred = prediction
                    correct += (prediction == labels[i]).item()
                    correct_sys += (prediction == labels[i]).item()
                if r == 1:
                    final_pred = exp_prediction[i]
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
                if labels[i].item() == final_pred:
                    correct_pred[cifar_classes[labels[i].item()]] += 1
                total_pred[cifar_classes[labels[i].item()]] += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001)}
    print(to_print)
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.3f} %".format(classname,
                                                    accuracy))



def metrics_print_classifier(model, data_loader, defer_net = False):
    '''
    Prints metrics for classifier on label (no deferral)
    model: model
    data_loader: data loader
    defer_net: boolean to indicate if model is a deferral module (has n_classes +1 outputs)
    '''
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    correct = 0
    total = 0
    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, labels, _, _ ,_ = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1) # maybe no .data
            if defer_net:
                predictions_fixed = predictions
                for i in range(len(predictions_fixed)):
                    if predictions_fixed[i] == 10: #max class
                        max_idx = 0
                        # get second max
                        for j in range(0, 10):
                            if outputs.data[i][j] >= outputs.data[i][max_idx]:
                                max_idx = j
                        prediction = max_idx
                        predictions_fixed[i] = prediction
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[cifar_classes[label]] += 1
                total_pred[cifar_classes[label]] += 1

    print('Accuracy of the network on the %d test images: %.3f %%' % (len(data_loader),
        100 * correct / total))
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.3f} %".format(classname,
                                                    accuracy))
                                                    
                                                    
def metrics_print_expert(model, data_loader, defer_net = False):
    '''
    Computes metrics for expert model error prediction
    model: model
    data_loader: data loader
    '''
    correct = 0
    total = 0
    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, label, expert_pred, _ ,_ = data
            expert_pred = expert_pred.long()
            expert_pred = (expert_pred == label) *1
            images, labels = images.to(device), expert_pred.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1) # maybe no .data

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    print('Accuracy of the network on the %d test images: %.3f %%' % (total,
        100 * correct / total))
