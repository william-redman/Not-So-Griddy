#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:31:13 2024

@author: redmawt1
"""

import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import argparse
from utils import generate_run_ID
import random
from visualize import compute_ratemaps, compute_ratemaps_single_agent, plot_ratemaps
from tqdm import tqdm
from scores import GridScorer

random.seed(0)

# Globals
n_seeds = 5
n_finetune_epochs = 10
save_flag = False
save_path = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/Figures/Plotting network generalization/'
model_dir = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/models/'

# Loading single agent task decoding errors
single_agent_decoding_error = np.zeros([3, n_seeds, 2 * n_finetune_epochs])
single_agent_loss = np.zeros([3, n_seeds, 2 * n_finetune_epochs])

for ii in range(n_seeds):
    # Original model (asympotics)
    model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
    l = np.load(model_dir + model_folder + model_parameters + 'loss.npy')
    single_agent_decoding_error[0, ii, :] = de[-1] * np.ones(2 * n_finetune_epochs)
    single_agent_loss[0, ii, :] = l[-1] * np.ones(2 * n_finetune_epochs)
        
    # Dual agent model (all weights fine-tuned)
    model_folder = 'Single agent path integration dual agent all weights finetuned/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
    l = np.load(model_dir + model_folder + model_parameters + 'loss.npy')
    single_agent_decoding_error[1, ii, :] = de[np.arange(0, len(de), 500)][:(2 * n_finetune_epochs)]
    single_agent_loss[1, ii, :] = l[np.arange(0, len(l), 500)][:(2 * n_finetune_epochs)]
    
    # Dual agent model (RNN weights frozen)
    model_folder = 'Single agent path integration dual agent RNN weights frozen/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
    l = np.load(model_dir + model_folder + model_parameters + 'loss.npy')
    single_agent_decoding_error[2, ii, :] = de[np.arange(0, len(de), 500)][:(2 * n_finetune_epochs)]
    single_agent_loss[2, ii, :] = l[np.arange(0, len(l), 500)][:(2 * n_finetune_epochs)]
        
# Loading dual agent task decoding errors
dual_agent_decoding_error = np.zeros([4, n_seeds, 2 * n_finetune_epochs])
dual_agent_loss = np.zeros([4, n_seeds, 2 * n_finetune_epochs])

for ii in range(n_seeds):
    # Original model (asympotics and random initialization)
    model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
    l =  np.load(model_dir + model_folder + model_parameters + 'loss.npy')
    dual_agent_decoding_error[0, ii, :] = de[-1] * np.ones(2 * n_finetune_epochs)
    dual_agent_decoding_error[1, ii, :] = de[:20]
    dual_agent_loss[0, ii, :] = l[-1] * np.ones(2 * n_finetune_epochs)
    dual_agent_loss[1, ii, :] = l[np.arange(0, len(l), 500)][:(2 * n_finetune_epochs)]
       
    # Single agent model (all weights fine-tuned)
    model_folder = 'Dual agent path integration single agent all weights finetuned/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
    l = np.load(model_dir + model_folder + model_parameters + 'loss.npy')
    dual_agent_decoding_error[2, ii, :] = de
    dual_agent_loss[2, ii, :] = l[np.arange(0, len(l), 500)][:(2 * n_finetune_epochs)]
    
    
    # Single agent model (RNN weights frozen)
    model_folder = 'Dual agent path integration single agent RNN weights frozen/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
    l = np.load(model_dir + model_folder + model_parameters + 'loss.npy')
    dual_agent_decoding_error[3, ii, :] = de  
    dual_agent_loss[3, ii, :] = l[np.arange(0, len(l), 500)][:(2 * n_finetune_epochs)]
    
# Plotting 
fig = plt.figure()
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(single_agent_decoding_error[0, :, :], axis = 0), np.max(single_agent_decoding_error[0, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(single_agent_decoding_error[0, :, :], axis = 0), '-', color = [0, 1, 0], label='Train: single (asymptotic)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(single_agent_decoding_error[1, :, :], axis = 0), np.max(single_agent_decoding_error[1, :, :], axis = 0), alpha=0.5, color = [0.9, 0.5, 0.5])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(single_agent_decoding_error[1, :, :], axis = 0), '-', color = [0.9, 0.5, 0.5], label='Train: dual; FT: single (all weights)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(single_agent_decoding_error[2, :, :], axis = 0), np.max(single_agent_decoding_error[2, :, :], axis = 0), alpha=0.5, color = [0.8, 0.3, 0.2])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(single_agent_decoding_error[2, :, :], axis = 0), '--', color = [0.8, 0.3, 0.2], label='Train: dual; FT: single (RNN frozen)')
plt.legend()
plt.ylabel('Decoding error')
plt.xlabel('Epochs')
plt.yscale('log')
if save_flag:
    plt.savefig(save_path + 'Decoding_error_single_agent_task.svg', format = 'svg')
plt.show()
    
fig = plt.figure()
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_decoding_error[0, :, :], axis = 0), np.max(dual_agent_decoding_error[0, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_decoding_error[0, :, :], axis = 0), '-', color = [0, 1, 0], label='Train: dual (asymptotic)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_decoding_error[1, :, :], axis = 0), np.max(dual_agent_decoding_error[1, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_decoding_error[1, :, :], axis = 0), '--', color = [0, 1, 0], label='Train: dual (random)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_decoding_error[2, :, :], axis = 0), np.max(dual_agent_decoding_error[2, :, :], axis = 0), alpha=0.5, color = [0.9, 0.5, 0.5])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_decoding_error[2, :, :], axis = 0), '-', color = [0.9, 0.5, 0.5], label='Train: single; FT: dual (all weights)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_decoding_error[3, :, :], axis = 0), np.max(dual_agent_decoding_error[3, :, :], axis = 0), alpha=0.5, color = [0.8, 0.3, 0.2])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_decoding_error[3, :, :], axis = 0), '--', color = [0.8, 0.3, 0.2], label='Train: dual; FT: dual (RNN frozen)')
plt.legend()
plt.ylabel('Decoding error')
plt.xlabel('Epochs')
plt.yscale('log')
if save_flag:
    plt.savefig(save_path + 'Decoding_error_dual_agent_task.svg', format = 'svg')
plt.show()

fig = plt.figure()
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(single_agent_loss[0, :, :], axis = 0), np.max(single_agent_loss[0, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(single_agent_loss[0, :, :], axis = 0), '-', color = [0, 1, 0], label='Train: single (asymptotic)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(single_agent_loss[1, :, :], axis = 0), np.max(single_agent_loss[1, :, :], axis = 0), alpha=0.5, color = [0.9, 0.5, 0.5])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(single_agent_loss[1, :, :], axis = 0), '-', color = [0.9, 0.5, 0.5], label='Train: dual; FT: single (all weights)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(single_agent_loss[2, :, :], axis = 0), np.max(single_agent_loss[2, :, :], axis = 0), alpha=0.5, color = [0.8, 0.3, 0.2])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(single_agent_loss[2, :, :], axis = 0), '--', color = [0.8, 0.3, 0.2], label='Train: dual; FT: single (RNN frozen)')
plt.legend()
plt.ylabel('Training loss')
plt.xlabel('Epochs')
if save_flag:
    plt.savefig(save_path + 'Loss_single_agent_task.svg', format = 'svg')
plt.show()

fig = plt.figure()
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_loss[0, :, :], axis = 0), np.max(dual_agent_loss[0, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_loss[0, :, :], axis = 0), '-', color = [0, 1, 0], label='Train: dual (asymptotic)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_loss[1, :, :], axis = 0), np.max(dual_agent_loss[1, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_loss[1, :, :], axis = 0), '--', color = [0, 1, 0], label='Train: dual (random)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_loss[2, :, :], axis = 0), np.max(dual_agent_loss[2, :, :], axis = 0), alpha=0.5, color = [0.9, 0.5, 0.5])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_loss[2, :, :], axis = 0), '-', color = [0.9, 0.5, 0.5], label='Train: single; FT: dual (all weights)')
plt.fill_between(np.arange(0, n_finetune_epochs, 0.5),  np.min(dual_agent_loss[3, :, :], axis = 0), np.max(dual_agent_loss[3, :, :], axis = 0), alpha=0.5, color = [0.8, 0.3, 0.2])
plt.plot(np.arange(0, n_finetune_epochs, 0.5),  np.mean(dual_agent_loss[3, :, :], axis = 0), '--', color = [0.8, 0.3, 0.2], label='Train: dual; FT: dual (RNN frozen)')
plt.legend()
plt.ylabel('Training loss')
plt.xlabel('Epochs')
if save_flag:
    plt.savefig(save_path + 'Loss_dual_agent_task.svg', format = 'svg')
plt.show()




















