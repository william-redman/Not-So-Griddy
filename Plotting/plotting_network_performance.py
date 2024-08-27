#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:31:08 2023

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
import os

# Parameters necessary to evaluate network
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=1,
                    help='number of training epochs') #100
parser.add_argument('--n_steps',
                    default= 1,
                    help='batches per epoch') #1000
parser.add_argument('--batch_size',
                    default=1000,
                    help='number of trajectories per batch') #200
parser.add_argument('--sequence_length',
                    default=20,
                    help='number of steps in trajectory') #20
parser.add_argument('--learning_rate',
                    default=1e-4,
                    help='gradient descent learning rate') #1e-4
parser.add_argument('--Np',
                    default=512, 
                    help='number of place cells') #512
parser.add_argument('--Ng',
                    default=4096,
                    help='number of grid cells') #4096
parser.add_argument('--place_cell_rf',
                    default=0.12,
                    help='width of place cell center tuning curve (m)') #0.12
parser.add_argument('--surround_scale',
                    default=2,
                    help='if DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN or LSTM')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--weight_decay',
                    default=1e-6,
                    help='strength of weight decay on recurrent weights') #1e-4
parser.add_argument('--DoG',
                    default=True,
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default=False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    default= 2.2, 
                    help='width of training environment') #2.2
parser.add_argument('--box_height',
                    default=2.2, 
                    help='height of training environment') #2.2
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training')

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f'Using device: {options.device}')

random.seed(0)

# Globals
n_seeds = 5
save_flag = False
save_path = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/Figures/Plotting network performance/'
model_dir = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/models/'

# Loading losses and decoding errors
training_loss = np.nan * np.zeros([4, n_seeds, 200]) # number of network types x number of seeds x number of time steps with saved loss and decoding error
decoding_error = np.nan * np.zeros([4, n_seeds, 200])

for ii in range(n_seeds):
    model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-04/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_00001/' 
    
    if os.path.isfile(model_dir + model_folder + model_parameters + 'loss.npy'):  
        tl = np.load(model_dir + model_folder + model_parameters + 'loss.npy')    
        training_loss[0, ii, :] = tl[np.arange(0, len(tl), 500)]

        de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
        decoding_error[0, ii, :] = de[np.arange(0, len(tl), 500)]

for ii in range(n_seeds):
    model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    
    if os.path.isfile(model_dir + model_folder + model_parameters + 'loss.npy'):
        tl = np.load(model_dir + model_folder + model_parameters + 'loss.npy')    
        training_loss[1, ii, :] = tl[np.arange(0, len(tl), 500)]

        de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
        decoding_error[1, ii, :] = de[np.arange(0, len(tl), 500)]

for ii in range(n_seeds):
    model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-04/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_00001/' 
    
    if os.path.isfile(model_dir + model_folder + model_parameters + 'loss.npy'):
        tl = np.load(model_dir + model_folder + model_parameters + 'loss.npy')    
        training_loss[2, ii, :] = tl[np.arange(0, len(tl), 500)]

        de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
        decoding_error[2, ii, :] = de

for ii in range(n_seeds):
    model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    
    if os.path.isfile(model_dir + model_folder + model_parameters + 'loss.npy'):
        tl = np.load(model_dir + model_folder + model_parameters + 'loss.npy')    
        training_loss[3, ii, :] = tl[np.arange(0, len(tl), 500)]

        de = np.load(model_dir + model_folder + model_parameters + 'decoding_error.npy')
        decoding_error[3, ii, :] = de

# Generating samples of path integration errors
decoding_error_final = np.zeros([2, n_seeds, options.batch_size])
pos = np.zeros([n_seeds, options.sequence_length, options.batch_size, 4])
pred_pos = np.zeros([n_seeds, options.sequence_length, options.batch_size, 4])
distance =  np.zeros([n_seeds, options.batch_size])

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
place_cells = PlaceCells(options)
if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
elif options.RNN_type == 'LSTM':
    raise NotImplementedError
model = model.to(options.device)
trajectory_generator = TrajectoryGenerator(options, place_cells)

for ii in range(n_seeds):
    model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    model_name = 'final_model.pth'
    saved_model = torch.load(model_dir + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model)

    inputs, p, pc_outputs = trajectory_generator.get_test_batch()
    pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
    us = place_cells.us.cpu()
        
    e = torch.sqrt(((p - pp)**2).sum(-1)).median(dim = 0).values
    e = e.numpy()
    decoding_error_final[0, ii, :] = e

from place_cells_dual_path_integration import PlaceCells
from trajectory_generator_dual_path_integration import TrajectoryGenerator
from model_dual_path_integration import RNN
from trainer_dual_path_integration import Trainer
place_cells = PlaceCells(options)
if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
elif options.RNN_type == 'LSTM':
    raise NotImplementedError
model = model.to(options.device)
trajectory_generator = TrajectoryGenerator(options, place_cells)

for ii in range(n_seeds):
    model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    model_name = 'final_model.pth'
    saved_model = torch.load(model_dir + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model)

    inputs, p, pc_outputs = trajectory_generator.get_test_batch()
    pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
    us = place_cells.us.cpu()
      
    d = torch.sqrt(((p[:, :, :2] - p[:, :, 2:])**2).sum(-1)).median(dim = 0).values
    distance[ii, :] = d
    
    pos[ii, :, :, :] = p
    pred_pos[ii, :, :, :] = pp
    
    dist = torch.sqrt(((p - pp)**2).sum(-1))
    dist_flip = torch.sqrt(((p - pp[:, :,[2, 3, 0, 1]])**2).sum(-1))
    dist_stacked = torch.stack([dist, dist_flip], axis = -1)
    min_dist, _ = torch.min(dist_stacked, dim = -1)
    e = min_dist.median(dim = 0).values
    e = e.numpy()
    decoding_error_final[1, ii, :] = e

p0 = np.percentile(decoding_error_final[1, :, :], 0)
id0 = np.argwhere(np.abs(decoding_error_final[1, :, :] - p0) == np.min(np.abs(decoding_error_final[1, :, :] - p0)))[0]
p25 = np.percentile(decoding_error_final[1, :, :], 25)
id25 = np.argwhere(np.abs(decoding_error_final[1, :, :] - p25) == np.min(np.abs(decoding_error_final[1, :, :] - p25)))[0]
p50 = np.percentile(decoding_error_final[1, :, :], 50)
id50 = np.argwhere(np.abs(decoding_error_final[1, :, :] - p50) == np.min(np.abs(decoding_error_final[1, :, :] - p50)))[0]
p75 = np.percentile(decoding_error_final[1, :, :], 75)
id75 = np.argwhere(np.abs(decoding_error_final[1, :, :] - p75) == np.min(np.abs(decoding_error_final[1, :, :] - p75)))[0]

# Plotting
fig = plt.figure()
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(training_loss[0, :, :], axis = 0), np.nanmax(training_loss[0, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(training_loss[0, :, :], axis = 0), '-', color = [0, 1, 0],  label = 'Single agent: reg. $10^{-5}$')
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(training_loss[1, :, :], axis = 0), np.nanmax(training_loss[1, :, :], axis = 0), alpha=0.5, color = [0, 0.5, 0])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(training_loss[1, :, :], axis = 0), '-', color = [0, 0.5, 0],  label = 'Single agent: reg. $10^{-7}$')
plt.legend()
plt.ylabel('Training loss')
plt.xlabel('Epochs')
if save_flag:
    plt.savefig(save_path + 'Single_agent_loss.svg', format = 'svg')
plt.show()

fig = plt.figure()
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(training_loss[2, :, :], axis = 0), np.nanmax(training_loss[2, :, :], axis = 0), alpha=0.5, color = [0.9, 0.5, 0.5])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(training_loss[2, :, :], axis = 0), '-', color = [0.9, 0.5, 0.5], label = 'Dual agent: reg. $10^{-5}$')
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(training_loss[3, :, :], axis = 0), np.nanmax(training_loss[3, :, :], axis = 0), alpha=0.5, color = [0.8, 0.3, 0.2])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(training_loss[3, :, :], axis = 0), '-', color = [0.8, 0.3, 0.2], label = 'Dual agent: reg. $10^{-7}$')
plt.legend()
plt.ylabel('Training loss')
plt.xlabel('Epochs')
if save_flag:
    plt.savefig(save_path + 'Dual_agent_loss.svg', format = 'svg')
plt.show()

fig = plt.figure()
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(decoding_error[0, :, :], axis = 0), np.nanmax(decoding_error[0, :, :], axis = 0), alpha=0.5, color = [0, 1, 0])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(decoding_error[0, :, :], axis = 0), '-', color = [0, 1, 0], label = 'Single agent: reg. $10^{-5}$')
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(decoding_error[1, :, :], axis = 0), np.nanmax(decoding_error[1, :, :], axis = 0), alpha=0.5, color = [0, 0.5, 0])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(decoding_error[1, :, :], axis = 0), '-', color = [0, 0.5, 0], label = 'Single agent: reg. $10^{-7}$')
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(decoding_error[2, :, :], axis = 0), np.nanmax(decoding_error[2, :, :], axis = 0), alpha=0.5, color = [0.9, 0.5, 0.5])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(decoding_error[2, :, :], axis = 0), '-', color = [0.9, 0.5, 0.5], label = 'Dual agent: reg. $10^{-5}$')
plt.fill_between(np.arange(0, 100, 0.5),  np.nanmin(decoding_error[3, :, :], axis = 0), np.nanmax(decoding_error[3, :, :], axis = 0), alpha=0.5, color = [0.8, 0.3, 0.2])
plt.plot(np.arange(0, 100, 0.5), np.nanmean(decoding_error[3, :, :], axis = 0), '-', color = [0.8, 0.3, 0.2], label = 'Dual agent: reg. $10^{-7}$')
plt.legend()
plt.ylabel('Decoding error')
plt.xlabel('Epochs')
if save_flag:
    plt.savefig(save_path + 'Decoding_error.svg', format = 'svg')
plt.show()

fig = plt.figure()
plt.hist(decoding_error_final[0, :, :].flatten(), range=(0, 0.5), bins=15, alpha=0.5, color=[0, 0.5, 0], label = 'Single agent: reg. $10^{-7}$')
plt.hist(decoding_error_final[1, :, :].flatten(), range=(0, 0.5), bins=15, alpha=0.5, color=[0.8, 0.3, 0.2], label = 'Single agent: reg. $10^{-7}$')
plt.xlabel('Median decoding error (m.)')
plt.ylabel('Count')
plt.legend()
if save_flag:
    plt.savefig(save_path + 'Decoding_error_histogram.svg', format = 'svg')
plt.show()

plt.figure(figsize=(6,6))
plt.subplot(2, 2, 1)
plt.plot(pos[id0[0], :, id0[1], 0], pos[id0[0], :, id0[1], 1], 'k.', label='True position')
plt.plot(pred_pos[id0[0], :, id0[1], 0], pred_pos[id0[0], :, id0[1], 1], 'r.', label='Decoded position')
plt.plot(pos[id0[0], :, id0[1], 2], pos[id0[0], :, id0[1], 3], 'k.')
plt.plot(pred_pos[id0[0], :, id0[1], 2], pred_pos[id0[0], :, id0[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$0^{th}$ percentile: err. = ' + str(np.around(decoding_error_final[1, id0[0], id0[1]], decimals = 2)) + ' m.')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(pos[id25[0], :, id25[1], 0], pos[id25[0], :, id25[1], 1], 'k.', label='True position')
plt.plot(pred_pos[id25[0], :, id25[1], 0], pred_pos[id25[0], :, id25[1], 1], 'r.', label='Decoded position')
plt.plot(pos[id25[0], :, id25[1], 2], pos[id25[0], :, id25[1], 3], 'k.')
plt.plot(pred_pos[id25[0], :, id25[1], 2], pred_pos[id25[0], :, id25[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$25^{th}$ percentile: err. = ' + str(np.around(decoding_error_final[1, id25[0], id25[1]], decimals = 2)) + ' m.')

plt.subplot(2, 2, 3)
plt.plot(pos[id50[0], :, id50[1], 0], pos[id50[0], :, id50[1], 1], 'k.', label='True position')
plt.plot(pred_pos[id50[0], :, id50[1], 0], pred_pos[id50[0], :, id50[1], 1], 'r.', label='Decoded position')
plt.plot(pos[id50[0], :, id50[1], 2], pos[id50[0], :, id50[1], 3], 'k.')
plt.plot(pred_pos[id50[0], :, id50[1], 2], pred_pos[id50[0], :, id50[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$50^{th}$ percentile: err. = ' + str(np.around(decoding_error_final[1, id50[0], id50[1]], decimals = 2)) + ' m.')

plt.subplot(2, 2, 4)
plt.plot(pos[id75[0], :, id75[1], 0], pos[id75[0], :, id75[1], 1], 'k.', label='True position')
plt.plot(pred_pos[id75[0], :, id75[1], 0], pred_pos[id75[0], :, id75[1], 1], 'r.', label='Decoded position')
plt.plot(pos[id75[0], :, id75[1], 2], pos[id75[0], :, id75[1], 3], 'k.')
plt.plot(pred_pos[id75[0], :, id75[1], 2], pred_pos[id75[0], :, id75[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$75^{th}$ percentile: err. = ' + str(np.around(decoding_error_final[1, id75[0], id75[1]], decimals = 2)) + ' m.')

if save_flag:
    plt.savefig(save_path + 'Example_trajectories.svg', format = 'svg')
plt.show()

plt.plot(distance.flatten(), decoding_error_final[1, :, :].flatten(), 'k.')
plt.xlabel('Median distance between agents')
plt.ylabel('Median decoding error')
plt.show()


near_trajs = np.argwhere(distance.flatten() < 0.1)
plt.hist(distance.flatten()[near_trajs], range=(0, 0.25), bins=20, alpha=0.5, color=[0.5, 0, 0], label='Near trajectories')
plt.xlabel('Median distance between agents')
plt.ylabel('Median decoding error')
if save_flag:
    plt.savefig(save_path + 'Nearby_trajectories_decoding_error.svg', format = 'svg')
plt.show()

nearest_dist = np.sort(distance.flatten())[:4]
close_id0 = np.argwhere(np.abs(distance - nearest_dist[0]) == np.min(np.abs(distance - nearest_dist[0])))[0]
close_id1 = np.argwhere(np.abs(distance - nearest_dist[1]) == np.min(np.abs(distance - nearest_dist[1])))[0]
close_id2 = np.argwhere(np.abs(distance - nearest_dist[2]) == np.min(np.abs(distance - nearest_dist[2])))[0]
close_id3 = np.argwhere(np.abs(distance - nearest_dist[3]) == np.min(np.abs(distance - nearest_dist[3])))[0]

plt.figure(figsize=(6,6))
plt.subplot(2, 2, 1)
plt.plot(pos[close_id0[0], :, close_id0[1], 0], pos[close_id0[0], :, close_id0[1], 1], 'k.', label='True position')
plt.plot(pred_pos[close_id0[0], :, close_id0[1], 0], pred_pos[close_id0[0], :, close_id0[1], 1], 'r.', label='Decoded position')
plt.plot(pos[close_id0[0], :, close_id0[1], 2], pos[close_id0[0], :, close_id0[1], 3], 'k.')
plt.plot(pred_pos[close_id0[0], :, close_id0[1], 2], pred_pos[close_id0[0], :, close_id0[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$err. = ' + str(np.around(decoding_error_final[1, close_id0[0], close_id0[1]], decimals = 2)) + ' m.')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(pos[close_id1[0], :, close_id1[1], 0], pos[close_id1[0], :, close_id1[1], 1], 'k.', label='True position')
plt.plot(pred_pos[close_id1[0], :, close_id1[1], 0], pred_pos[close_id1[0], :, close_id1[1], 1], 'r.', label='Decoded position')
plt.plot(pos[close_id1[0], :, close_id1[1], 2], pos[close_id1[0], :, close_id1[1], 3], 'k.')
plt.plot(pred_pos[close_id1[0], :, close_id1[1], 2], pred_pos[close_id1[0], :, close_id1[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$err. = ' + str(np.around(decoding_error_final[1, close_id1[0], close_id1[1]], decimals = 2)) + ' m.')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(pos[close_id2[0], :, close_id2[1], 0], pos[close_id2[0], :, close_id2[1], 1], 'k.', label='True position')
plt.plot(pred_pos[close_id2[0], :, close_id2[1], 0], pred_pos[close_id2[0], :, close_id2[1], 1], 'r.', label='Decoded position')
plt.plot(pos[close_id2[0], :, close_id2[1], 2], pos[close_id2[0], :, close_id2[1], 3], 'k.')
plt.plot(pred_pos[close_id2[0], :, close_id2[1], 2], pred_pos[close_id2[0], :, close_id2[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$err. = ' + str(np.around(decoding_error_final[1, close_id2[0], close_id2[1]], decimals = 2)) + ' m.')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(pos[close_id3[0], :, close_id3[1], 0], pos[close_id3[0], :, close_id3[1], 1], 'k.', label='True position')
plt.plot(pred_pos[close_id3[0], :, close_id3[1], 0], pred_pos[close_id3[0], :, close_id3[1], 1], 'r.', label='Decoded position')
plt.plot(pos[close_id3[0], :, close_id3[1], 2], pos[close_id3[0], :, close_id3[1], 3], 'k.')
plt.plot(pred_pos[close_id3[0], :, close_id3[1], 2], pred_pos[close_id3[0], :, close_id3[1], 3], 'r.')
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2])
plt.title('$err. = ' + str(np.around(decoding_error_final[1, close_id3[0], close_id3[1]], decimals = 2)) + ' m.')
plt.legend()

if save_flag:
    plt.savefig(save_path + 'Example_close_trajectories.svg', format = 'svg')
plt.show()

















