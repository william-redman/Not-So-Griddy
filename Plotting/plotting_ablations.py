#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:08:16 2024

@author: redmawt1
"""

import numpy as np
#import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import argparse
from utils import generate_run_ID
import random
from visualize import compute_ratemaps, compute_ratemaps_single_agent, compute_relative_ratemaps, plot_ratemaps
from tqdm import tqdm
from scores import GridScorer

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
n_g = 4096
n_shuffle = 10
save_flag = False
parent_dir = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/'
fig_path = parent_dir + 'Figures/Plotting ablations/'
result_path = parent_dir + 'Results/'
model_path = parent_dir + 'models/'

# Loading functional class scores
grid_scores = np.load(result_path + 'Functional classes/grid_scores.npy')
band_scores = np.load(result_path + 'Functional classes/band_scores.npy')
border_scores = np.load(result_path + 'Functional classes/border_scores.npy')

# Performing ablations
ablate_percentiles =  [0, 5, 10, 15, 20, 25]

grid_cell_ablations =  np.zeros((2, n_seeds, len(ablate_percentiles)))
band_cell_ablations =  np.zeros((2, n_seeds, len(ablate_percentiles)))
border_cell_ablations =  np.zeros((2, n_seeds, len(ablate_percentiles)))
random_shuffle_ablations = np.zeros((2, n_seeds, n_shuffle, len(ablate_percentiles)))

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

print('Ablating single agent RNNs')
for ii in range(n_seeds):
    # Ablate grid cells
    print('Ablating grid cells')
    for jj in range(len(ablate_percentiles)):
        model = RNN(options, place_cells)
        model = model.to(options.device)
        model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
        model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
        model_name = 'final_model.pth'
        saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
        
        ablate_ids = np.argwhere(grid_scores[0, ii, :] > np.nanpercentile(grid_scores[0, ii, :], 100 - ablate_percentiles[jj]))    
        saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
        
        model.load_state_dict(saved_model)

        inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
        us = place_cells.us.cpu()

        e = torch.sqrt(((p - pp)**2).sum(-1)).median(dim = 0).values
        e = e.numpy()
        grid_cell_ablations[0, ii, jj] = np.mean(e)

    # Ablate border cells
    print('Ablating border cells')
    for jj in range(len(ablate_percentiles)):
        model = RNN(options, place_cells)
        model = model.to(options.device)
        model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
        model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
        model_name = 'final_model.pth'
        saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
        
        ablate_ids = np.argwhere(border_scores[0, ii, :] > np.nanpercentile(border_scores[0, ii, :], 100 - ablate_percentiles[jj]))    
        saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
        
        model.load_state_dict(saved_model)

        inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
        us = place_cells.us.cpu()

        e = torch.sqrt(((p - pp)**2).sum(-1)).median(dim = 0).values
        e = e.numpy()
        border_cell_ablations[0, ii, jj] = np.mean(e)

    # Ablate band cells
    print('Ablating band cells')
    for jj in range(len(ablate_percentiles)):
        model = RNN(options, place_cells)
        model = model.to(options.device)
        model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
        model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
        model_name = 'final_model.pth'
        saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
        
        ablate_ids = np.argwhere(band_scores[0, ii, :] > np.nanpercentile(band_scores[0, ii, :], 100 - ablate_percentiles[jj]))    
        saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
        
        model.load_state_dict(saved_model)

        inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
        us = place_cells.us.cpu()

        e = torch.sqrt(((p - pp)**2).sum(-1)).median(dim = 0).values
        e = e.numpy()
        band_cell_ablations[0, ii, jj] = np.mean(e)

    # Ablating random shuffle
    print('Ablating random shuffle')
    for jj in range(len(ablate_percentiles)):
        rand_ablations = np.zeros(n_shuffle)
        for kk in range(n_shuffle):
            model = RNN(options, place_cells)
            model = model.to(options.device)
            model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
            model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
            model_name = 'final_model.pth'
            saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
            
            rand_score = np.random.rand(n_g)
            ablate_ids = np.argwhere(rand_score > np.nanpercentile(rand_score, 100 - ablate_percentiles[jj]))    
            saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
            saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
            
            model.load_state_dict(saved_model)

            inputs, p, pc_outputs = trajectory_generator.get_test_batch()
            pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
            us = place_cells.us.cpu()

            e = torch.sqrt(((p - pp)**2).sum(-1)).median(dim = 0).values
            e = e.numpy()
            rand_ablations[kk] = np.mean(e)
        random_shuffle_ablations[0, ii, :, jj] = rand_ablations

from place_cells_dual_path_integration import PlaceCells
from trajectory_generator_dual_path_integration import TrajectoryGenerator
from model_dual_path_integration import RNN
from trainer_dual_path_integration import Trainer
place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

print('Ablating dual agent RNNs')
for ii in range(n_seeds):
    # Ablate grid cells
    print('Ablating grid cells')
    for jj in range(len(ablate_percentiles)):
        model = RNN(options, place_cells)
        model = model.to(options.device)
        model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
        model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
        model_name = 'final_model.pth'
        saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
        
        ablate_ids = np.argwhere(grid_scores[1, ii, :] > np.nanpercentile(grid_scores[1, ii, :], 100 - ablate_percentiles[jj]))    
        saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
        
        model.load_state_dict(saved_model)

        inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
        us = place_cells.us.cpu()

        dist = torch.sqrt(((p - pp)**2).sum(-1))
        dist_flip = torch.sqrt(((p - pp[:, :,[2, 3, 0, 1]])**2).sum(-1))
        dist_stacked = torch.stack([dist, dist_flip], axis = -1)
        min_dist, _ = torch.min(dist_stacked, dim = -1)
        e = min_dist.median(dim = 0).values
        e = e.numpy()
        grid_cell_ablations[1, ii, jj] = np.mean(e)

    # Ablate border cells
    print('Ablating border cells')
    for jj in range(len(ablate_percentiles)):
        model = RNN(options, place_cells)
        model = model.to(options.device)
        model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
        model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
        model_name = 'final_model.pth'
        saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
        
        ablate_ids = np.argwhere(border_scores[1, ii, :] > np.nanpercentile(border_scores[1, ii, :], 100 - ablate_percentiles[jj]))    
        saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
        
        model.load_state_dict(saved_model)

        inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
        us = place_cells.us.cpu()

        dist = torch.sqrt(((p - pp)**2).sum(-1))
        dist_flip = torch.sqrt(((p - pp[:, :,[2, 3, 0, 1]])**2).sum(-1))
        dist_stacked = torch.stack([dist, dist_flip], axis = -1)
        min_dist, _ = torch.min(dist_stacked, dim = -1)
        e = min_dist.median(dim = 0).values
        e = e.numpy()
        border_cell_ablations[1, ii, jj] = np.mean(e)

    # Ablate band cells
    print('Ablating band cells')
    for jj in range(len(ablate_percentiles)):
        model = RNN(options, place_cells)
        model = model.to(options.device)
        model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
        model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
        model_name = 'final_model.pth'
        saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
        
        ablate_ids = np.argwhere(band_scores[1, ii, :] > np.nanpercentile(band_scores[1, ii, :], 100 - ablate_percentiles[jj]))    
        saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
        saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
        
        model.load_state_dict(saved_model)

        inputs, p, pc_outputs = trajectory_generator.get_test_batch()
        pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
        us = place_cells.us.cpu()

        dist = torch.sqrt(((p - pp)**2).sum(-1))
        dist_flip = torch.sqrt(((p - pp[:, :,[2, 3, 0, 1]])**2).sum(-1))
        dist_stacked = torch.stack([dist, dist_flip], axis = -1)
        min_dist, _ = torch.min(dist_stacked, dim = -1)
        e = min_dist.median(dim = 0).values
        e = e.numpy()
        band_cell_ablations[1, ii, jj] = np.mean(e)

    # Ablating random shuffle
    print('Ablating random shuffle')
    for jj in range(len(ablate_percentiles)):
        rand_ablations = np.zeros(n_shuffle)
        for kk in range(n_shuffle):
            model = RNN(options, place_cells)
            model = model.to(options.device)
            model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
            model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
            model_name = 'final_model.pth'
            saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
            
            rand_score = np.random.rand(n_g)
            ablate_ids = np.argwhere(rand_score > np.nanpercentile(rand_score, 100 - ablate_percentiles[jj]))    
            saved_model['RNN.weight_hh_l0'][:, ablate_ids] = 0
            saved_model['RNN.weight_hh_l0'][ablate_ids, :] = 0
            
            model.load_state_dict(saved_model)

            inputs, p, pc_outputs = trajectory_generator.get_test_batch()
            pp = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
            us = place_cells.us.cpu()

            dist = torch.sqrt(((p - pp)**2).sum(-1))
            dist_flip = torch.sqrt(((p - pp[:, :,[2, 3, 0, 1]])**2).sum(-1))
            dist_stacked = torch.stack([dist, dist_flip], axis = -1)
            min_dist, _ = torch.min(dist_stacked, dim = -1)
            e = min_dist.median(dim = 0).values
            e = e.numpy()
            rand_ablations[kk] = np.mean(e)
        random_shuffle_ablations[1, ii, :, jj] = rand_ablations
     
# Saving ablation matrices
if save_flag: 
    np.save(result_path + 'Ablations/grid_cell_ablations.npy', grid_cell_ablations)
    np.save(result_path + 'Ablations/border_cell_ablations.npy', border_cell_ablations)     
    np.save(result_path + 'Ablations/band_cell_ablations.npy', band_cell_ablations) 
    np.save(result_path + 'Ablations/random_shuffle_ablations.npy', random_shuffle_ablations)


# Plotting 
plt.figure(figsize= (5, 4))
plt.plot(ablate_percentiles, np.mean(grid_cell_ablations[0, :, :], axis = 0), 'g-', label='Single agent grid score')
plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations[0, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations[0, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations[0, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations[0, :, :], axis = (0, 1)), color='g', alpha=0.5)
plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations[0, :, :], axis = (0, 1)), 'g--', label='Single agent shuff.')
plt.plot(ablate_percentiles, np.mean(grid_cell_ablations[1, :, :], axis = 0), 'k-', label='Dual agent grid score')
plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations[1, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations[1, :, :], axis = (0, 1)), color='k', alpha=0.5)
plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), 'k--', label='Dual agent shuff.')
plt.xlabel('Percentage of grid cells removed')
plt.ylabel('Decoding error (m.)')
plt.legend()
if save_flag:
    plt.savefig(fig_path + 'grid_cell_ablations.svg', format = 'svg')

plt.figure(figsize= (5, 4))
plt.plot(ablate_percentiles, np.mean(border_cell_ablations[0, :, :], axis = 0), 'g-', label='Single agent border score')
plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations[0, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations[0, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations[0, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations[0, :, :, :], axis = (0, 1)), color='g', alpha=0.5)
plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations[0, :, :, :], axis = (0, 1)), 'g--', label='Single agent shuff.')
plt.plot(ablate_percentiles, np.mean(border_cell_ablations[1, :, :], axis = 0), 'k-', label='Dual agent border score')
plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), color='k', alpha=0.5)
plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), 'k--', label='Dual agent shuff.')
plt.xlabel('Percentage of border cells removed')
plt.ylabel('Decoding error (m.)')
plt.legend()
if save_flag:
    plt.savefig(fig_path + 'border_cell_ablations.svg', format = 'svg')

plt.figure(figsize= (5, 4))
plt.plot(ablate_percentiles, np.mean(band_cell_ablations[0, :, :], axis = 0), 'g-', label='Single agent band score')
plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations[0, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations[0, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations[0, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations[0, :, :, :], axis = (0, 1)), color='g', alpha=0.5)
plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations[0, :, :, :], axis = (0, 1)), 'g--', label='Single agent shuff.')
plt.plot(ablate_percentiles, np.mean(band_cell_ablations[1, :, :], axis = 0), 'k-', label='Dual agent band score')
plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), color='k', alpha=0.5)
plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations[1, :, :, :], axis = (0, 1)), 'k--', label='Dual agent shuff.')
plt.xlabel('Percentage of band cells removed')
plt.ylabel('Decoding error (m.)')
plt.legend()
if save_flag:
    plt.savefig(fig_path + 'band_cell_ablations.svg', format = 'svg')



# # Normalize to unablated model
# grid_cell_ablations_norm = np.copy(grid_cell_ablations)
# border_cell_ablations_norm = np.copy(border_cell_ablations)
# band_cell_ablations_norm = np.copy(band_cell_ablations)
# random_shuffle_ablations_norm = np.copy(random_shuffle_ablations)

# grid_cell_ablations_norm[0, :, :] = np.reshape(grid_cell_ablations[0, :, 0], (n_seeds, 1)) / grid_cell_ablations[0, :, :] 
# grid_cell_ablations_norm[1, :, :] = np.reshape(grid_cell_ablations[1, :, 0], (n_seeds, 1)) / grid_cell_ablations[1, :, :] 
# border_cell_ablations_norm[0, :, :] = np.reshape(border_cell_ablations[0, :, 0], (n_seeds, 1)) / border_cell_ablations[0, :, :] 
# border_cell_ablations_norm[1, :, :] = np.reshape(border_cell_ablations[1, :, 0], (n_seeds, 1)) / border_cell_ablations[1, :, :] 
# band_cell_ablations_norm[0, :, :] = np.reshape(band_cell_ablations[0, :, 0], (n_seeds, 1)) / band_cell_ablations[0, :, :] 
# band_cell_ablations_norm[1, :, :] = np.reshape(band_cell_ablations[1, :, 0], (n_seeds, 1)) / band_cell_ablations[1, :, :] 
# random_shuffle_ablations_norm[0, :, :, :] = np.reshape(random_shuffle_ablations[0, :, :, 0], (n_seeds, n_shuffle, 1)) / random_shuffle_ablations[0, :, :, :] 
# random_shuffle_ablations_norm[1, :, :, :] = np.reshape(random_shuffle_ablations[1, :, :, 0], (n_seeds, n_shuffle, 1)) / random_shuffle_ablations[1, :, :, :] 

# # Plotting 
# plt.figure(figsize= (5, 4))
# plt.plot(ablate_percentiles, np.mean(grid_cell_ablations_norm[0, :, :], axis = 0), 'g-', label='Single agent grid score')
# plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations_norm[0, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations_norm[0, :, :], axis = (0, 1)), color='g', alpha=0.5)
# plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations_norm[0, :, :], axis = (0, 1)), 'g--', label='Single agent shuff.')
# plt.plot(ablate_percentiles, np.mean(grid_cell_ablations_norm[1, :, :], axis = 0), 'k-', label='Dual agent grid score')
# plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations_norm[1, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations_norm[1, :, :], axis = (0, 1)), color='k', alpha=0.5)
# plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), 'k--', label='Dual agent shuff.')
# plt.xlabel('Percentage of grid cells removed')
# plt.ylabel('% performance relative to original model')
# plt.legend()
# if save_flag:
#     plt.savefig(fig_path + 'grid_cell_ablations.svg', format = 'svg')

# plt.figure(figsize= (5, 4))
# plt.plot(ablate_percentiles, np.mean(border_cell_ablations_norm[0, :, :], axis = 0), 'g-', label='Single agent border score')
# plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)), color='g', alpha=0.5)
# plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)), 'g--', label='Single agent shuff.')
# plt.plot(ablate_percentiles, np.mean(border_cell_ablations_norm[1, :, :], axis = 0), 'k-', label='Dual agent border score')
# plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), color='k', alpha=0.5)
# plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), 'k--', label='Dual agent shuff.')
# plt.xlabel('Percentage of border cells removed')
# plt.ylabel('% performance relative to original model')
# plt.legend()
# if save_flag:
#     plt.savefig(fig_path + 'border_cell_ablations.svg', format = 'svg')

# plt.figure(figsize= (5, 4))
# plt.plot(ablate_percentiles, np.mean(band_cell_ablations_norm[0, :, :], axis = 0), 'g-', label='Single agent band score')
# plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)), color='g', alpha=0.5)
# plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations_norm[0, :, :, :], axis = (0, 1)), 'g--', label='Single agent shuff.')
# plt.plot(ablate_percentiles, np.mean(band_cell_ablations_norm[1, :, :], axis = 0), 'k-', label='Dual agent band score')
# plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)) - np.std(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)) + np.std(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), color='k', alpha=0.5)
# plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations_norm[1, :, :, :], axis = (0, 1)), 'k--', label='Dual agent shuff.')
# plt.xlabel('Percentage of band cells removed')
# plt.ylabel('% performance relative to original model')
# plt.legend()
# if save_flag:
#     plt.savefig(fig_path + 'band_cell_ablations.svg', format = 'svg')










