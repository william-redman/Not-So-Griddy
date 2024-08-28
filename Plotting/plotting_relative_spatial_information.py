#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:14:55 2024

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
from visualize import compute_ratemaps, compute_ratemaps_single_agent, compute_relative_ratemaps, plot_ratemaps
from tqdm import tqdm
from scores import GridScorer
from place_cells_dual_path_integration import PlaceCells
from trajectory_generator_dual_path_integration import TrajectoryGenerator
from model_dual_path_integration import RNN
from trainer_dual_path_integration import Trainer


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

random.seed(2)

# Globals
n_seeds = 5
seed_plot = 2
n_g = 4096
save_flag = False
fig_path = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/Figures/'
model_path = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/models/'
result_path = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/Results/'

# Computing spatial information (in relative space)
res = 20
n_avg = 100

place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

spatial_info = np.zeros([n_seeds, n_g])
activations_plot = np.zeros([n_seeds, n_g, res, res])

print('Relative space spatial information')
for ii in range(n_seeds):
    model = RNN(options, place_cells)
    model = model.to(options.device)
    model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    model_name = 'final_model.pth'
    saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model)

    activations, _, _, _, occupancy = compute_relative_ratemaps(model, trajectory_generator, options, res=res, Ng=n_g)
    activations_plot[ii, :, :, :] = activations

    for jj in range(n_g):
        a = activations[jj, :, :]
        mean_a = np.nanmean(a)
        p = occupancy / np.sum(occupancy)
        spatial_info[ii, jj] = np.nansum(p.flatten() * a.flatten() * np.log2(a.flatten() / mean_a)) / mean_a
        
spatial_info[np.isnan(spatial_info)] = -100 # setting any cells with nan to -100 so they will not be in the top ranking cells
        
# Plotting units with highest spatial info (in relative space)
n_plot = 24

plt.figure(figsize=(5,4))
plt.hist(spatial_info.flatten(), range=(-0.5,3), bins=10, alpha = 0.5, color = [0, 0.5, 0]);
plt.xlabel('Relative space spatial information')
plt.ylabel('Count');
plt.legend()
if save_flag:
    plt.savefig(fig_path + 'Plotting functional classes/Relative_space_spatial_information_distribution.svg', format = 'svg')


plt.figure(figsize= (6, 4)) 
spatial_info_ids = np.flip(np.argsort(spatial_info[seed_plot, :]))
rm_fig = plot_ratemaps(activations_plot[seed_plot, spatial_info_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Relative space spatial info. '+str(np.round(spatial_info[seed_plot, spatial_info_ids[0]], 2))
             +' -- '+ str(np.round(spatial_info[seed_plot, spatial_info_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(fig_path + 'Plotting functional classes/Relative_space_units_highest_spatial_info.svg', format = 'svg')

# Ablations
n_shuffle = 10
ablate_percentiles = [0, 5, 10, 15, 20, 25]
spatial_info_cell_ablations =  np.zeros([n_seeds, len(ablate_percentiles)])
random_shuffle_ablations = np.zeros([n_seeds, n_shuffle, len(ablate_percentiles)])

for ii in range(n_seeds):
    print('Ablating high spatial info cells')
    for jj in range(len(ablate_percentiles)):
        model = RNN(options, place_cells)
        model = model.to(options.device)
        model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
        model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
        model_name = 'final_model.pth'
        saved_model = torch.load(model_path + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
        
        ablate_ids = np.argwhere(spatial_info[ii, :] > np.nanpercentile(spatial_info[ii, :], 100 - ablate_percentiles[jj]))    
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
        spatial_info_cell_ablations[ii, jj] = np.mean(e)

random_shuffle_ablations = np.load(result_path + 'Ablations/random_shuffle_ablations.npy')
random_shuffle_ablations = random_shuffle_ablations[1, :, :, :]


# Saving ablation matrices
if save_flag: 
    np.save(result_path + 'Ablations/spatial_info_cell_ablations.npy', spatial_info_cell_ablations)

# Plotting ablation results
# spatial_info_cell_ablations_norm = np.copy(spatial_info_cell_ablations)
# random_shuffle_ablations_norm = np.copy(random_shuffle_ablations)
# spatial_info_cell_ablations_norm = np.reshape(spatial_info_cell_ablations[:, 0], (n_seeds, 1)) / spatial_info_cell_ablations 
# random_shuffle_ablations_norm = np.reshape(random_shuffle_ablations[:, 0], (n_seeds, n_shuffle, 1)) / random_shuffle_ablations 

plt.figure(figsize= (5, 4))
plt.plot(ablate_percentiles, np.mean(spatial_info_cell_ablations, axis = 0), 'k-', label='Dual agent relative space spatial info')
plt.fill_between(ablate_percentiles, np.mean(random_shuffle_ablations, axis = (0, 1)) - np.std(random_shuffle_ablations, axis = (0, 1)), np.mean(random_shuffle_ablations, axis = (0, 1)) + np.std(random_shuffle_ablations, axis = (0, 1)), color='k', alpha=0.5)
plt.plot(ablate_percentiles, np.mean(random_shuffle_ablations, axis = (0, 1)), 'k--', label='Dual agent shuff.')
plt.xlabel('Percentage of relative space spatial info cells removed')
plt.ylabel('Decoding error (m.)')
plt.legend()
if save_flag:
    plt.savefig(fig_path + 'Plotting ablations/relative_space_spatial_info_ablations.svg', format = 'svg')

