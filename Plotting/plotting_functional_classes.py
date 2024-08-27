#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:14:07 2024

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
from visualize import compute_ratemaps, compute_ratemaps_single_agent, compute_relative_ratemaps,compute_other_agent_ratemaps, plot_ratemaps
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

random.seed(2)

# Globals
n_seeds = 3
seed_plot = 2
n_g = 4096
save_flag = False
save_path = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/Figures/Plotting functional classes/All seeds/'
model_dir = '/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/models/'

# Computing grid, border, and band scores 
res = 20
n_avg = 100

starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
box_width=options.box_width
box_height=options.box_height
coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
masks_parameters = zip(starts, ends.tolist())
scorer = GridScorer(res, coord_range, masks_parameters)

activations_plot = np.zeros((2, n_seeds, n_g, res, res))
single_agent_activations_plot = np.zeros((n_seeds, n_g, res, res))
relative_activations_plot = np.zeros((n_seeds, n_g, res, res))
other_agent_activations_plot = np.zeros((n_seeds, n_g, res, res))
grid_scores = np.zeros((2, n_seeds, n_g))
relative_grid_scores = np.zeros((1, n_seeds, n_g))
grid_spacings = np.zeros((2, n_seeds, n_g))
grid_orientations = np.zeros((2, n_seeds, n_g))
border_scores = np.zeros((2, n_seeds, n_g))
band_scores = np.zeros((2, n_seeds, n_g))
band_lengths = np.zeros((2, n_seeds, n_g))

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

print('Computing single agent scores')
for ii in range(n_seeds):
    model = RNN(options, place_cells)
    model = model.to(options.device)
    model_folder = 'Single agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    model_name = 'final_model.pth'
    saved_model = torch.load(model_dir + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model)

    activations, rate_map, _, _ = compute_ratemaps(model, trajectory_generator, options, res=res, Ng=n_g)

    activations_plot[0, ii, :, :, :] = activations
        
    score_60, _, _, _, _, _ = zip(
        *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(rate_map)])
    score_60 = np.array(score_60)
    grid_scores[0, ii, :] = score_60
    
    spacing = []
    orientation = []
    counter = 0
    for rm in tqdm(rate_map):
        if score_60[counter] > 0:
            gs, go = scorer.get_spacing(rm.reshape(res, res))
        else:
            gs = np.nan
            go = np.nan
        spacing.append(gs)
        orientation.append(go)
        counter += 1
    grid_spacings[0, ii, :] = spacing
    grid_orientations[0, ii, :] = orientation

    score_border = []
    for rm in tqdm(rate_map):
        bs, _, _ = scorer.border_score(rm.reshape(res, res), res, box_width)
        score_border.append(bs)
    border_scores[0, ii, :] = score_border

    score_band = []
    length_band = []
    for rm in tqdm(rate_map):
        bs, bl = scorer.band_score(rm.reshape(res, res), res, box_width)
        score_band.append(bs)
        length_band.append(bl)
    band_scores[0, ii, :] = score_band
    band_lengths[0, ii, :] = length_band

from place_cells_dual_path_integration import PlaceCells
from trajectory_generator_dual_path_integration import TrajectoryGenerator
from model_dual_path_integration import RNN
from trainer_dual_path_integration import Trainer
place_cells = PlaceCells(options)
trajectory_generator = TrajectoryGenerator(options, place_cells)

print('Computing dual agent scores')
for ii in range(n_seeds):
    model = RNN(options, place_cells)
    model = model.to(options.device)
    model_folder = 'Dual agent path integration/Seed ' + str(ii) + ' weight decay 1e-06/'
    model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 
    model_name = 'final_model.pth'
    saved_model = torch.load(model_dir + model_folder + model_parameters + model_name, map_location=torch.device('cpu'))
    model.load_state_dict(saved_model)

    activations, rate_map, _, _ = compute_ratemaps(model, trajectory_generator, options, res=res, Ng=n_g)

    activations_plot[1, ii, :, :, :] = activations
        
    score_60, _, _, _, _, _ = zip(
        *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(rate_map)])
    score_60 = np.array(score_60)
    grid_scores[1, ii, :] = score_60

    spacing = []
    orientation = []
    counter = 0
    for rm in tqdm(rate_map):
        if score_60[counter] > 0:
            gs, go = scorer.get_spacing(rm.reshape(res, res))
        else:
            gs = np.nan
            go = np.nan
        spacing.append(gs)
        orientation.append(go)
        counter += 1
    grid_spacings[1, ii, :] = spacing
    grid_orientations[1, ii, :] = orientation

    score_border = []
    for rm in tqdm(rate_map):
        bs, _, _ = scorer.border_score(rm.reshape(res, res), res, box_width)
        score_border.append(bs)
    border_scores[1, ii, :] = score_border

    score_band = []
    length_band = []
    for rm in tqdm(rate_map):
        bs, bl = scorer.band_score(rm.reshape(res, res), res, box_width)
        score_band.append(bs)
        length_band.append(bl)
    band_scores[1, ii, :] = score_band
    band_lengths[1, ii, :] = length_band
    
    activations, _, _, _ = compute_ratemaps_single_agent(model, trajectory_generator, options, res=res, Ng=n_g)
    single_agent_activations_plot[ii, :, :, :] = activations
        
    activations, relative_rate_map, _, _, occupancy = compute_relative_ratemaps(model, trajectory_generator, options, res=res, Ng=n_g)
    relative_activations_plot[ii, :, :, :] = activations
    
    activations, other_agent_rate_map, _, _ = compute_other_agent_ratemaps(model, trajectory_generator, options, res=res, Ng=n_g)
    other_agent_activations_plot[ii, :, :, :] = activations

    score_60, _, _, _, _, _ = zip(
        *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(relative_rate_map)])
    score_60 = np.array(score_60)
    relative_grid_scores[0, ii, :] = score_60

# Saving scores
if save_flag:
    np.save(save_path + 'grid_scores_weight_decay_1e-06.npy', grid_scores)
    np.save(save_path + 'border_scores_weight_decay_1e-06.npy', border_scores)
    np.save(save_path + 'band_scores_weight_decay_1e-06.npy', band_scores)
    np.save(save_path + 'relative_grid_scores_weight_decay_1e-06.npy', relative_grid_scores)

# Plotting 
n_plot = 24

plt.figure(figsize=(5,4))
plt.hist(grid_scores[0, :, :].flatten(), range=(-0.5,1.5), bins=10, label='Single agent', alpha = 0.5, color = [0, 0.5, 0]);
plt.hist(grid_scores[1, :, :].flatten(), range=(-0.5,1.5), bins=10, label='Dual agent', alpha = 0.5, color = [0.8, 0.3, 0.2]);
plt.xlabel('Grid score')
plt.ylabel('Count');
plt.legend()
if save_flag:
    plt.savefig(save_path + 'Grid_score_distribution_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

plt.figure(figsize= (6, 4)) 
grid_ids = np.flip(np.argsort(grid_scores[0, seed_plot, :]))
rm_fig = plot_ratemaps(activations_plot[0, seed_plot, grid_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Single agent grid scores '+str(np.round(grid_scores[0, seed_plot, grid_ids[0]], 2))
             +' -- '+ str(np.round(grid_scores[0, seed_plot, grid_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Single_agent_grid_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

plt.figure(figsize= (6, 4)) 
grid_ids = np.flip(np.argsort(grid_scores[1, seed_plot, :]))
rm_fig = plot_ratemaps(activations_plot[1, seed_plot, grid_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent grid scores '+str(np.round(grid_scores[1, seed_plot, grid_ids[0]], 2))
             +' -- '+ str(np.round(grid_scores[1, seed_plot, grid_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_grid_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')
    
plt.figure(figsize= (6, 4)) 
grid_ids = np.flip(np.argsort(grid_scores[1, seed_plot, :]))
rm_fig = plot_ratemaps(other_agent_activations_plot[seed_plot, grid_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent o.a. grid scores '+str(np.round(grid_scores[1, seed_plot, grid_ids[0]], 2))
             +' -- '+ str(np.round(grid_scores[1, seed_plot, grid_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_other_agent_grid_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')
      
plt.figure(figsize= (6, 4)) 
rm_fig = plot_ratemaps(single_agent_activations_plot[seed_plot, grid_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent grid cells: single agent ratemap', fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_grid_cells_single_agent_ratemap_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')   
    
# plt.figure(figsize= (6, 4)) 
# rm_fig = plot_ratemaps(relative_activations_plot[seed_plot, grid_ids, :, :], n_plot, smooth=False, width=6)
# plt.imshow(rm_fig)
# plt.suptitle('Dual agent grid cells: relative ratemap', fontsize=16)
# plt.axis('off');
# if save_flag:
#     plt.savefig(save_path + 'Dual_agent_grid_cells_relative_ratemap_weight_decay_1e-04.svg', format = 'svg')        

plt.figure(figsize=(5,4))
plt.hist(border_scores[0, :, :].flatten(), range=(-1.0, 1.0), bins=10, label='Single agent', alpha = 0.5, color = [0, 0.5, 0]);
plt.hist(border_scores[1, :, :].flatten(), range=(-1.0, 1.0), bins=10, label='Dual agent', alpha = 0.5, color = [0.8, 0.3, 0.2]);
plt.xlabel('Border score')
plt.ylabel('Count');
plt.legend()
if save_flag:
    plt.savefig(save_path + 'Border_score_distribution_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

border_scores[np.isnan(border_scores)] = -100
plt.figure(figsize= (6, 4)) 
border_ids = np.flip(np.argsort(border_scores[0, seed_plot, :]))
rm_fig = plot_ratemaps(activations_plot[0, seed_plot, border_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Single agent border scores '+str(np.round(border_scores[0, seed_plot, border_ids[0]], 2))
             +' -- '+ str(np.round(border_scores[0, seed_plot, border_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Single_agent_border_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

plt.figure(figsize= (6, 4)) 
border_ids = np.flip(np.argsort(border_scores[1, seed_plot, :]))
rm_fig = plot_ratemaps(activations_plot[1, seed_plot, border_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent border scores '+str(np.round(border_scores[1, seed_plot, border_ids[0]], 2))
             +' -- '+ str(np.round(border_scores[1, seed_plot, border_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_border_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

plt.figure(figsize= (6, 4)) 
border_ids = np.flip(np.argsort(border_scores[1, seed_plot, :]))
rm_fig = plot_ratemaps(other_agent_activations_plot[seed_plot, border_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent o.a. border scores '+str(np.round(border_scores[1, seed_plot, border_ids[0]], 2))
             +' -- '+ str(np.round(border_scores[1, seed_plot, border_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_other_agent_border_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

plt.figure(figsize= (6, 4)) 
rm_fig = plot_ratemaps(single_agent_activations_plot[seed_plot, border_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent border cells: single agent ratemap', fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_border_cells_single_agent_ratemap_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')   
    
# plt.figure(figsize= (6, 4)) 
# rm_fig = plot_ratemaps(relative_activations_plot[seed_plot, border_ids, :, :], n_plot, smooth=False, width=6)
# plt.imshow(rm_fig)
# plt.suptitle('Dual agent border cells: relative ratemap', fontsize=16)
# plt.axis('off');
# if save_flag:
#     plt.savefig(save_path + 'Dual_agent_border_cells_relative_ratemap_weight_decay_1e-04.svg', format = 'svg')        

plt.figure(figsize=(5,4))
plt.hist(band_scores[0, :, :].flatten(), range=(0, 1.0), bins=10, label='Single agent', alpha = 0.5, color = [0, 0.5, 0]);
plt.hist(band_scores[1, :, :].flatten(), range=(0, 1.0), bins=10, label='Dual agent', alpha = 0.5, color = [0.8, 0.3, 0.2]);
plt.xlabel('Band score')
plt.ylabel('Count');
plt.legend()
if save_flag:
    plt.savefig(save_path + 'Band_score_distribution_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

band_scores[np.isnan(band_scores)] = -100

plt.figure(figsize= (6, 4)) 
band_ids = np.flip(np.argsort(band_scores[0, seed_plot, :]))
rm_fig = plot_ratemaps(activations_plot[0, seed_plot, band_ids, :, :], n_plot, smooth=False, width=6)
plt.imshow(rm_fig)
plt.suptitle('Single agent band scores '+str(np.round(band_scores[0, seed_plot, band_ids[0]], 2))
             +' -- '+ str(np.round(band_scores[0, seed_plot, band_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Single_agent_band_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')

plt.figure(figsize= (6, 4)) 
band_ids = np.flip(np.argsort(band_scores[1, seed_plot, :]))
rm_fig = plot_ratemaps(activations_plot[1, seed_plot, band_ids, :, :], n_plot, smooth=True, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent band scores '+str(np.round(band_scores[1, seed_plot, band_ids[0]], 2))
             +' -- '+ str(np.round(band_scores[1, seed_plot, band_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_band_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')
    
plt.figure(figsize= (6, 4)) 
band_ids = np.flip(np.argsort(band_scores[1, seed_plot, :]))
rm_fig = plot_ratemaps(other_agent_activations_plot[seed_plot, band_ids, :, :], n_plot, smooth=True, width=6)
plt.imshow(rm_fig)
plt.suptitle('Dual agent o.a. band scores '+str(np.round(band_scores[1, seed_plot, band_ids[0]], 2))
             +' -- '+ str(np.round(band_scores[1, seed_plot, band_ids[n_plot]], 2)),
            fontsize=16)
plt.axis('off');
if save_flag:
    plt.savefig(save_path + 'Dual_agent_other_agent_band_cells_weight_decay_1e-06_seed_' + str(seed_plot) + '.svg', format = 'svg')
    

# plt.figure(figsize= (6, 4)) 
# rm_fig = plot_ratemaps(single_agent_activations_plot[seed_plot, band_ids, :, :], n_plot, smooth=False, width=6)
# plt.imshow(rm_fig)
# plt.suptitle('Dual agent band cells: single agent ratemap', fontsize=16)
# plt.axis('off');
# if save_flag:
#     plt.savefig(save_path + 'Dual_agent_band_cells_single_agent_ratemap_weight_decay_1e-04.svg', format = 'svg')   
    
# plt.figure(figsize= (6, 4)) 
# rm_fig = plot_ratemaps(relative_activations_plot[seed_plot, band_ids, :, :], n_plot, smooth=False, width=6)
# plt.imshow(rm_fig)
# plt.suptitle('Dual agent band cells: relative ratemap', fontsize=16)
# plt.axis('off');
# if save_flag:
#     plt.savefig(save_path + 'Dual_agent_band_cells_relative_ratemap_weight_decay_1e-04.svg', format = 'svg')        

# plt.figure(figsize=(5,4))
# plt.hist(relative_grid_scores[0, :, :].flatten(), range=(-0.5,1.5), bins=10,  alpha = 0.5, color = [0, 0.5, 0]);
# plt.xlabel('Grid score')
# plt.ylabel('Count');
# plt.legend()
# if save_flag:
#     plt.savefig(save_path + 'Relative_space_grid_score_distribution_weight_decay_1e-04.svg', format = 'svg')

# plt.figure(figsize= (6, 4)) 
# relative_grid_ids = np.flip(np.argsort(relative_grid_scores[0, seed_plot, :]))
# rm_fig = plot_ratemaps(relative_activations_plot[seed_plot, relative_grid_ids, :, :], n_plot, smooth=False, width=6)
# plt.imshow(rm_fig)
# plt.suptitle('Dual agent relative grid scores '+str(np.round(relative_grid_scores[0, seed_plot, relative_grid_ids[0]], 2))
#              +' -- '+ str(np.round(relative_grid_scores[0, seed_plot, relative_grid_ids[n_plot]], 2)),
#             fontsize=16)
# plt.axis('off');
# if save_flag:
#     plt.savefig(save_path + 'Relative_space_grid_cells_weight_decay_1e-04.svg', format = 'svg')





