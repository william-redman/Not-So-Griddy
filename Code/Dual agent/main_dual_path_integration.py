import numpy as np
import tensorflow as tf
import torch.cuda
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import generate_run_ID
from place_cells_dual_path_integration import PlaceCells
from trajectory_generator_dual_path_integration import TrajectoryGenerator
from model_dual_path_integration import RNN
from trainer_dual_path_integration import Trainer
from visualize import compute_ratemaps, plot_ratemaps
import random
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    # default='/mnt/fs2/bsorsch/grid_cells/models/',
                    default='models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=100,
                    help='number of training epochs') #100
parser.add_argument('--n_steps',
                    default= 1000,
                    help='batches per epoch') #1000
parser.add_argument('--batch_size',
                    default=200,
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

place_cells = PlaceCells(options)
if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
elif options.RNN_type == 'LSTM':
    # model = LSTM(options, place_cells)
    raise NotImplementedError

# Put model on GPU if using GPU
model = model.to(options.device)

# Loading the single agengt model weights
#single_agent_model = torch.load('/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/models/Single agent path integration/Seed 1 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth')
#single_agent_model['RNN.weight_ih_l0'] = torch.cat((single_agent_model['RNN.weight_ih_l0'], single_agent_model['RNN.weight_ih_l0']), 1)
#model.load_state_dict(single_agent_model)

#recurrent_freeze_flag = True # set to true to freeze recurrent weights
#if recurrent_freeze_flag:
#   counter = 0 
#    for param in model.parameters():
#        if counter == 2:
#            param.requires_grad = False # Freezing only recurrent weights
#        counter = counter + 1

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

# Train
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)

# Saving model
torch.save(model.state_dict(), os.path.join(options.save_dir, options.run_ID) + '/final_model.pth')

# Plot training loss and decoding error
plt.figure(figsize=(12,3))
plt.subplot(121)
err = [err for err in trainer.err if err > 0]
np.save(os.path.join(options.save_dir, options.run_ID) + '/decoding_error.npy', err)
plt.plot(err, c='black')
plt.title('Decoding error (m)'); plt.xlabel('train step')

plt.subplot(122)
plt.plot(trainer.loss, c='black');
plt.title('Loss'); plt.xlabel('train step');
np.save(os.path.join(options.save_dir, options.run_ID) + '/loss.npy', trainer.loss)

# Plot paths
inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
pos = pos.cpu()
pred_pos = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
us = place_cells.us.cpu()

plt.figure(figsize=(5,5))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(pos[:,i,0], pos[:,i,1], 'k.', label='True position')
    plt.plot(pred_pos[:,i,0], pred_pos[:,i,1], 'r.', label='Decoded position')
    plt.plot(pos[:,i,2], pos[:,i,3], 'k.')
    plt.plot(pred_pos[:,i,2], pred_pos[:,i,3], 'r.')
    if i==0:
        plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlim([-options.box_width/2,options.box_width/2])
plt.ylim([-options.box_height/2,options.box_height/2]);

# Plot ratemaps
from tqdm import tqdm
from visualize import compute_ratemaps, plot_ratemaps
res = 20
n_avg = 100
Ng = options.Ng
activations, rate_map, g, pos = compute_ratemaps(model,
                                                 trajectory_generator,
                                                 options,
                                                 res=res,
                                                 #n_avg=n_avg,
                                                 Ng=Ng)


n_plot = 512
plt.figure(figsize=(16,4*n_plot//8**2))
rm_fig = plot_ratemaps(activations, n_plot, smooth=True)
plt.imshow(rm_fig)
plt.axis('off');

# Compute grid scores
from scores import GridScorer
starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
box_width=options.box_width
box_height=options.box_height
coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
masks_parameters = zip(starts, ends.tolist())
scorer = GridScorer(res, coord_range, masks_parameters)

score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
      *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(rate_map)])

plt.figure(figsize=(5,5))
plt.hist(score_60, range=(-1,2.5), bins=15);
plt.xlabel('Grid score')
plt.ylabel('Count');

idxs = np.flip(np.argsort(score_60))
Ng = options.Ng

# Plot high grid scores
n_plot = 128
plt.figure(figsize=(16,4*n_plot//8**2))
rm_fig = plot_ratemaps(activations[idxs], n_plot, smooth=True)
plt.imshow(rm_fig)
plt.suptitle('Grid scores '+str(np.round(score_60[idxs[0]], 2))
             +' -- '+ str(np.round(score_60[idxs[n_plot]], 2)),
            fontsize=16)
plt.axis('off');

# Plot medium grid scores
plt.figure(figsize=(16,4*n_plot//8**2))
rm_fig = plot_ratemaps(activations[idxs[Ng//4:]], n_plot, smooth=True)
plt.imshow(rm_fig)
plt.suptitle('Grid scores '+str(np.round(score_60[idxs[Ng//2]], 2))
             +' -- ' + str(np.round(score_60[idxs[Ng//2+n_plot]], 2)),
            fontsize=16)
plt.axis('off');

# Plot low grid scores
plt.figure(figsize=(16,4*n_plot//8**2))
rm_fig = plot_ratemaps(activations[np.flip(idxs)], n_plot, smooth=True)
plt.imshow(rm_fig)
plt.suptitle('Grid scores '+str(np.round(score_60[idxs[-n_plot]], 2))
             +' -- ' + str(np.round(score_60[idxs[-1]], 2)),
            fontsize=16)
plt.axis('off');

































