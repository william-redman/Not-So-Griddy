import numpy as np
import tensorflow as tf
import torch.cuda
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
from visualize import compute_ratemaps, plot_ratemaps
import os
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    # default='/mnt/fs2/bsorsch/grid_cells/models/',
                    default='models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=100, 
                    help='number of training epochs') #100,
parser.add_argument('--n_steps',
                    default= 1000, 
                    help='batches per epoch') #1000,
parser.add_argument('--batch_size',
                    default=200,
                    help='number of trajectories per batch') #200
parser.add_argument('--sequence_length',
                    default=20,
                    help='number of steps in trajectory') #20
parser.add_argument('--learning_rate',
                    default= 1e-4, 
                    help='gradient descent learning rate') #1e-4
parser.add_argument('--Np',
                    default= 512, 
                    help='number of place cells') #512
parser.add_argument('--Ng',
                    default= 4096,                                                                                                                                                                                                                                                             
                    help='number of grid cells') #4096
parser.add_argument('--place_cell_rf',
                    default=0.12,
                    help='width of place cell center tuning curve (m)') #0.12
parser.add_argument('--surround_scale',
                    default=2,
                    help='if DoG, ratio of sigma2^2 to sigma1^2') #2
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
                    default=2.2, #
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


place_cells = PlaceCells(options)
#if options.RNN_type =='RNN':
model = RNN(options, place_cells) #.cpu()
#elif options.RNN_type == 'LSTM':
    # model = LSTM(options, place_cells)
    #raise NotImplementedError

# Put model on GPU if using GPU
model = model.to(options.device)

#dual_agent_model = torch.load('/Users/redmawt1/Documents/Internal Representations of Space in Multi-Agent Environments/grid-pattern-formation/models/Dual agent path integration disjoint PCs/Seed 1 weight decay 1e-06/steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/final_model.pth')
#dual_agent_model['RNN.weight_ih_l0'] = dual_agent_model['RNN.weight_ih_l0'][:, :2]
#model.load_state_dict(dual_agent_model)

#recurrent_freeze_flag = True # set to true to freeze recurrent weights
#if recurrent_freeze_flag:
#    counter = 0 
#    for param in model.parameters():
#        if counter == 2:
#            param.requires_grad = False # Freezing  only the recurrent weights
#        counter = counter + 1

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

# Train
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)

# Saving model
torch.save(model.state_dict(), os.path.join(options.save_dir, options.run_ID) + '/final_model.pth')

# Plot training metrics
plt.figure(figsize=(12,3))
plt.subplot(121)
plt.plot(trainer.err, c='black')
np.save(os.path.join(options.save_dir, options.run_ID) + '/decoding_error.npy', trainer.err)

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
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-options.box_width/2,options.box_width/2])
    plt.ylim([-options.box_height/2,options.box_height/2])
    if i==0:
        plt.legend()

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

# Compute border scores
border_scores = []
for rm in tqdm(rate_map):
    bs, _, _ = scorer.border_score(rm.reshape(res, res), res, box_width)
    border_scores.append(bs)

# Plot high border scores
border_idxs = np.flip(np.argsort(border_scores))
n_plot = 128
plt.figure(figsize=(16,4*n_plot//8**2))
rm_fig = plot_ratemaps(activations[border_idxs], n_plot, smooth=True)
plt.imshow(rm_fig)
plt.suptitle('Border scores '+str(np.round(border_scores[border_idxs[0]], 2))
             +' -- '+ str(np.round(border_scores[border_idxs[n_plot]], 2)),
            fontsize=16)
plt.axis('off');

# MANIFOLD DISTANCE
# Keeping all grid cells
origins = np.stack(np.mgrid[:3,:3] - 1) * res//4 + res//2

fig = plt.figure(figsize=(8,8))
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,3*i+j+1)
        origin=np.random.randint(0,100,2)
        origin_idx = np.ravel_multi_index((origins[0,i,j],origins[1,i,j]), (res,res))
        r0 = rate_map[:,origin_idx,None]
        dists = np.linalg.norm(r0 - rate_map, axis=0)
        im = plt.imshow(dists.reshape(res,res)/np.max(dists),
                        cmap='viridis_r', interpolation='gaussian')
        plt.axis('off')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.12, 0.02, 0.74])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.locator_params(nbins=3)
cbar.ax.tick_params(labelsize=20) 
cbar.outline.set_visible(False)

# Keeping only top 500 grid cells
n_grid_cells = 500
grid_sort = np.flip(np.argsort(score_60))
origins = np.stack(np.mgrid[:3,:3] - 1) * res//4 + res//2

fig = plt.figure(figsize=(8,8))
for i in range(3):
    for j in range(3):
        plt.subplot(3,3,3*i+j+1)
        origin=np.random.randint(0,100,2)
        origin_idx = np.ravel_multi_index((origins[0,i,j],origins[1,i,j]), (res,res))
        r0 = rate_map[grid_sort[:n_grid_cells],origin_idx,None]
        dists = np.linalg.norm(r0 - rate_map[grid_sort[:n_grid_cells]], axis=0)
        im = plt.imshow(dists.reshape(res,res)/np.max(dists),
                        cmap='viridis_r', interpolation='gaussian')
        plt.axis('off')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.12, 0.02, 0.74])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.locator_params(nbins=3)
cbar.ax.tick_params(labelsize=20) 
cbar.outline.set_visible(False)

## Neural sheet
# Fourier transform 
Ng = options.Ng
rm_fft_real = np.zeros([Ng,res,res])
rm_fft_imag = np.zeros([Ng,res,res])

for i in tqdm(range(Ng)):
    rm_fft_real[i] = np.real(np.fft.fft2(rate_map[i].reshape([res,res])))
    rm_fft_imag[i] = np.imag(np.fft.fft2(rate_map[i].reshape([res,res])))
    
rm_fft = rm_fft_real + 1j * rm_fft_imag
fig = plt.figure(figsize=(4,4))
width = 6
idxs = np.arange(-width+1, width)
x2, y2 = np.meshgrid(np.arange(2*width-1), np.arange(2*width-1))
im = (np.real(rm_fft)**2).mean(0)
im[0,0] = 0
plt.scatter(x2,y2,c=im[idxs][:,idxs], s=300, cmap='Oranges')
plt.axis('equal')
plt.axis('off');
plt.title('Mean power');

k1 = [3,0]
k2 = [2,3]
k3 = [-1,3]
k4=k5=k6=k1

freq = 1
ks = freq*np.array([k1,k2,k3,k4,k5,k6])
ks = ks.astype('int')

modes = np.stack([rm_fft[:,k[0],k[1]] for k in ks])

# Find phases
phases = [np.angle(mode) for mode in modes]

plt.figure(figsize=(15,5))
plt.subplot(131)
plt.scatter(phases[0], phases[1], c='black', s=10)
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.subplot(132)
plt.scatter(phases[1], phases[2], c='black', s=10)
plt.xlabel(r'$\phi_2$')
plt.ylabel(r'$\phi_3$')
plt.subplot(133)
plt.scatter(phases[0], phases[2], c='black', s=10)
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_3$')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(phases[0], phases[1], phases[2], c='black', s=2)
ax.view_init(azim=60)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.set_xlabel(r'$\phi_1$')
ax.set_ylabel(r'$\phi_2$')
ax.set_zlabel(r'$\phi_3$')

freq = 1
crop = 0
cmaps = ['Blues', 'Oranges', 'Greens']
x = np.mgrid[:res,:res] * 2*np.pi/ res
x = x.reshape(2, -1)
k = freq*np.stack([k1,k2,k3])
X = np.concatenate([np.cos(k.dot(x)), np.sin(k.dot(x))], axis=0)
idxs1, idxs2 = np.mgrid[crop:res-crop, crop:res-crop]
idxs = np.ravel_multi_index((idxs1,idxs2), (res,res)).ravel()

plt.figure(figsize=(12,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    B = np.stack([np.cos(phases[i]), np.sin(phases[i])])
    test = B@rate_map
    plt.scatter(test[0], test[1], c=X[i][idxs], cmap=cmaps[i], s=20)
    plt.axis('off')
    
from utils import get_2d_sort
import scipy

N = rate_map.shape[0]
n = int(np.sqrt(N))
width = int(np.sqrt(N))
freq = 1
X,Y = np.meshgrid(np.arange(width),np.arange(width))
X = X*2*np.pi/width
Y = Y*2*np.pi/width

s1 = np.zeros(phases[0].shape)
s2 = np.zeros(phases[0].shape)

fac = np.sqrt(3)/2

for i in range(Ng):
    penalty_1 = np.cos(freq*X - phases[0][i]/fac)
    penalty_2 = np.cos(freq*Y - phases[2][i]/fac)
    penalty_3 = np.cos(freq*(X+Y) - phases[1][i]/fac)
    ind = np.argmax(penalty_1+penalty_2 + penalty_3  + np.random.randn()/100)
    s1[i],s2[i] = np.unravel_index([ind],penalty_1.shape)
    
total_order = get_2d_sort(s1,s2)
rm_sort_square = rate_map[total_order.ravel()].reshape([n,n,-1])

# Skew matrix to transform parallelogram unit cells to squares
A = np.asarray([[2,1],[0, np.sqrt(3)]])/4
Ainv = np.linalg.inv(A)

freq = 2
nplots=10
fig, axes = plt.subplots(nplots,nplots, figsize=(16,16))
for i in range(nplots):
    for j in range(nplots):
        idx = np.ravel_multi_index(((i+nplots//2)*res//nplots//freq,
                                    (j+nplots//2)*res//nplots//freq), (res,res))
        im = rm_sort_square[:,:,idx]
        im = scipy.ndimage.affine_transform(im, Ainv, mode='wrap')
        im = scipy.ndimage.gaussian_filter(im, sigma=(2,2))
        axes[j,i].imshow(im.T, cmap='jet')
        axes[j,i].axis('off')

J = model.RNN._parameters['weight_hh_l0'].detach().numpy().T
plt.figure(figsize=(6,6))
plt.imshow(J, cmap='RdBu');
plt.title('J');

# Eigenvalues
eigs, eigvs = np.linalg.eig(J)

fig, ax = plt.subplots()
plt.scatter(np.real(eigs), np.imag(eigs), c='black', s=20)
plt.scatter(np.real(eigs[:9]), np.imag(eigs[:9]), c='C1', s=20)
circle1 = plt.Circle((0, 0), 1, color='tan',
                     fill=False, linestyle='dashed', linewidth=2)
ax.add_artist(circle1)
plt.xlim([-1.1,2.5])
plt.ylim([-1.1,1.1])
plt.gca().set_aspect('equal', adjustable='box')
plt.locator_params(nbins=4)

U,S,V = np.linalg.svd(J)

plt.figure(figsize=(8,4))
plt.plot(S[:100], 'o-', c='black')
plt.plot(S[:9], 'o-', c='C1')

A = np.asarray([[2,1],[0, np.sqrt(3)]])/2
Ainv = np.linalg.inv(A)

plt.figure(figsize=(12,8))
idxs = [1,3,5,4,0,2]
for i in range(6):
    plt.subplot(2,3,i+1)
    im = eigvs[idxs[i]].reshape(n,n)
    im = np.real(im)
    im = np.roll(np.roll(im, n//4, axis=1), -n//4, axis=0)
    im = scipy.ndimage.affine_transform(im, Ainv, mode='wrap')
    if i==1 or i==4:
        im = np.roll(im, -n//3, axis=0)
    im = scipy.ndimage.gaussian_filter(im, sigma=(2,2))
    
    plt.imshow(im, cmap='coolwarm')
    plt.axis('off')
plt.tight_layout()

idxs = np.flip(np.argsort(eigs))

A = np.asarray([[2,1],[0, np.sqrt(3)]])/2
Ainv = np.linalg.inv(A)

n = int(np.sqrt(Ng))
plt.figure(figsize=(16,4))
for i in range(24):
    plt.subplot(2,12,i+1)
    idx = idxs[i]
    im = np.real(eigvs[total_order,idx].reshape(n,n))
    im = scipy.ndimage.affine_transform(im, Ainv, mode='wrap')
    im = scipy.ndimage.gaussian_filter(im, sigma=(2,2)) 
    plt.imshow(im, cmap='coolwarm', interpolation='gaussian')
    plt.axis('off')
    
n = int(np.sqrt(Ng))
Jsort = J[total_order][:, total_order]
J_square = np.reshape(Jsort, (n,n,n,n))

Jmean = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        Jmean += np.roll(np.roll(J_square[i,j], -i, axis=0), -j, axis=1)
        
Jmean[0,0] = np.max(Jmean[1:,1:])
Jmean = np.roll(np.roll(Jmean, n//2, axis=0), n//2, axis=1)
# plt.imshow(Jmean, cmap='coolwarm', interpolation='gaussian')

A = np.asarray([[2,1],[0, np.sqrt(3)]])/2
Ainv = np.linalg.inv(A)
im = Jmean
im = scipy.ndimage.affine_transform(Jmean, Ainv, mode='wrap')

imroll = im
imroll = np.roll(np.roll(im, -n//4, axis=0), 0, axis=1)
# imroll = scipy.ndimage.gaussian_filter(imroll, sigma=(1,1))

plt.figure(figsize=(5,5))
plt.imshow(imroll, cmap='coolwarm')
plt.title('J (sorted)')
plt.axis('off');    

n = int(np.sqrt(N))
width = 18
xs = np.arange(2*width-1)
X,Y = np.meshgrid(xs,xs)
XY = np.stack((X.ravel(),Y.ravel()),0)
T = np.array([[1,0.5],[0,np.sqrt(3)/2]])
XY = T.dot(XY)#+ np.random.randn(*XY.shape)/100 

idxs = np.arange(-width+1, width)
im_fft = np.abs(np.fft.fft2(im))
im_fft[0,0] = 0

plt.figure(figsize=(10,10))
plt.scatter(XY[0],XY[1],s=120,c=im_fft[idxs][:,idxs].ravel(),
            marker='h', cmap='viridis')
plt.axis('equal')
plt.axis('off');





































