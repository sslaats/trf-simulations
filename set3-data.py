#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:25:43 2022

@author: sopsla
"""
import os
import time
import sys
import pickle

# math
import numpy as np
import scipy

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# trf stuff
import simfun as sf
from pyeeg.utils import lag_span
import mne

# %% general settings
arrayID = int(sys.argv[1])

sfreq = 120 # Hz
isi = 1.0 # 1000 milliseconds between impulses
response_no = 1000 # number of responses
tmin = -0.2
tmax = 1.0
lags = lag_span(tmin,tmax,sfreq)

# %% adaptable kernel
def moveable_kernel(value, kernel, multiplication=1):
    n_zeros = int(np.round(value * multiplication))
    return np.concatenate([np.zeros((n_zeros)), kernel])

# %% setting up the kernel and predictors
# predictor & data timing # always the same
isi_samples = isi * sfreq * np.ones((1, response_no))
tmp_samp = np.asarray(np.cumsum(isi_samples), int)
mult_values = np.linspace(0, 20, num=20)/10 
n_iterations = 10

# %% PART 1: 500MS NO NOISE
noise = False
kernel_width = 0.5 # 500 milliseconds of response
kernel = sf.impulse_response(sfreq, kernel_width)
resultsdir = '/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set3-figures/log-500ms'

systematic = np.zeros((n_iterations, 8, len(mult_values)))
systematic_betas = []

other_variable = systematic.copy()
other_variable_betas = systematic_betas.copy()

# %% we do this n_iterations of times
start = time.time()

for iteration in list(range(0,n_iterations)):

    # get values for the split predictor
    brackets = 2 * np.random.randn(1, len(tmp_samp)) + 5
    brackets = np.abs(brackets)
    surprisal = np.random.lognormal(mean=2.3, sigma=0.41, size=(1,len(tmp_samp))) # np.random.lognormal(mean=0.5, sigma=1.2, size=(1,len(tmp_samp)))# 4 * np.random.randn(1, len(tmp_samp)) + 17
    #surprisal = np.abs(surprisal) # in 1.02 % of cases, lowest value is below zero, triggers error
    
    some_other_variable = np.random.lognormal(mean=2.3, sigma=0.41,size=(1,len(tmp_samp))) #4 * np.random.randn(1, len(tmp_samp)) + 17 # different var, same distribution
    #some_other_variable = np.abs(some_other_variable)
        
    data_duration = int(max(tmp_samp)+sfreq+max(surprisal[0,:])*max(mult_values))
    
    predictor1 = np.zeros((data_duration))
    predictor2 = np.zeros((data_duration))

    # data - with linear moveable kernel, adjusting the influence of surprisal   
    for var,varname in zip([surprisal, some_other_variable], 
                                 ['systematic', 'random']):
        simulated_rvals = np.zeros((8,len(mult_values)))
        betatmp = {}
        
        for idx,mult in enumerate(mult_values):
            betalist = []
            
            # surprisal kernels
            # in the case of some_other_variable we will use this to generate the data
            # but then never use it again
            surprisal_kernels = [moveable_kernel(surp, kernel,multiplication=mult) for surp in var[0,:]]
            data_list = []
            
            for i,(br,sk) in enumerate(zip(brackets[0,:], surprisal_kernels)):
                response = scipy.signal.convolve(np.expand_dims(br,0), sk)
                data_list.append(response)
        
            # Just this once: plot the data
            #lens = [len(d) for d in data_list]
            #data_list_long = [np.pad(d, (0, max(lens)-len(d)), 'constant', constant_values=(0)) for d in data_list]
            #data_frame = pd.DataFrame(data_list_long, index=np.squeeze(var))
            #data_frame = data_frame.sort_index()
            #fig,ax=plt.subplots()
            #sns.heatmap(data_frame, ax=ax, cmap='viridis')
            #ax.set_title(f'{varname}, {mult}x')
            
            # create data
            data_empty = np.zeros((data_duration+2000))
            
            for index,response in zip(tmp_samp, data_list):
                data_empty[index:index+len(response)] += response
                
            data = data_empty.copy()
            
            if noise: # false
                ns = (sf.spectrum_noise(sf.pink_spectrum, samples=len(data), rate=sfreq) + \
                    np.squeeze(mne.filter.filter_data(np.random.randn(1, len(data)), sfreq=sfreq, \
                                                         l_freq=7, h_freq=12, fir_design='firwin', verbose=False))) * (np.std(data)/10**(0.1/10))
                data += ns
                
            # create the predictors
            predictor_brackets = np.zeros((len(data)))
            predictor_surprisal = np.zeros((len(data)))
            
            predictor_brackets[tmp_samp] = brackets
            predictor_surprisal[tmp_samp] = surprisal
            
            train_predictor_brackets = predictor_brackets[0:int(4*(len(data)/5)),]
            test_predictor_brackets = predictor_brackets[int(4*(len(data)/5)):,]
            
            ######## one predictor TRF ('node')
            train_data = data.copy()[0:int(4*(len(data)/5))]
            test_data = data.copy()[int(4*(len(data)/5)):]
            
            betas = sf.simulate_TRF(train_data, train_predictor_brackets, lags, alpha=20, fit_intercept=False)
            rvals = sf.simulate_prediction(betas, test_data, test_predictor_brackets, lags)
                    
            simulated_rvals[0,idx] = rvals
            betalist.append(betas)
        
            ######## both predictors ('node + surprisal')
            predictor_both = np.vstack([predictor_brackets, predictor_surprisal]).T
            train_predictor_both = predictor_both[0:int(4*(len(data)/5)),]
            test_predictor_both = predictor_both[int(4*(len(data)/5)):,]
            
            betas_both = sf.simulate_TRF(train_data, train_predictor_both, lags, alpha=20, fit_intercept=False)
            rvals_both = sf.simulate_prediction(betas_both, test_data, test_predictor_both, lags)
            
            simulated_rvals[1,idx] = rvals_both
            betalist.append(betas_both)
        
            ######## split brackets by surprisal
            predictor_high = np.zeros((len(data)))
            predictor_low = np.zeros((len(data)))
            
            low_idx = np.where(surprisal < np.median(surprisal))
            high_idx = np.where(surprisal > np.median(surprisal))
            
            predictor_high[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split = np.vstack([predictor_high, predictor_low]).T
            
            train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
            test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
            
            betas_split = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=20, fit_intercept=False)
            rvals_split = sf.simulate_prediction(betas_split, test_data, test_predictor_split, lags)
            
            simulated_rvals[2,idx] = rvals_split
            betalist.append(betas_split)
        
            ######## split brackets by surprisal + surprisal predictor
            predictor_split_surprisal = np.vstack([predictor_high, predictor_low, predictor_surprisal]).T
            
            train_predictor_split_surprisal = predictor_split_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_surprisal = predictor_split_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_surprisal = sf.simulate_TRF(train_data, train_predictor_split_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_surprisal = sf.simulate_prediction(betas_split_surprisal, test_data, test_predictor_split_surprisal, lags)
            
            simulated_rvals[3,idx] = rvals_split_surprisal
            betalist.append(betas_split_surprisal)
        
            ######## random split predictor
            predictor_high_random = np.zeros((len(data)))
            predictor_low_random = np.zeros((len(data)))
            
            # get indices for random split
            # we generate another random 1000 samples
            random_for_split = np.random.randn(1, len(tmp_samp))
            
            low_idx = np.where(random_for_split < np.median(random_for_split))
            high_idx = np.where(random_for_split > np.median(random_for_split))
                                    
            predictor_high_random[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low_random[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split_random = np.vstack([predictor_high_random, predictor_low_random]).T
            
            train_predictor_split_random = predictor_split_random[0:int(4*(len(data)/5)),]
            test_predictor_split_random = predictor_split_random[int(4*(len(data)/5)):,]
            
            betas_split_random = sf.simulate_TRF(train_data, train_predictor_split_random, lags, alpha=20, fit_intercept=False)
            rvals_split_random = sf.simulate_prediction(betas_split_random, test_data, test_predictor_split_random, lags)
            
            simulated_rvals[4,idx] = rvals_split_random
            betalist.append(betas_split_random)
        
            ######## random split predictor + surprisal
            predictor_split_random_surprisal = np.vstack([predictor_high_random, predictor_low_random, predictor_surprisal]).T
            
            train_predictor_split_random_surprisal = predictor_split_random_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_random_surprisal = predictor_split_random_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_random_surprisal = sf.simulate_TRF(train_data, train_predictor_split_random_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_random_surprisal = sf.simulate_prediction(betas_split_random_surprisal, test_data, test_predictor_split_random_surprisal, lags)
               
            simulated_rvals[5,idx] = rvals_split_random_surprisal
            betalist.append(betas_split_random_surprisal)
        
            ######## interaction term
            interaction = brackets * surprisal
            predictor_interaction = np.zeros((len(data)))
            predictor_interaction[tmp_samp] = interaction
            
            train_predictor_interaction = predictor_interaction[0:int(4*(len(data)/5)),]
            test_predictor_interaction = predictor_interaction[int(4*(len(data)/5)):,]
            
            betas_interaction = sf.simulate_TRF(train_data, train_predictor_interaction, lags, alpha=20, fit_intercept=False)
            rvals_interaction = sf.simulate_prediction(betas_interaction, test_data, test_predictor_interaction, lags)
           
            simulated_rvals[6,idx] = rvals_interaction
            betalist.append(betas_interaction)
        
            ######## add both predictors + interaction
            predictor_both_interaction = np.vstack([predictor_brackets, predictor_surprisal, predictor_interaction]).T
            train_predictor_both_interaction = predictor_both_interaction[0:int(4*(len(data)/5)),]
            test_predictor_both_interaction = predictor_both_interaction[int(4*(len(data)/5)):,]
            
            betas_both_interaction = sf.simulate_TRF(train_data, train_predictor_both_interaction, lags, alpha=20, fit_intercept=False)
            rvals_both_interaction = sf.simulate_prediction(betas_both_interaction, test_data, test_predictor_both_interaction, lags)
            
            simulated_rvals[7,idx] = rvals_both_interaction
            betalist.append(betas_both_interaction)
            
            betatmp[idx] = betalist
        
        if varname == 'systematic':
            systematic[iteration,:,:] = simulated_rvals
            systematic_betas.append(betatmp)
            
        elif varname == 'random':
            other_variable[iteration,:,:] = simulated_rvals
            other_variable_betas.append(betatmp)
    
stop = time.time()
print(f'Duration: {stop-start} s')

# %% store these results
with open(os.path.join(resultsdir, f'systematic_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(systematic, f)
    
with open(os.path.join(resultsdir, f'systematic_betas_{n_iterations}it{arrayID}.pkl'),'wb') as f:
    pickle.dump(systematic_betas, f)
    
with open(os.path.join(resultsdir, f'other_variable_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable, f)
     
with open(os.path.join(resultsdir, f'other_variable_betas_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable_betas, f)
    
# %% viz the rvals
fig,axes = plt.subplots(ncols=2, figsize=(10,3),sharex=True,sharey=True)

for ax, data in zip(axes, [systematic, other_variable]):
    ax.margins(x=0)
    to_viz = data[:,:,:]
    means = np.mean(to_viz, 0)
    std = np.std(to_viz,0)
        
    ax.plot(mult_values,means.T)
            
    for line in list(range(0,to_viz.shape[1])):
        ax.fill_between(mult_values, means.T[:,line]-std.T[:,line], means.T[:,line]+std.T[:,line], alpha=0.4)
        
    ax.set_xlabel('Influence of surprisal (multiplication)')
    ax.set_ylabel('R-value')
        
axes[1].legend(['Node', 'Node + surprisal', 'Node split', 'Node split + surprisal', \
                'Node random split', 'Node random split + surprisal', 'Interaction', \
                    'Node + surprisal + interaction'], frameon=False, bbox_to_anchor=[1,1]) # 'Interaction','Node + surprisal + interaction'
axes[1].get_yaxis().set_visible(False)

axes[0].set_title('Node * surprisal')
axes[1].set_title('Node * third variable')

plt.tight_layout()
sns.despine()

fig.savefig(os.path.join(resultsdir, f'all_ras_comparison_{n_iterations}it{arrayID}.svg'))

#####################################################################################################
# %% PART 2: 800MS NO NOISE
noise = False
kernel_width = 0.8 # 800 milliseconds of response
kernel = sf.impulse_response(sfreq, kernel_width)
resultsdir = '/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set3-figures/log-800ms'

systematic = np.zeros((n_iterations, 8, len(mult_values)))
systematic_betas = []

other_variable = systematic.copy()
other_variable_betas = systematic_betas.copy()

# %% we do this n_iterations of times
start = time.time()

for iteration in list(range(0,n_iterations)):

    # get values for the split predictor
    brackets = 2 * np.random.randn(1, len(tmp_samp)) + 5
    brackets = np.abs(brackets)
    surprisal = np.random.lognormal(mean=2.3, sigma=0.41, size=(1,len(tmp_samp))) # np.random.lognormal(mean=0.5, sigma=1.2, size=(1,len(tmp_samp)))# 4 * np.random.randn(1, len(tmp_samp)) + 17
    #surprisal = np.abs(surprisal) # in 1.02 % of cases, lowest value is below zero, triggers error
    
    some_other_variable = np.random.lognormal(mean=2.3, sigma=0.41,size=(1,len(tmp_samp))) #4 * np.random.randn(1, len(tmp_samp)) + 17 # different var, same distribution
    #some_other_variable = np.abs(some_other_variable)
        
    data_duration = int(max(tmp_samp)+sfreq+max(surprisal[0,:])*max(mult_values))
    
    predictor1 = np.zeros((data_duration))
    predictor2 = np.zeros((data_duration))

    # data - with linear moveable kernel, adjusting the influence of surprisal   
    for var,varname in zip([surprisal, some_other_variable], 
                                 ['systematic', 'random']):
        simulated_rvals = np.zeros((8,len(mult_values)))
        betatmp = {}
        
        for idx,mult in enumerate(mult_values):
            betalist = []
            
            # surprisal kernels
            # in the case of some_other_variable we will use this to generate the data
            # but then never use it again
            surprisal_kernels = [moveable_kernel(surp, kernel,multiplication=mult) for surp in var[0,:]]
            data_list = []
            
            for i,(br,sk) in enumerate(zip(brackets[0,:], surprisal_kernels)):
                response = scipy.signal.convolve(np.expand_dims(br,0), sk)
                data_list.append(response)
        
            # Just this once: plot the data
            #lens = [len(d) for d in data_list]
            #data_list_long = [np.pad(d, (0, max(lens)-len(d)), 'constant', constant_values=(0)) for d in data_list]
            #data_frame = pd.DataFrame(data_list_long, index=np.squeeze(var))
            #data_frame = data_frame.sort_index()
            #fig,ax=plt.subplots()
            #sns.heatmap(data_frame, ax=ax, cmap='viridis')
            #ax.set_title(f'{varname}, {mult}x')
            
            # create data
            data_empty = np.zeros((data_duration+2000))
            
            for index,response in zip(tmp_samp, data_list):
                data_empty[index:index+len(response)] += response
                
            data = data_empty.copy()
            
            if noise: # false
                ns = (sf.spectrum_noise(sf.pink_spectrum, samples=len(data), rate=sfreq) + \
                    np.squeeze(mne.filter.filter_data(np.random.randn(1, len(data)), sfreq=sfreq, \
                                                         l_freq=7, h_freq=12, fir_design='firwin', verbose=False))) * (np.std(data)/10**(0.1/10))
                data += ns
                        
            # create the predictors
            predictor_brackets = np.zeros((len(data)))
            predictor_surprisal = np.zeros((len(data)))
            
            predictor_brackets[tmp_samp] = brackets
            predictor_surprisal[tmp_samp] = surprisal
            
            train_predictor_brackets = predictor_brackets[0:int(4*(len(data)/5)),]
            test_predictor_brackets = predictor_brackets[int(4*(len(data)/5)):,]
            
            ######## one predictor TRF ('node')
            train_data = data.copy()[0:int(4*(len(data)/5))]
            test_data = data.copy()[int(4*(len(data)/5)):]
            
            betas = sf.simulate_TRF(train_data, train_predictor_brackets, lags, alpha=20, fit_intercept=False)
            rvals = sf.simulate_prediction(betas, test_data, test_predictor_brackets, lags)
                    
            simulated_rvals[0,idx] = rvals
            betalist.append(betas)
        
            ######## both predictors ('node + surprisal')
            predictor_both = np.vstack([predictor_brackets, predictor_surprisal]).T
            train_predictor_both = predictor_both[0:int(4*(len(data)/5)),]
            test_predictor_both = predictor_both[int(4*(len(data)/5)):,]
            
            betas_both = sf.simulate_TRF(train_data, train_predictor_both, lags, alpha=20, fit_intercept=False)
            rvals_both = sf.simulate_prediction(betas_both, test_data, test_predictor_both, lags)
            
            simulated_rvals[1,idx] = rvals_both
            betalist.append(betas_both)
        
            ######## split brackets by surprisal
            predictor_high = np.zeros((len(data)))
            predictor_low = np.zeros((len(data)))
            
            low_idx = np.where(surprisal < np.median(surprisal))
            high_idx = np.where(surprisal > np.median(surprisal))
            
            predictor_high[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split = np.vstack([predictor_high, predictor_low]).T
            
            train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
            test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
            
            betas_split = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=20, fit_intercept=False)
            rvals_split = sf.simulate_prediction(betas_split, test_data, test_predictor_split, lags)
            
            simulated_rvals[2,idx] = rvals_split
            betalist.append(betas_split)
        
            ######## split brackets by surprisal + surprisal predictor
            predictor_split_surprisal = np.vstack([predictor_high, predictor_low, predictor_surprisal]).T
            
            train_predictor_split_surprisal = predictor_split_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_surprisal = predictor_split_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_surprisal = sf.simulate_TRF(train_data, train_predictor_split_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_surprisal = sf.simulate_prediction(betas_split_surprisal, test_data, test_predictor_split_surprisal, lags)
            
            simulated_rvals[3,idx] = rvals_split_surprisal
            betalist.append(betas_split_surprisal)
        
            ######## random split predictor
            predictor_high_random = np.zeros((len(data)))
            predictor_low_random = np.zeros((len(data)))
            
            # get indices for random split
            # we generate another random 1000 samples
            random_for_split = np.random.randn(1, len(tmp_samp))
            
            low_idx = np.where(random_for_split < np.median(random_for_split))
            high_idx = np.where(random_for_split > np.median(random_for_split))
                                    
            predictor_high_random[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low_random[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split_random = np.vstack([predictor_high_random, predictor_low_random]).T
            
            train_predictor_split_random = predictor_split_random[0:int(4*(len(data)/5)),]
            test_predictor_split_random = predictor_split_random[int(4*(len(data)/5)):,]
            
            betas_split_random = sf.simulate_TRF(train_data, train_predictor_split_random, lags, alpha=20, fit_intercept=False)
            rvals_split_random = sf.simulate_prediction(betas_split_random, test_data, test_predictor_split_random, lags)
            
            simulated_rvals[4,idx] = rvals_split_random
            betalist.append(betas_split_random)
        
            ######## random split predictor + surprisal
            predictor_split_random_surprisal = np.vstack([predictor_high_random, predictor_low_random, predictor_surprisal]).T
            
            train_predictor_split_random_surprisal = predictor_split_random_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_random_surprisal = predictor_split_random_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_random_surprisal = sf.simulate_TRF(train_data, train_predictor_split_random_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_random_surprisal = sf.simulate_prediction(betas_split_random_surprisal, test_data, test_predictor_split_random_surprisal, lags)
               
            simulated_rvals[5,idx] = rvals_split_random_surprisal
            betalist.append(betas_split_random_surprisal)
        
            ######## interaction term
            interaction = brackets * surprisal
            predictor_interaction = np.zeros((len(data)))
            predictor_interaction[tmp_samp] = interaction
            
            train_predictor_interaction = predictor_interaction[0:int(4*(len(data)/5)),]
            test_predictor_interaction = predictor_interaction[int(4*(len(data)/5)):,]
            
            betas_interaction = sf.simulate_TRF(train_data, train_predictor_interaction, lags, alpha=20, fit_intercept=False)
            rvals_interaction = sf.simulate_prediction(betas_interaction, test_data, test_predictor_interaction, lags)
           
            simulated_rvals[6,idx] = rvals_interaction
            betalist.append(betas_interaction)
        
            ######## add both predictors + interaction
            predictor_both_interaction = np.vstack([predictor_brackets, predictor_surprisal, predictor_interaction]).T
            train_predictor_both_interaction = predictor_both_interaction[0:int(4*(len(data)/5)),]
            test_predictor_both_interaction = predictor_both_interaction[int(4*(len(data)/5)):,]
            
            betas_both_interaction = sf.simulate_TRF(train_data, train_predictor_both_interaction, lags, alpha=20, fit_intercept=False)
            rvals_both_interaction = sf.simulate_prediction(betas_both_interaction, test_data, test_predictor_both_interaction, lags)
            
            simulated_rvals[7,idx] = rvals_both_interaction
            betalist.append(betas_both_interaction)
            
            betatmp[idx] = betalist
        
        if varname == 'systematic':
            systematic[iteration,:,:] = simulated_rvals
            systematic_betas.append(betatmp)
            
        elif varname == 'random':
            other_variable[iteration,:,:] = simulated_rvals
            other_variable_betas.append(betatmp)
    
stop = time.time()
print(f'Duration: {stop-start} s')

# %% store these results
with open(os.path.join(resultsdir, f'systematic_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(systematic, f)
    
with open(os.path.join(resultsdir, f'systematic_betas_{n_iterations}it{arrayID}.pkl'),'wb') as f:
    pickle.dump(systematic_betas, f)
    
with open(os.path.join(resultsdir, f'other_variable_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable, f)
     
with open(os.path.join(resultsdir, f'other_variable_betas_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable_betas, f)
    
# %% viz the rvals
fig,axes = plt.subplots(ncols=2, figsize=(10,3),sharex=True,sharey=True)

for ax, data in zip(axes, [systematic, other_variable]):
    ax.margins(x=0)
    to_viz = data[:,:,:] # [0,1,2,4,6,7]
    means = np.mean(to_viz, 0)
    std = np.std(to_viz,0)
        
    ax.plot(mult_values,means.T)
            
    for line in list(range(0,to_viz.shape[1])):
        ax.fill_between(mult_values, means.T[:,line]-std.T[:,line], means.T[:,line]+std.T[:,line], alpha=0.4)
        
    ax.set_xlabel('Influence of surprisal (multiplication)')
    ax.set_ylabel('R-value')
        
axes[1].legend(['Node', 'Node + surprisal', 'Node split', 'Node split + surprisal', \
                'Node random split', 'Node random split + surprisal', 'Interaction', \
                    'Node + surprisal + interaction'], frameon=False, bbox_to_anchor=[1,1]) # 'Interaction','Node + surprisal + interaction'
axes[1].get_yaxis().set_visible(False)

axes[0].set_title('Node * surprisal')
axes[1].set_title('Node * third variable')

plt.tight_layout()
sns.despine()

fig.savefig(os.path.join(resultsdir, f'all_ras_comparison_{n_iterations}it{arrayID}.svg'))

# %% PART 3: 500MS NOISE
noise = True
kernel_width = 0.5 # 500 milliseconds of response
kernel = sf.impulse_response(sfreq, kernel_width)
resultsdir = '/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set3-figures/noise-log-500ms'

systematic = np.zeros((n_iterations, 8, len(mult_values)))
systematic_betas = []

other_variable = systematic.copy()
other_variable_betas = systematic_betas.copy()

# %% we do this n_iterations of times
start = time.time()

for iteration in list(range(0,n_iterations)):

    # get values for the split predictor
    brackets = 2 * np.random.randn(1, len(tmp_samp)) + 5
    brackets = np.abs(brackets)
    surprisal = np.random.lognormal(mean=2.3, sigma=0.41, size=(1,len(tmp_samp))) # np.random.lognormal(mean=0.5, sigma=1.2, size=(1,len(tmp_samp)))# 4 * np.random.randn(1, len(tmp_samp)) + 17
    #surprisal = np.abs(surprisal) # in 1.02 % of cases, lowest value is below zero, triggers error
    
    some_other_variable = np.random.lognormal(mean=2.3, sigma=0.41,size=(1,len(tmp_samp))) #4 * np.random.randn(1, len(tmp_samp)) + 17 # different var, same distribution
    #some_other_variable = np.abs(some_other_variable)
        
    data_duration = int(max(tmp_samp)+sfreq+max(surprisal[0,:])*max(mult_values))
    
    predictor1 = np.zeros((data_duration))
    predictor2 = np.zeros((data_duration))

    # data - with linear moveable kernel, adjusting the influence of surprisal   
    for var,varname in zip([surprisal, some_other_variable], 
                                 ['systematic', 'random']):
        simulated_rvals = np.zeros((8,len(mult_values)))
        betatmp = {}
        
        for idx,mult in enumerate(mult_values):
            betalist = []
            
            # surprisal kernels
            # in the case of some_other_variable we will use this to generate the data
            # but then never use it again
            surprisal_kernels = [moveable_kernel(surp, kernel,multiplication=mult) for surp in var[0,:]]
            data_list = []
            
            for i,(br,sk) in enumerate(zip(brackets[0,:], surprisal_kernels)):
                response = scipy.signal.convolve(np.expand_dims(br,0), sk)
                data_list.append(response)
        
            # Just this once: plot the data
            #lens = [len(d) for d in data_list]
            #data_list_long = [np.pad(d, (0, max(lens)-len(d)), 'constant', constant_values=(0)) for d in data_list]
            #data_frame = pd.DataFrame(data_list_long, index=np.squeeze(var))
            #data_frame = data_frame.sort_index()
            #fig,ax=plt.subplots()
            #sns.heatmap(data_frame, ax=ax, cmap='viridis')
            #ax.set_title(f'{varname}, {mult}x')
            
            # create data
            data_empty = np.zeros((data_duration+2000))
            
            for index,response in zip(tmp_samp, data_list):
                data_empty[index:index+len(response)] += response
                
            data = data_empty.copy()
            
            if noise: # false
                ns = (sf.spectrum_noise(sf.pink_spectrum, samples=len(data), rate=sfreq) + \
                    np.squeeze(mne.filter.filter_data(np.random.randn(1, len(data)), sfreq=sfreq, \
                                                         l_freq=7, h_freq=12, fir_design='firwin', verbose=False))) * (np.std(data)/10**(0.1/10))
                data += ns
            
            # create the predictors
            predictor_brackets = np.zeros((len(data)))
            predictor_surprisal = np.zeros((len(data)))
            
            predictor_brackets[tmp_samp] = brackets
            predictor_surprisal[tmp_samp] = surprisal
            
            train_predictor_brackets = predictor_brackets[0:int(4*(len(data)/5)),]
            test_predictor_brackets = predictor_brackets[int(4*(len(data)/5)):,]
            
            ######## one predictor TRF ('node')
            train_data = data.copy()[0:int(4*(len(data)/5))]
            test_data = data.copy()[int(4*(len(data)/5)):]
            
            betas = sf.simulate_TRF(train_data, train_predictor_brackets, lags, alpha=20, fit_intercept=False)
            rvals = sf.simulate_prediction(betas, test_data, test_predictor_brackets, lags)
                    
            simulated_rvals[0,idx] = rvals
            betalist.append(betas)
        
            ######## both predictors ('node + surprisal')
            predictor_both = np.vstack([predictor_brackets, predictor_surprisal]).T
            train_predictor_both = predictor_both[0:int(4*(len(data)/5)),]
            test_predictor_both = predictor_both[int(4*(len(data)/5)):,]
            
            betas_both = sf.simulate_TRF(train_data, train_predictor_both, lags, alpha=20, fit_intercept=False)
            rvals_both = sf.simulate_prediction(betas_both, test_data, test_predictor_both, lags)
            
            simulated_rvals[1,idx] = rvals_both
            betalist.append(betas_both)
        
            ######## split brackets by surprisal
            predictor_high = np.zeros((len(data)))
            predictor_low = np.zeros((len(data)))
            
            low_idx = np.where(surprisal < np.median(surprisal))
            high_idx = np.where(surprisal > np.median(surprisal))
            
            predictor_high[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split = np.vstack([predictor_high, predictor_low]).T
            
            train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
            test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
            
            betas_split = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=20, fit_intercept=False)
            rvals_split = sf.simulate_prediction(betas_split, test_data, test_predictor_split, lags)
            
            simulated_rvals[2,idx] = rvals_split
            betalist.append(betas_split)
        
            ######## split brackets by surprisal + surprisal predictor
            predictor_split_surprisal = np.vstack([predictor_high, predictor_low, predictor_surprisal]).T
            
            train_predictor_split_surprisal = predictor_split_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_surprisal = predictor_split_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_surprisal = sf.simulate_TRF(train_data, train_predictor_split_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_surprisal = sf.simulate_prediction(betas_split_surprisal, test_data, test_predictor_split_surprisal, lags)
            
            simulated_rvals[3,idx] = rvals_split_surprisal
            betalist.append(betas_split_surprisal)
        
            ######## random split predictor
            predictor_high_random = np.zeros((len(data)))
            predictor_low_random = np.zeros((len(data)))
            
            # get indices for random split
            # we generate another random 1000 samples
            random_for_split = np.random.randn(1, len(tmp_samp))
            
            low_idx = np.where(random_for_split < np.median(random_for_split))
            high_idx = np.where(random_for_split > np.median(random_for_split))
                                    
            predictor_high_random[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low_random[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split_random = np.vstack([predictor_high_random, predictor_low_random]).T
            
            train_predictor_split_random = predictor_split_random[0:int(4*(len(data)/5)),]
            test_predictor_split_random = predictor_split_random[int(4*(len(data)/5)):,]
            
            betas_split_random = sf.simulate_TRF(train_data, train_predictor_split_random, lags, alpha=20, fit_intercept=False)
            rvals_split_random = sf.simulate_prediction(betas_split_random, test_data, test_predictor_split_random, lags)
            
            simulated_rvals[4,idx] = rvals_split_random
            betalist.append(betas_split_random)
        
            ######## random split predictor + surprisal
            predictor_split_random_surprisal = np.vstack([predictor_high_random, predictor_low_random, predictor_surprisal]).T
            
            train_predictor_split_random_surprisal = predictor_split_random_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_random_surprisal = predictor_split_random_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_random_surprisal = sf.simulate_TRF(train_data, train_predictor_split_random_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_random_surprisal = sf.simulate_prediction(betas_split_random_surprisal, test_data, test_predictor_split_random_surprisal, lags)
               
            simulated_rvals[5,idx] = rvals_split_random_surprisal
            betalist.append(betas_split_random_surprisal)
        
            ######## interaction term
            interaction = brackets * surprisal
            predictor_interaction = np.zeros((len(data)))
            predictor_interaction[tmp_samp] = interaction
            
            train_predictor_interaction = predictor_interaction[0:int(4*(len(data)/5)),]
            test_predictor_interaction = predictor_interaction[int(4*(len(data)/5)):,]
            
            betas_interaction = sf.simulate_TRF(train_data, train_predictor_interaction, lags, alpha=20, fit_intercept=False)
            rvals_interaction = sf.simulate_prediction(betas_interaction, test_data, test_predictor_interaction, lags)
           
            simulated_rvals[6,idx] = rvals_interaction
            betalist.append(betas_interaction)
        
            ######## add both predictors + interaction
            predictor_both_interaction = np.vstack([predictor_brackets, predictor_surprisal, predictor_interaction]).T
            train_predictor_both_interaction = predictor_both_interaction[0:int(4*(len(data)/5)),]
            test_predictor_both_interaction = predictor_both_interaction[int(4*(len(data)/5)):,]
            
            betas_both_interaction = sf.simulate_TRF(train_data, train_predictor_both_interaction, lags, alpha=20, fit_intercept=False)
            rvals_both_interaction = sf.simulate_prediction(betas_both_interaction, test_data, test_predictor_both_interaction, lags)
            
            simulated_rvals[7,idx] = rvals_both_interaction
            betalist.append(betas_both_interaction)
            
            betatmp[idx] = betalist
        
        if varname == 'systematic':
            systematic[iteration,:,:] = simulated_rvals
            systematic_betas.append(betatmp)
            
        elif varname == 'random':
            other_variable[iteration,:,:] = simulated_rvals
            other_variable_betas.append(betatmp)
    
stop = time.time()
print(f'Duration: {stop-start} s')

# %% store these results
with open(os.path.join(resultsdir, f'systematic_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(systematic, f)
    
with open(os.path.join(resultsdir, f'systematic_betas_{n_iterations}it{arrayID}.pkl'),'wb') as f:
    pickle.dump(systematic_betas, f)
    
with open(os.path.join(resultsdir, f'other_variable_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable, f)
     
with open(os.path.join(resultsdir, f'other_variable_betas_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable_betas, f)
    
# %% viz the rvals
fig,axes = plt.subplots(ncols=2, figsize=(10,3),sharex=True,sharey=True)
#plt.xscale('log')

for ax, data in zip(axes, [systematic, other_variable]):
    ax.margins(x=0)
    to_viz = data[:,:,:] # [0,1,2,4,6,7]
    means = np.mean(to_viz, 0)
    std = np.std(to_viz,0)
        
    ax.plot(mult_values,means.T)
            
    for line in list(range(0,to_viz.shape[1])):
        ax.fill_between(mult_values, means.T[:,line]-std.T[:,line], means.T[:,line]+std.T[:,line], alpha=0.4)
        
    ax.set_xlabel('Influence of surprisal (multiplication)')
    ax.set_ylabel('R-value')
        
axes[1].legend(['Node', 'Node + surprisal', 'Node split', 'Node split + surprisal', \
                'Node random split', 'Node random split + surprisal', 'Interaction', \
                    'Node + surprisal + interaction'], frameon=False, bbox_to_anchor=[1,1]) # 'Interaction','Node + surprisal + interaction'
axes[1].get_yaxis().set_visible(False)

axes[0].set_title('Node * surprisal')
axes[1].set_title('Node * third variable')

plt.tight_layout()
sns.despine()

fig.savefig(os.path.join(resultsdir, f'all_ras_comparison_{n_iterations}it{arrayID}.svg'))

# %% PART 3: 800MS NOISE
noise = True
kernel_width = 0.8 # 800 milliseconds of response
kernel = sf.impulse_response(sfreq, kernel_width)
resultsdir = '/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set3-figures/noise-log-800ms'

systematic = np.zeros((n_iterations, 8, len(mult_values)))
systematic_betas = []

other_variable = systematic.copy()
other_variable_betas = systematic_betas.copy()

# %% we do this n_iterations of times
start = time.time()

for iteration in list(range(0,n_iterations)):

    # get values for the split predictor
    brackets = 2 * np.random.randn(1, len(tmp_samp)) + 5
    brackets = np.abs(brackets)
    surprisal = np.random.lognormal(mean=2.3, sigma=0.41, size=(1,len(tmp_samp))) # np.random.lognormal(mean=0.5, sigma=1.2, size=(1,len(tmp_samp)))# 4 * np.random.randn(1, len(tmp_samp)) + 17
    #surprisal = np.abs(surprisal) # in 1.02 % of cases, lowest value is below zero, triggers error
    
    some_other_variable = np.random.lognormal(mean=2.3, sigma=0.41,size=(1,len(tmp_samp))) #4 * np.random.randn(1, len(tmp_samp)) + 17 # different var, same distribution
    #some_other_variable = np.abs(some_other_variable)
        
    data_duration = int(max(tmp_samp)+sfreq+max(surprisal[0,:])*max(mult_values))
    
    predictor1 = np.zeros((data_duration))
    predictor2 = np.zeros((data_duration))

    # data - with linear moveable kernel, adjusting the influence of surprisal   
    for var,varname in zip([surprisal, some_other_variable], 
                                 ['systematic', 'random']):
        simulated_rvals = np.zeros((8,len(mult_values)))
        betatmp = {}
        
        for idx,mult in enumerate(mult_values):
            betalist = []
            
            # surprisal kernels
            # in the case of some_other_variable we will use this to generate the data
            # but then never use it again
            surprisal_kernels = [moveable_kernel(surp, kernel,multiplication=mult) for surp in var[0,:]]
            data_list = []
            
            for i,(br,sk) in enumerate(zip(brackets[0,:], surprisal_kernels)):
                response = scipy.signal.convolve(np.expand_dims(br,0), sk)
                data_list.append(response)
        
            # Just this once: plot the data
            #lens = [len(d) for d in data_list]
            #data_list_long = [np.pad(d, (0, max(lens)-len(d)), 'constant', constant_values=(0)) for d in data_list]
            #data_frame = pd.DataFrame(data_list_long, index=np.squeeze(var))
            #data_frame = data_frame.sort_index()
            #fig,ax=plt.subplots()
            #sns.heatmap(data_frame, ax=ax, cmap='viridis')
            #ax.set_title(f'{varname}, {mult}x')
            
            # create data
            data_empty = np.zeros((data_duration+2000))
            
            for index,response in zip(tmp_samp, data_list):
                data_empty[index:index+len(response)] += response
                
            data = data_empty.copy()
            
            if noise: # false
                ns = (sf.spectrum_noise(sf.pink_spectrum, samples=len(data), rate=sfreq) + \
                    np.squeeze(mne.filter.filter_data(np.random.randn(1, len(data)), sfreq=sfreq, \
                                                         l_freq=7, h_freq=12, fir_design='firwin', verbose=False))) * (np.std(data)/10**(0.1/10))
                data += ns
                
            # create the predictors
            predictor_brackets = np.zeros((len(data)))
            predictor_surprisal = np.zeros((len(data)))
            
            predictor_brackets[tmp_samp] = brackets
            predictor_surprisal[tmp_samp] = surprisal
            
            train_predictor_brackets = predictor_brackets[0:int(4*(len(data)/5)),]
            test_predictor_brackets = predictor_brackets[int(4*(len(data)/5)):,]
            
            ######## one predictor TRF ('node')
            train_data = data.copy()[0:int(4*(len(data)/5))]
            test_data = data.copy()[int(4*(len(data)/5)):]
            
            betas = sf.simulate_TRF(train_data, train_predictor_brackets, lags, alpha=20, fit_intercept=False)
            rvals = sf.simulate_prediction(betas, test_data, test_predictor_brackets, lags)
                    
            simulated_rvals[0,idx] = rvals
            betalist.append(betas)
        
            ######## both predictors ('node + surprisal')
            predictor_both = np.vstack([predictor_brackets, predictor_surprisal]).T
            train_predictor_both = predictor_both[0:int(4*(len(data)/5)),]
            test_predictor_both = predictor_both[int(4*(len(data)/5)):,]
            
            betas_both = sf.simulate_TRF(train_data, train_predictor_both, lags, alpha=20, fit_intercept=False)
            rvals_both = sf.simulate_prediction(betas_both, test_data, test_predictor_both, lags)
            
            simulated_rvals[1,idx] = rvals_both
            betalist.append(betas_both)
        
            ######## split brackets by surprisal
            predictor_high = np.zeros((len(data)))
            predictor_low = np.zeros((len(data)))
            
            low_idx = np.where(surprisal < np.median(surprisal))
            high_idx = np.where(surprisal > np.median(surprisal))
            
            predictor_high[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split = np.vstack([predictor_high, predictor_low]).T
            
            train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
            test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
            
            betas_split = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=20, fit_intercept=False)
            rvals_split = sf.simulate_prediction(betas_split, test_data, test_predictor_split, lags)
            
            simulated_rvals[2,idx] = rvals_split
            betalist.append(betas_split)
        
            ######## split brackets by surprisal + surprisal predictor
            predictor_split_surprisal = np.vstack([predictor_high, predictor_low, predictor_surprisal]).T
            
            train_predictor_split_surprisal = predictor_split_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_surprisal = predictor_split_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_surprisal = sf.simulate_TRF(train_data, train_predictor_split_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_surprisal = sf.simulate_prediction(betas_split_surprisal, test_data, test_predictor_split_surprisal, lags)
            
            simulated_rvals[3,idx] = rvals_split_surprisal
            betalist.append(betas_split_surprisal)
        
            ######## random split predictor
            predictor_high_random = np.zeros((len(data)))
            predictor_low_random = np.zeros((len(data)))
            
            # get indices for random split
            # we generate another random 1000 samples
            random_for_split = np.random.randn(1, len(tmp_samp))
            
            low_idx = np.where(random_for_split < np.median(random_for_split))
            high_idx = np.where(random_for_split > np.median(random_for_split))
                                    
            predictor_high_random[tmp_samp[high_idx[1]]] = brackets[high_idx]
            predictor_low_random[tmp_samp[low_idx[1]]] = brackets[low_idx]
            predictor_split_random = np.vstack([predictor_high_random, predictor_low_random]).T
            
            train_predictor_split_random = predictor_split_random[0:int(4*(len(data)/5)),]
            test_predictor_split_random = predictor_split_random[int(4*(len(data)/5)):,]
            
            betas_split_random = sf.simulate_TRF(train_data, train_predictor_split_random, lags, alpha=20, fit_intercept=False)
            rvals_split_random = sf.simulate_prediction(betas_split_random, test_data, test_predictor_split_random, lags)
            
            simulated_rvals[4,idx] = rvals_split_random
            betalist.append(betas_split_random)
        
            ######## random split predictor + surprisal
            predictor_split_random_surprisal = np.vstack([predictor_high_random, predictor_low_random, predictor_surprisal]).T
            
            train_predictor_split_random_surprisal = predictor_split_random_surprisal[0:int(4*(len(data)/5)),]
            test_predictor_split_random_surprisal = predictor_split_random_surprisal[int(4*(len(data)/5)):,]
            
            betas_split_random_surprisal = sf.simulate_TRF(train_data, train_predictor_split_random_surprisal, lags, alpha=20, fit_intercept=False)
            rvals_split_random_surprisal = sf.simulate_prediction(betas_split_random_surprisal, test_data, test_predictor_split_random_surprisal, lags)
               
            simulated_rvals[5,idx] = rvals_split_random_surprisal
            betalist.append(betas_split_random_surprisal)
        
            ######## interaction term
            interaction = brackets * surprisal
            predictor_interaction = np.zeros((len(data)))
            predictor_interaction[tmp_samp] = interaction
            
            train_predictor_interaction = predictor_interaction[0:int(4*(len(data)/5)),]
            test_predictor_interaction = predictor_interaction[int(4*(len(data)/5)):,]
            
            betas_interaction = sf.simulate_TRF(train_data, train_predictor_interaction, lags, alpha=20, fit_intercept=False)
            rvals_interaction = sf.simulate_prediction(betas_interaction, test_data, test_predictor_interaction, lags)
           
            simulated_rvals[6,idx] = rvals_interaction
            betalist.append(betas_interaction)
        
            ######## add both predictors + interaction
            predictor_both_interaction = np.vstack([predictor_brackets, predictor_surprisal, predictor_interaction]).T
            train_predictor_both_interaction = predictor_both_interaction[0:int(4*(len(data)/5)),]
            test_predictor_both_interaction = predictor_both_interaction[int(4*(len(data)/5)):,]
            
            betas_both_interaction = sf.simulate_TRF(train_data, train_predictor_both_interaction, lags, alpha=20, fit_intercept=False)
            rvals_both_interaction = sf.simulate_prediction(betas_both_interaction, test_data, test_predictor_both_interaction, lags)
            
            simulated_rvals[7,idx] = rvals_both_interaction
            betalist.append(betas_both_interaction)
            
            betatmp[idx] = betalist
        
        if varname == 'systematic':
            systematic[iteration,:,:] = simulated_rvals
            systematic_betas.append(betatmp)
            
        elif varname == 'random':
            other_variable[iteration,:,:] = simulated_rvals
            other_variable_betas.append(betatmp)
    
stop = time.time()
print(f'Duration: {stop-start} s')

# %% store these results
with open(os.path.join(resultsdir, f'systematic_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(systematic, f)
    
with open(os.path.join(resultsdir, f'systematic_betas_{n_iterations}it{arrayID}.pkl'),'wb') as f:
    pickle.dump(systematic_betas, f)
    
with open(os.path.join(resultsdir, f'other_variable_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable, f)
     
with open(os.path.join(resultsdir, f'other_variable_betas_{n_iterations}it{arrayID}.pkl'), 'wb') as f:
    pickle.dump(other_variable_betas, f)
    
# %% viz the rvals
fig,axes = plt.subplots(ncols=2, figsize=(10,3),sharex=True,sharey=True)

for ax, data in zip(axes, [systematic, other_variable]):
    ax.margins(x=0)
    to_viz = data[:,:,:] # [0,1,2,4,6,7]
    means = np.mean(to_viz, 0)
    std = np.std(to_viz,0)
        
    ax.plot(mult_values,means.T)
            
    for line in list(range(0,to_viz.shape[1])):
        ax.fill_between(mult_values, means.T[:,line]-std.T[:,line], means.T[:,line]+std.T[:,line], alpha=0.4)
        
    ax.set_xlabel('Influence of surprisal (multiplication)')
    ax.set_ylabel('R-value')
        
axes[1].legend(['Node', 'Node + surprisal', 'Node split', 'Node split + surprisal', \
                'Node random split', 'Node random split + surprisal', 'Interaction', \
                    'Node + surprisal + interaction'], frameon=False, bbox_to_anchor=[1,1]) # 'Interaction','Node + surprisal + interaction'
axes[1].get_yaxis().set_visible(False)

axes[0].set_title('Node * surprisal')
axes[1].set_title('Node * third variable')

plt.tight_layout()
sns.despine()

fig.savefig(os.path.join(resultsdir, f'all_ras_comparison_{n_iterations}it{arrayID}.svg'))

###############################

print('finished...')