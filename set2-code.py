#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:25:43 2022

@author: sopsla
"""
# math
import numpy as np
import pandas as pd
import scipy

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# trf stuff
import simfun as sf
from pyeeg.utils import lag_span
import mne

# %% general settings
sfreq = 120
kernel_width = 0.5 # 500 milliseconds of response
isi = 0.6 # 600 milliseconds between impulses
response_no = 100 # number of responses
participant_no = 40 # number of participants
noise = False
tmin = -0.2
tmax = 0.8

savedir = '/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set2-figures'

# %% setting up the kernel and predictors
# kernel
kernel = sf.impulse_response(sfreq, kernel_width)

fig,ax = plt.subplots(figsize=(5,3))
ax.plot(np.asarray(range(len(kernel)))/sfreq, kernel)
ax.set_xlabel('Time (s)')
ax.set_ylabel('coeff. (a.u.)')
ax.set_title('Impulse response (ground truth)')
sns.despine()
plt.tight_layout()

fig.savefig(f'{savedir}/simulations-impulseresponse.svg')

# predictor & data timing
isi_samples = isi * sfreq * np.ones((1, response_no))
tmp_samp = np.asarray(np.cumsum(isi_samples), int)
data_duration = int(max(tmp_samp))+sfreq

predictor1 = np.zeros((data_duration))
predictor2 = np.zeros((data_duration))

# %% SET 1: two factors, factor 2 does nothing, both normally distr., no interaction
betas_one = []
rvals_one = []
betas_two = []
rvals_two = []
    
# get values for the split predictor
values2 = np.random.randn(1, len(tmp_samp))
allvals = pd.DataFrame(data = np.asarray([np.zeros((len(values2[0]))),values2[0]]).T, columns=['values1', 'values2'])

# high and low predictors -- *EXACTLY* THE SAME HERE
high = np.random.randn(1, int(len(tmp_samp)/2))
low = high.copy()

allvals['high'] = np.zeros((len(allvals)))
allvals['low'] = np.zeros((len(allvals)))

allvals.loc[allvals['values2'] > np.median(allvals['values2']), 'high'] = np.squeeze(high)
allvals.loc[allvals['values2'] <= np.median(allvals['values2']), 'low'] = np.squeeze(low)
allvals = allvals.fillna(0)

allvals['values1'] = allvals['high'] + allvals['low']

predictor1[tmp_samp] = allvals['values1']

for participant in list(range(participant_no)):
    
    # base data
    data = np.convolve(predictor1, np.pad(kernel, (int(sfreq*(abs(max(0.7+0, 0))))-1,
                                           int(sfreq*(abs(min(0.7+0, 0))))), mode='constant'), 'same')
    #data = scipy.signal.convolve(predictor1, kernel, 'same')
    
    if noise:
        data += np.squeeze(mne.filter.filter_data(np.random.randn(1, data_duration), sfreq=sfreq,
                                             l_freq=7, h_freq=12, fir_design='firwin', verbose=False))
    
    # simple TRF on data
    lags = lag_span(tmin, tmax, sfreq)
    train_data = data[0:int(4*(len(data)/5))]
    test_data = data[int(4*(len(data)/5)):]
    
    #predictor1 = np.expand_dims(predictor1,1)
    train_predictor = predictor1[0:int(4*(len(data)/5)),]
    test_predictor = predictor1[int(4*(len(data)/5)):,]
    
    beta_one = sf.simulate_TRF(train_data, train_predictor, lags, alpha=1, fit_intercept=False)
    betas_one.append(beta_one)
    
    rvals_one.append(sf.simulate_prediction(beta_one, test_data, test_predictor, lags))
    
    predictor_high = predictor2.copy()
    predictor_low = predictor2.copy()
    predictor_high[tmp_samp] = allvals['high']
    predictor_low[tmp_samp] = allvals['low']
    
    # plot the distributions of the predictors
    predictor_split = np.vstack([predictor_high, predictor_low]).T
    train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
    test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
    
    # the TRF
    beta_two = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=1, fit_intercept=False)
    betas_two.append(beta_two)
    rvals_two.append(sf.simulate_prediction(beta_two, test_data, test_predictor_split, lags))
    
# %% plotting
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(9,6))

# plot the predictors - together
sns.kdeplot(data=allvals['values1'], fill=True, ax=ax[0,0]) # they are the same
ax[0,0].set_ylabel('Density')
ax[0,0].set_xlabel('predictor value')
ax[0,0].set_title('Single predictor values')

# plot the single TRF
ax[0,1].plot(lags/sfreq, np.mean(betas_one, axis=0))
ax[0,1].fill_between(x=lags/sfreq, y1=np.mean(betas_one, axis=0)-np.std(betas_one, axis=0), y2=np.mean(betas_one, axis=0)+np.std(betas_one,axis=0), alpha=0.5)
ax[0,1].set_ylabel('coeff. (a.u.)')
ax[0,1].set_xlabel('time (s)')
ax[0,1].set_title('Single TRF')

# plot the predictors - split
sns.kdeplot(data=allvals[['high', 'low']], fill=True, ax=ax[1,0]) # they are the same
ax[1,0].legend(['high', 'low'], frameon=False)
ax[1,0].set_ylabel('Density')
ax[1,0].set_xlabel('predictor value')
ax[1,0].set_title('Split predictor values')

ax[1,1].plot(lags/sfreq, np.mean(betas_two, axis=0))
for ft in [0,1]:
    ax[1,1].fill_between(x=lags/sfreq, y1=np.mean(betas_two, axis=0)[:,ft]-np.std(betas_two, axis=0)[:,ft], y2=np.mean(betas_two, axis=0)[:,ft]+np.std(betas_two,axis=0)[:,ft], alpha=0.5)
ax[1,1].legend(['high', 'low'], frameon=False)
ax[1,1].set_ylabel('coeff. (a.u.)')
ax[1,1].set_xlabel('time (s)')
ax[1,1].set_title('Two TRFs')
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-samedistribution-noise.svg'
else:
    fname = 'simulations-samedistribution.svg'
fig.savefig(f'{savedir}/{fname}')

fig,ax=plt.subplots(figsize=(4,4))
sns.barplot(data=[rvals_one, rvals_two], alpha=0.7)
ax.set_xticklabels(['one TRF', 'two TRFs'])
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-samedistribution-noise-bar.svg'
else:
    fname = 'simulations-samedistribution-bar.svg'
fig.savefig(f'{savedir}/{fname}')

# %% SET 2: two factors, different averages. This is the most important one!!!
betas_one = []
rvals_one = []
betas_two = []
rvals_two = []

predictor1 = np.zeros((data_duration))
predictor2 = np.zeros((data_duration))
    
# get values for the split predictor
values2 = np.random.randn(1, len(tmp_samp))
allvals = pd.DataFrame(data = np.asarray([np.zeros((len(values2[0]))),values2[0]]).T, columns=['values1', 'values2'])

# high and low predictors -- *EXACTLY* THE SAME HERE
high = np.random.randn(1, int(len(tmp_samp)/2))
low = 3.5 * np.random.randn(1, int(len(tmp_samp)/2)) + 4.02

allvals.loc[allvals['values2'] > np.median(allvals['values2']), 'high'] = np.squeeze(high)
allvals.loc[allvals['values2'] <= np.median(allvals['values2']), 'low'] = np.squeeze(low)
allvals = allvals.fillna(0)

allvals['values1'] = allvals['high'] + allvals['low']

predictor1[tmp_samp] = allvals['values1']

for participant in list(range(participant_no)):
    
    # base data
    data = np.convolve(predictor1, np.pad(kernel, (int(sfreq*(abs(max(0.7+0, 0))))-1,
                                           int(sfreq*(abs(min(0.7+0, 0))))), mode='constant'), 'same')
    
    if noise:
        data += np.squeeze(mne.filter.filter_data(np.random.randn(1, data_duration), sfreq=sfreq,
                                             l_freq=7, h_freq=12, fir_design='firwin', verbose=False))
    
    # simple TRF on data
    lags = lag_span(tmin, tmax, sfreq)
    train_data = data[0:int(4*(len(data)/5))]
    test_data = data[int(4*(len(data)/5)):]
    
    #predictor1 = np.expand_dims(predictor1,1)
    train_predictor = predictor1[0:int(4*(len(data)/5)),]
    test_predictor = predictor1[int(4*(len(data)/5)):,]
    
    beta_one = sf.simulate_TRF(train_data, train_predictor, lags, alpha=1, fit_intercept=False)
    betas_one.append(beta_one)
    
    rvals_one.append(sf.simulate_prediction(beta_one, test_data, test_predictor, lags))
    
    predictor_high = predictor2.copy()
    predictor_low = predictor2.copy()
    predictor_high[tmp_samp] = allvals['high']
    predictor_low[tmp_samp] = allvals['low']
    
    # plot the distributions of the predictors
    predictor_split = np.vstack([predictor_high, predictor_low]).T
    train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
    test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
    
    # the TRF
    beta_two = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=1, fit_intercept=False)
    betas_two.append(beta_two)
    rvals_two.append(sf.simulate_prediction(beta_two, test_data, test_predictor_split, lags))
    
# %% plotting
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(9,6))

# plot the predictors - together
sns.kdeplot(data=allvals['values1'], fill=True, ax=ax[0,0]) # they are the same
ax[0,0].set_ylabel('Density')
ax[0,0].set_xlabel('predictor value')
ax[0,0].set_title('Single predictor values')

# plot the single TRF
ax[0,1].plot(lags/sfreq, np.mean(betas_one, axis=0))
ax[0,1].fill_between(x=lags/sfreq, y1=np.mean(betas_one, axis=0)-np.std(betas_one, axis=0), y2=np.mean(betas_one, axis=0)+np.std(betas_one,axis=0), alpha=0.5)
ax[0,1].set_ylabel('coeff. (a.u.)')
ax[0,1].set_xlabel('time (s)')
ax[0,1].set_title('Single TRF')

# plot the predictors - split
sns.kdeplot(data=allvals[['high', 'low']], fill=True, ax=ax[1,0]) # they are the same
ax[1,0].legend(['high', 'low'], frameon=False)
ax[1,0].set_ylabel('Density')
ax[1,0].set_xlabel('predictor value')
ax[1,0].set_title('Split predictor values')

ax[1,1].plot(lags/sfreq, np.mean(betas_two, axis=0))
for ft in [0,1]:
    ax[1,1].fill_between(x=lags/sfreq, y1=np.mean(betas_two, axis=0)[:,ft]-np.std(betas_two, axis=0)[:,ft], y2=np.mean(betas_two, axis=0)[:,ft]+np.std(betas_two,axis=0)[:,ft], alpha=0.5)
ax[1,1].legend(['high', 'low'], frameon=False)
ax[1,1].set_ylabel('coeff. (a.u.)')
ax[1,1].set_xlabel('time (s)')
ax[1,1].set_title('Two TRFs')
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-diffdistribution-noise.svg'
else:
    fname = 'simulations-diffdistribution.svg'
fig.savefig(f'{savedir}/{fname}')

fig,ax=plt.subplots(figsize=(4,4))
sns.barplot(data=[rvals_one, rvals_two], alpha=0.7)
ax.set_xticklabels(['one TRF', 'two TRFs'])
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-diffdistribution-noise-bar.svg'
else:
    fname = 'simulations-diffdistribution-bar.svg'
fig.savefig(f'{savedir}/{fname}')
# so far so good

# %% set 3: interaction between the response of the two predictors but the division does not lead to different averages
betas_one = []
rvals_one = []
betas_two = []
rvals_two = []

predictor1 = np.zeros((data_duration))
predictor2 = np.zeros((data_duration))
    
# get values for the split predictor
values2 = np.random.randn(1, len(tmp_samp))
allvals = pd.DataFrame(data = np.asarray([np.zeros((len(values2[0]))),values2[0]]).T, columns=['values1', 'values2'])

# high and low predictors -- *EXACTLY* THE SAME HERE
high = np.random.randn(1, int(len(tmp_samp)/2))
low = high.copy()

allvals.loc[allvals['values2'] > np.median(allvals['values2']), 'high'] = np.squeeze(high)
allvals.loc[allvals['values2'] <= np.median(allvals['values2']), 'low'] = np.squeeze(low)
allvals = allvals.fillna(0)

allvals['values1'] = allvals['high'] + allvals['low']

predictor1[tmp_samp] = allvals['values1']
predictor2[tmp_samp] = allvals['values1'] * allvals['values2']

for participant in list(range(participant_no)):
    
    # base data
    data = np.convolve(predictor2, np.pad(kernel, (int(sfreq*(abs(max(0.7+0, 0))))-1,
                                           int(sfreq*(abs(min(0.7+0, 0))))), mode='constant'), 'same')
    if noise:
        data += np.squeeze(mne.filter.filter_data(np.random.randn(1, data_duration), sfreq=sfreq,
                                             l_freq=7, h_freq=12, fir_design='firwin', verbose=False))
    
    # simple TRF on data
    lags = lag_span(tmin, tmax, sfreq)
    train_data = data[0:int(4*(len(data)/5))]
    test_data = data[int(4*(len(data)/5)):]
    
    #predictor1 = np.expand_dims(predictor1,1)
    train_predictor = predictor1[0:int(4*(len(data)/5)),]
    test_predictor = predictor1[int(4*(len(data)/5)):,]
    
    beta_one = sf.simulate_TRF(train_data, train_predictor, lags, alpha=1, fit_intercept=False)
    betas_one.append(beta_one)
    
    rvals_one.append(sf.simulate_prediction(beta_one, test_data, test_predictor, lags))
    
    predictor_high = predictor2.copy()
    predictor_low = predictor2.copy()
    predictor_high[tmp_samp] = allvals['high']
    predictor_low[tmp_samp] = allvals['low']
    
    # plot the distributions of the predictors
    predictor_split = np.vstack([predictor_high, predictor_low]).T
    train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
    test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
    
    # the TRF
    beta_two = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=1, fit_intercept=False)
    betas_two.append(beta_two)
    rvals_two.append(sf.simulate_prediction(beta_two, test_data, test_predictor_split, lags))
    
# %% plotting
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(9,6))

# plot the predictors - together
sns.kdeplot(data=allvals['values1'], fill=True, ax=ax[0,0]) # they are the same
ax[0,0].set_ylabel('Density')
ax[0,0].set_xlabel('predictor value')
ax[0,0].set_title('Single predictor values')

# plot the single TRF
ax[0,1].plot(lags/sfreq, np.mean(betas_one, axis=0))
ax[0,1].fill_between(x=lags/sfreq, y1=np.mean(betas_one, axis=0)-np.std(betas_one, axis=0), y2=np.mean(betas_one, axis=0)+np.std(betas_one,axis=0), alpha=0.5)
ax[0,1].set_ylabel('coeff. (a.u.)')
ax[0,1].set_xlabel('time (s)')
ax[0,1].set_title('Single TRF')

# plot the predictors - split
sns.kdeplot(data=allvals[['high', 'low']], fill=True, ax=ax[1,0]) # they are the same
ax[1,0].legend(['high', 'low'], frameon=False)
ax[1,0].set_ylabel('Density')
ax[1,0].set_xlabel('predictor value')
ax[1,0].set_title('Split predictor values')

ax[1,1].plot(lags/sfreq, np.mean(betas_two, axis=0))
for ft in [0,1]:
    ax[1,1].fill_between(x=lags/sfreq, y1=np.mean(betas_two, axis=0)[:,ft]-np.std(betas_two, axis=0)[:,ft], y2=np.mean(betas_two, axis=0)[:,ft]+np.std(betas_two,axis=0)[:,ft], alpha=0.5)
ax[1,1].legend(['high', 'low'], frameon=False)
ax[1,1].set_ylabel('coeff. (a.u.)')
ax[1,1].set_xlabel('time (s)')
ax[1,1].set_title('Two TRFs')
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-samedistribution-interaction-noise.svg'
else:
    fname = 'simulations-samedistribution-interaction.svg'
fig.savefig(f'{savedir}/{fname}')

fig,ax=plt.subplots(figsize=(4,4))
sns.barplot(data=[rvals_one, rvals_two], alpha=0.7)
ax.set_xticklabels(['one TRF', 'two TRFs'])
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-samedistribution-interaction-noise-bar.svg'
else:
    fname = 'simulations-samedistribution-interaction-bar.svg'
fig.savefig(f'{savedir}/{fname}')

# %% SET 4: different average distributions AND interaction
betas_one = []
rvals_one = []
betas_two = []
rvals_two = []

predictor1 = np.zeros((data_duration))
predictor2 = np.zeros((data_duration))
    
# get values for the split predictor
values2 = np.random.randn(1, len(tmp_samp))
allvals = pd.DataFrame(data = np.asarray([np.zeros((len(values2[0]))),values2[0]]).T, columns=['values1', 'values2'])

# high and low predictors -- *EXACTLY* THE SAME HERE
high = np.random.randn(1, int(len(tmp_samp)/2))
low = 3.5 * np.random.randn(1, int(len(tmp_samp)/2)) + 4.02

allvals.loc[allvals['values2'] > np.median(allvals['values2']), 'high'] = np.squeeze(high)
allvals.loc[allvals['values2'] <= np.median(allvals['values2']), 'low'] = np.squeeze(low)
allvals = allvals.fillna(0)

allvals['values1'] = allvals['high'] + allvals['low']

predictor1[tmp_samp] = allvals['values1']
predictor2[tmp_samp] = allvals['values1'] * allvals['values2']

for participant in list(range(participant_no)):
    
    # base data
    data = np.convolve(predictor2, np.pad(kernel, (int(sfreq*(abs(max(0.7+0, 0))))-1,
                                           int(sfreq*(abs(min(0.7+0, 0))))), mode='constant'), 'same')
    
    if noise:
        data += np.squeeze(mne.filter.filter_data(np.random.randn(1, data_duration), sfreq=sfreq,
                                             l_freq=7, h_freq=12, fir_design='firwin', verbose=False))
    
    # simple TRF on data
    lags = lag_span(tmin, tmax, sfreq)
    train_data = data[0:int(4*(len(data)/5))]
    test_data = data[int(4*(len(data)/5)):]
    
    #predictor1 = np.expand_dims(predictor1,1)
    train_predictor = predictor1[0:int(4*(len(data)/5)),]
    test_predictor = predictor1[int(4*(len(data)/5)):,]
    
    beta_one = sf.simulate_TRF(train_data, train_predictor, lags, alpha=1, fit_intercept=False)
    betas_one.append(beta_one)
    
    rvals_one.append(sf.simulate_prediction(beta_one, test_data, test_predictor, lags))
    
    predictor_high = predictor2.copy()
    predictor_low = predictor2.copy()
    predictor_high[tmp_samp] = allvals['high']
    predictor_low[tmp_samp] = allvals['low']
    
    # plot the distributions of the predictors
    predictor_split = np.vstack([predictor_high, predictor_low]).T
    train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
    test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
    
    # the TRF
    beta_two = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=1, fit_intercept=False)
    betas_two.append(beta_two)
    rvals_two.append(sf.simulate_prediction(beta_two, test_data, test_predictor_split, lags))
    
# %% plotting
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(9,6))

# plot the predictors - together
sns.kdeplot(data=allvals['values1'], fill=True, ax=ax[0,0]) # they are the same
ax[0,0].set_ylabel('Density')
ax[0,0].set_xlabel('predictor value')
ax[0,0].set_title('Single predictor values')

# plot the single TRF
ax[0,1].plot(lags/sfreq, np.mean(betas_one, axis=0))
ax[0,1].fill_between(x=lags/sfreq, y1=np.mean(betas_one, axis=0)-np.std(betas_one, axis=0), y2=np.mean(betas_one, axis=0)+np.std(betas_one,axis=0), alpha=0.5)
ax[0,1].set_ylabel('coeff. (a.u.)')
ax[0,1].set_xlabel('time (s)')
ax[0,1].set_title('Single TRF')

# plot the predictors - split
sns.kdeplot(data=allvals[['high', 'low']], fill=True, ax=ax[1,0]) # they are the same
ax[1,0].legend(['high', 'low'], frameon=False)
ax[1,0].set_ylabel('Density')
ax[1,0].set_xlabel('predictor value')
ax[1,0].set_title('Split predictor values')

ax[1,1].plot(lags/sfreq, np.mean(betas_two, axis=0))
for ft in [0,1]:
    ax[1,1].fill_between(x=lags/sfreq, y1=np.mean(betas_two, axis=0)[:,ft]-np.std(betas_two, axis=0)[:,ft], y2=np.mean(betas_two, axis=0)[:,ft]+np.std(betas_two,axis=0)[:,ft], alpha=0.5)
ax[1,1].legend(['high', 'low'], frameon=False)
ax[1,1].set_ylabel('coeff. (a.u.)')
ax[1,1].set_xlabel('time (s)')
ax[1,1].set_title('Two TRFs')
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-diffdistribution-interaction-noise.svg'
else:
    fname = 'simulations-diffdistribution-interaction.svg'
fig.savefig(f'{savedir}/{fname}')

fig,ax=plt.subplots(figsize=(4,4))
sns.barplot(data=[rvals_one, rvals_two], alpha=0.7)
ax.set_xticklabels(['one TRF', 'two TRFs'])
sns.despine()
plt.tight_layout()

if noise:
    fname = 'simulations-diffdistribution-interaction-noise-bar.svg'
else:
    fname = 'simulations-diffdistribution-interaction-bar.svg'
fig.savefig(f'{savedir}/{fname}')

# %% SET 5: smaller interaction effect
all_betas_one = []
all_rvals_one = []
all_betas_two = []
all_rvals_two = []

predictor1 = np.zeros((data_duration))
predictor2 = np.zeros((data_duration))

# get values for the split predictor
values2 = np.random.randn(1, len(tmp_samp))
allvals = pd.DataFrame(data = np.asarray([np.zeros((len(values2[0]))),values2[0]]).T, columns=['values1', 'values2'])

# high and low predictors -- *EXACTLY* THE SAME HERE
high = np.random.randn(1, int(len(tmp_samp)/2))
low = 1.5 * np.random.randn(1, int(len(tmp_samp)/2)) + 0.02

allvals.loc[allvals['values2'] > np.median(allvals['values2']), 'high'] = np.squeeze(high)
allvals.loc[allvals['values2'] <= np.median(allvals['values2']), 'low'] = np.squeeze(low)
allvals = allvals.fillna(0)

allvals['values1'] = allvals['high'] + allvals['low']

for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    
    betas_one = []
    rvals_one = []
    betas_two = []
    rvals_two = []
    
    predictor1[tmp_samp] = allvals['values1']
    predictor2[tmp_samp] = allvals['values1'] * (i * allvals['values2'])
    
    for participant in list(range(participant_no)):
        
        # base data
        data = np.convolve(predictor2, np.pad(kernel, (int(sfreq*(abs(max(0.7+0, 0))))-1,
                                           int(sfreq*(abs(min(0.7+0, 0))))), mode='constant'), 'same')
        
        if noise:
            data += np.squeeze(mne.filter.filter_data(np.random.randn(1, data_duration), sfreq=sfreq,
                                                 l_freq=7, h_freq=12, fir_design='firwin', verbose=False))
        
        # simple TRF on data
        lags = lag_span(tmin, tmax, sfreq)
        train_data = data[0:int(4*(len(data)/5))]
        test_data = data[int(4*(len(data)/5)):]
        
        #predictor1 = np.expand_dims(predictor1,1)
        train_predictor = predictor1[0:int(4*(len(data)/5)),]
        test_predictor = predictor1[int(4*(len(data)/5)):,]
        
        beta_one = sf.simulate_TRF(train_data, train_predictor, lags, alpha=1, fit_intercept=False)
        betas_one.append(beta_one)
        
        rvals_one.append(sf.simulate_prediction(beta_one, test_data, test_predictor, lags))
        
        predictor_high = predictor2.copy()
        predictor_low = predictor2.copy()
        predictor_high[tmp_samp] = allvals['high']
        predictor_low[tmp_samp] = allvals['low']
        
        # plot the distributions of the predictors
        predictor_split = np.vstack([predictor_high, predictor_low]).T
        train_predictor_split = predictor_split[0:int(4*(len(data)/5)),]
        test_predictor_split = predictor_split[int(4*(len(data)/5)):,]
        
        # the TRF
        beta_two = sf.simulate_TRF(train_data, train_predictor_split, lags, alpha=1, fit_intercept=False)
        betas_two.append(beta_two)
        rvals_two.append(sf.simulate_prediction(beta_two, test_data, test_predictor_split, lags))
        
    # plotting
    fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    
    # plot the predictors - together
    sns.kdeplot(data=allvals['values1'], fill=True, ax=ax[0,0]) # they are the same
    ax[0,0].set_ylabel('Density')
    ax[0,0].set_xlabel('predictor value')
    ax[0,0].set_title('Single predictor values')
    
    # plot the single TRF
    ax[0,1].plot(lags/sfreq, np.mean(betas_one, axis=0))
    ax[0,1].fill_between(x=lags/sfreq, y1=np.mean(betas_one, axis=0)-np.std(betas_one, axis=0), y2=np.mean(betas_one, axis=0)+np.std(betas_one,axis=0), alpha=0.5)
    ax[0,1].set_ylabel('coeff. (a.u.)')
    ax[0,1].set_xlabel('time (s)')
    ax[0,1].set_title('Single TRF')
    
    # plot the predictors - split
    sns.kdeplot(data=allvals[['high', 'low']], fill=True, ax=ax[1,0]) # they are the same
    ax[1,0].legend(['high', 'low'], frameon=False)
    ax[1,0].set_ylabel('Density')
    ax[1,0].set_xlabel('predictor value')
    ax[1,0].set_title('Split predictor values')
    
    ax[1,1].plot(lags/sfreq, np.mean(betas_two, axis=0))
    for ft in [0,1]:
        ax[1,1].fill_between(x=lags/sfreq, y1=np.mean(betas_two, axis=0)[:,ft]-np.std(betas_two, axis=0)[:,ft], y2=np.mean(betas_two, axis=0)[:,ft]+np.std(betas_two,axis=0)[:,ft], alpha=0.5)
    ax[1,1].legend(['high', 'low'], frameon=False)
    ax[1,1].set_ylabel('coeff. (a.u.)')
    ax[1,1].set_xlabel('time (s)')
    ax[1,1].set_title('Two TRFs')
    sns.despine()
    plt.tight_layout()
    
    fig.savefig(f'{savedir}/simulations-interaction-diffdistribution-nonoise-{str(i)}.svg')
    
    fig,ax=plt.subplots(figsize=(4,4))
    sns.barplot(data=[rvals_one, rvals_two], alpha=0.7)
    ax.set_xticklabels(['one TRF', 'two TRFs'])
    sns.despine()
    plt.tight_layout()
    
    fig.savefig(f'{savedir}/simulations-interaction-diffdistribution-nonoise-{str(i)}-bar.svg')
    
    all_betas_one.append(betas_one)
    all_rvals_one.append(rvals_one)
    all_betas_two.append(betas_two)
    all_rvals_two.append(rvals_two)
        