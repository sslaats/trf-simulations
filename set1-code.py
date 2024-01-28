#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:17:30 2023

@author: sopsla
"""
# global modules
import os
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# stats etc
import scipy.stats as stats
import scipy.signal as signal
import math
import random

# meg 
import mne
from pyeeg.utils import lag_span, lag_matrix

# local - simulation basics
import simfun as sf

# %% general parameters
random.seed(10)

# data
fs = 120
data_duration = 100 # seconds
dt = 1/fs
delta_freqs = [1, 3]
theta_freqs = [4, 8]
hp_freq = [0.1, None]

# stimuli
n_stimuli = 100

# TRF parameters
tmin = -0.2
tmax = 0.8
fit_intercept = True
alpha = 10
lags = lag_span(tmin=tmin,tmax=tmax, srate=fs)

ISIs = np.arange(0.1, 0.9, 0.05)
SNRs = np.append(np.arange(-3, 3, 0.5), 999)

# %% create the kernels
tlags, krn_delta = sf.generate_kernel(0.3, 0.8, type='sines', tmin=-0.2, tmax=0.8, freqs=[2.], fs=120)
tlags, krn_theta = sf.generate_kernel(0.3, 0.8, type='sines', tmin=-0.2, tmax=0.8, freqs=[6.], fs=120)

# %% plot them
fig, ax = plt.subplots(ncols=2, figsize=(9,3))

ax[0].plot(np.asarray(range(0, len(krn_delta)))/fs, krn_delta)
ax[0].plot(tlags, krn_theta)
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('Magnitude (a.u.)')

sf.plot_powerspectrum(krn_delta, fs=fs, n_samples=len(krn_delta), ax=ax[1], logscale=True)
sf.plot_powerspectrum(krn_theta, fs=fs, n_samples=len(krn_theta), ax=ax[1], logscale=True)

plt.legend(['delta', 'theta', 'beta'], frameon=False)
ax[1].axvspan(delta_freqs[0], delta_freqs[1], alpha=0.3, color='lightblue')
ax[1].axvspan(theta_freqs[0], theta_freqs[1], alpha=0.2, color='salmon')

fig.tight_layout()

sns.despine()

delta_theta_kernels = [krn_delta, krn_theta]

fig.savefig('/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set1-figures/kernels.svg')

# %% Question 1: What does ISI do to the accuracy?
# # We vary (1) the amount of noise and (2) whether data length is identical

# %% 1a we vary the interstimulus interval
df = pd.DataFrame(columns=['isi', 'snr', 'PearsonsR', 'PearsonsR_truncated'])

for isi in ISIs:
    for snr in SNRs:
        tmp_df = pd.DataFrame(columns=['isi', 'snr', 'PearsonsR', 'PearsonsR_truncated'])
        _, _, x, _ = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing=True)
        
        if snr == 999:
            
            # data
            y, _ = sf.simulate_response(krn_delta, x, snr)
                    
            # estimate TRF
            beta = sf.simulate_TRF(y, x, lags, alpha, fit_intercept=True)
            
            # generate test data
            _, _, xtest, tend = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing = True)
            ytest, _ = sf.simulate_response(krn_delta, xtest, snr)
                    
            snr = 'clean'
            
        else:
        
            # generate stimulus
            _, _, x, _ = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing=True)
            
            # data
            _, y = sf.simulate_response(krn_delta, x, snr)
                    
            # estimate TRF
            beta = sf.simulate_TRF(y, x, lags, alpha, fit_intercept=True)
            
            # generate test data
            _, _, xtest, tend = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing = True)
            _, ytest = sf.simulate_response(krn_delta, xtest, snr)
                    
        # obtain reconstruction accuracy
        accuracy = sf.simulate_prediction(beta, ytest, xtest, lags, prediction_type='corrcoef')
        tmp_df['PearsonsR'] = accuracy
        
        # score only on data that contains response
        xtest = xtest[:round((tend+tmax)*fs)]
        ytest = ytest[:round((tend+tmax)*fs)]
        
        accuracy = sf.simulate_prediction(beta, ytest, xtest, lags, prediction_type='corrcoef')
            
        tmp_df['PearsonsR_truncated'] = accuracy
        tmp_df['isi'] = isi
        tmp_df['snr'] = snr
        
        df = pd.concat([df, tmp_df], ignore_index=True)
        
# %% plot it
fig,axes=plt.subplots(ncols=2, figsize=(9,3), sharey=True, sharex=True)

for p_type,ax in zip(['PearsonsR', 'PearsonsR_truncated'], axes):

    sns.lineplot(x='isi', y=p_type, hue='snr', data=df.loc[df['snr'] != 'clean'], ax=ax, palette='crest')
    sns.lineplot(x='isi', y=p_type, data=df.loc[df['snr']=='clean'],color='red', label='no noise', ax=ax)
    ax.set_xlabel('Interstimulus interval (s)')
    ax.set_ylabel("Pearson's R")

axes[0].legend([],[],frameon=False)
axes[1].get_yaxis().set_visible(False)
legend = axes[1].legend(frameon=False, title='SNR', bbox_to_anchor=[1,1])

renderer = fig.canvas.get_renderer()

max_shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
for t in legend.get_texts():
    t.set_ha('right')  #
    temp_shift = max_shift - t.get_window_extent().width
    t.set_position((temp_shift, 0))

axes[0].set_title('Constant signal length')
axes[1].set_title('Variable signal length')

sns.despine()
plt.tight_layout()

fig.savefig('/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set1-figures/ISI-snr.svg')

# %% Question 2: Does finding an effect in one frequency band but not another mean that the
# response is truly in this frequency band?
# we do this, again, for varying amounts of SNR, but this time only a single ISI
isi = 0.9

freq_df = pd.DataFrame(columns=['snr','filter_band','train_predictor', 'PearsonsR'])

# generate stimulus
_, _, xtheta_train, _ = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing=True)
_, _, xdelta_train, _ = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing=True)

_, _, xtheta_test, _ = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing=True)
_, _, xdelta_test, _ = sf.simulate_stimulus(isi, n_stimuli, data_duration, smoothing=True)

xboth_train = np.vstack([xtheta_train, xdelta_train]).T
xboth_test = np.vstack([xtheta_test, xdelta_test]).T

for snr in SNRs:
    
    if snr == 999:
         # train data - the same in all cases
        ytheta, _ = sf.simulate_response(krn_theta, xtheta_train, snr)
        ydelta, _ = sf.simulate_response(krn_delta, xdelta_train, snr)
        
        ytrain_signal = ytheta + ydelta 
        ytrain = ytrain_signal
        
        # test data - same in all cases, except filtered
        ytheta, _ = sf.simulate_response(krn_theta, xtheta_test, snr)
        ydelta, _ = sf.simulate_response(krn_delta, xdelta_test, snr)
        
        ytest_signal = ytheta + ydelta 
        ytest = ytest_signal
    
        snr = 'clean'
        
    else:
      
        # train data - the same in all cases
        ytheta, _ = sf.simulate_response(krn_theta, xtheta_train, snr)
        ydelta, _ = sf.simulate_response(krn_delta, xdelta_train, snr)
        
        ytrain_signal = ytheta + ydelta 
        
        noise = (np.random.randn(data_duration*fs) + np.squeeze(mne.filter.filter_data(np.random.randn(1, data_duration*fs), sfreq=fs, \
                                                     l_freq=7, h_freq=12, fir_design='firwin', verbose=False))) * (np.std(ytrain_signal)/10**(snr/10))
        
        
        ytrain = ytrain_signal + noise
                
        # test data - same in all cases, except filtered
        ytheta, _ = sf.simulate_response(krn_theta, xtheta_test, snr)
        ydelta, _ = sf.simulate_response(krn_delta, xdelta_test, snr)
        
        ytest_signal = ytheta + ydelta 
        noise = (np.random.randn(data_duration*fs) + np.squeeze(mne.filter.filter_data(np.random.randn(1, data_duration*fs), sfreq=fs, \
                                                     l_freq=7, h_freq=12, fir_design='firwin', verbose=False))) * (np.std(ytest_signal)/10**(snr/10))
        
        ytest = ytest_signal + noise 
        
    # now we filter either in delta or in theta
    for filter_name,bp in zip(['delta', 'theta'], [delta_freqs, theta_freqs]):
        ytrain_filt = mne.filter.filter_data(ytrain, l_freq=bp[0], h_freq=bp[1], sfreq=fs, verbose=False)
        ytest_filt = mne.filter.filter_data(ytest, l_freq=bp[0], h_freq=bp[1], sfreq=fs, verbose=False)
        
        for train_predictor_name, train_predictor, test_predictor_name, test_predictor \
            in zip(['delta', 'theta', 'both'], [xdelta_train, xtheta_train, xboth_train],
                   ['delta', 'theta', 'both'], [xdelta_test, xtheta_test, xboth_test]):
            
            # we estimate the TRF model
            beta = sf.simulate_TRF(ytrain_filt, train_predictor, lags, alpha, fit_intercept=True)
            
            # and calc accuracy
            accuracy = sf.simulate_prediction(beta, ytest_filt, test_predictor, lags, prediction_type='corrcoef')
            
            tmp_df = pd.DataFrame(columns=['snr','filter_band','train_predictor', 'PearsonsR'])
            tmp_df['PearsonsR'] = accuracy
            tmp_df['snr'] = snr
            tmp_df['train_predictor'] = train_predictor_name
            tmp_df['filter_band'] = filter_name
        
            freq_df = pd.concat([freq_df, tmp_df], ignore_index=True)
            
# %% well now... let's plot it :D 
# increase from adding relevant predictor
freq_df_wide = freq_df.pivot(index=['snr', 'filter_band'], columns=['train_predictor'], values='PearsonsR')
freq_df_wide.reset_index(inplace=True)
freq_df_molten = pd.DataFrame(columns=['snr', 'filter_band', 'match', 'difference'])

for filter_band in ['delta', 'theta']: 
    if filter_band == 'delta':
        base = 'theta'
    elif filter_band == 'theta':
        base = 'delta'
    
    df = freq_df_wide.loc[freq_df_wide['filter_band'] == filter_band]
    df['match'] = df['both'] - df[base]
    df['mismatch'] = df['both'] - df[filter_band]

    print(np.mean(df['match'].values), np.mean(df['mismatch'].values))
    
    df = pd.melt(df, id_vars=['snr', 'filter_band'], value_vars=['match', 'mismatch'], var_name='match', value_name='difference')
    freq_df_molten = pd.concat([freq_df_molten, df], ignore_index=True)
    
# %%  plot it ovrall
fig,ax=plt.subplots(figsize=(5,3))

sns.scatterplot(x='filter_band', y='difference', hue='match', data=freq_df_molten, ax=ax,
            palette='crest')
ax.legend(frameon=False, title='Predictor-response', bbox_to_anchor=[1,1])

ax.set_xlabel('Frequency band of filtering')
ax.set_ylabel("$\Delta$ Pearson's R")

sns.despine()
plt.tight_layout()

fig.savefig('/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set1-figures/filter-response-match.svg')

# %% visualize per snr
fig,axes=plt.subplots(ncols=2, figsize=(7,3), sharey=True, sharex=True)

for ax,filter_band in zip(axes, ['delta', 'theta']):
    df = freq_df_molten.loc[freq_df_molten['filter_band'] == filter_band]
    
    df_clean = df.loc[df['snr']!='clean']
    df_clean['snr'] = df_clean['snr'].astype(float)
    
    sns.stripplot(x='match', y='difference', hue='snr', data=df_clean, ax=ax, palette='crest')
    sns.stripplot(x='match', y='difference', data=df.loc[df['snr']=='clean'],color='red', label='no noise', ax=ax,
                  size=5)
    ax.set_title(f'Filtered in {filter_band}')
    ax.set_xlabel('TRF')
    ax.set_ylabel("$\Delta$ Pearson's R")
    
axes[0].legend([],[], frameon=False)
axes[1].get_yaxis().set_visible(False)

handles,labels = axes[1].get_legend_handles_labels()

drop_list = ['-2.5', '-1.5', '-0.5', '0.5', '1.5', '2.5']

for handle, label in zip(handles, labels):
    if label in drop_list:
        handles.remove(handle)
        labels.remove(label)
        
labels.remove(labels[-1])
handles.remove(handles[-1])

legend = axes[1].legend(handles, labels, frameon=False, title='SNR', bbox_to_anchor=[1.5,1])

renderer = fig.canvas.get_renderer()

max_shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
for t in legend.get_texts():
    t.set_ha('right')  #
    temp_shift = max_shift - t.get_window_extent().width
    t.set_position((temp_shift, 0))
#axes[1].legend(frameon=False, title='SNR', bbox_to_anchor=[2,1])

sns.despine()
plt.tight_layout()
fig.savefig('/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set1-figures/filter-response-snr.svg')

