#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:33:52 2024

@author: sopsla
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import simfun as sf
from pyeeg.utils import lag_span

# %% directory
resultsdir = '/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set3-figures/log-800ms'

# dict of types of tests
testdict = {'node':0,
            'node + surprisal':1,
            'node split':2,
            'node split + surprisal':3,
            'node random split':4,
            'node random split + surprisal':5,
            'interaction':6,
            'node + surprisal + interaction':7}

# influence of surprisal (multiplication)
mult_values = np.linspace(0, 20, num=20)/10 # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# %% load the files of the reconstruction accuracies
it = 10
allfiles = os.listdir(resultsdir)
rafiles = [file for file in allfiles if file.startswith(f'systematic_{str(it)}it')]

# open the files
radata = []

for rafile in rafiles:
    with open(f'{resultsdir}/{rafile}', 'rb') as f:
        data = pickle.load(f)
        
    radata.append(data)
    
ovfiles = [file for file in allfiles if file.startswith(f'other_variable_{str(it)}it')]

ovdata = []

for ovfile in ovfiles:
    with open(f'{resultsdir}/{ovfile}', 'rb') as f:
        data = pickle.load(f)
    
    ovdata.append(data)

# shape of data: iterations, tests, multiplication values    
systematic = np.concatenate(radata, 0)
other_variable = np.concatenate(ovdata, 0)

# %% let's plot them all
fig,axes = plt.subplots(ncols=2, figsize=(9,4),sharex=True,sharey=True)
conds = testdict.keys() #['node','node + surprisal', 'node split', \
        # 'node split + surprisal', 'node random split',\
        # 'node random split + surprisal',
        # 'node + surprisal + interaction']

for ax, data in zip(axes, [systematic, other_variable]):
    ax.margins(x=0)
    cond_ix = [testdict[c] for c in conds]
    to_viz = data[:,cond_ix,:]
    means = np.mean(to_viz, 0)
    std = np.std(to_viz,0)
        
    ax.plot(mult_values,means.T)
            
    for line in list(range(0,to_viz.shape[1])):
        ax.fill_between(mult_values, means.T[:,line]-std.T[:,line], 
                        means.T[:,line]+std.T[:,line], alpha=0.2, zorder=-1)
        
    ax.set_xlabel('Influence of surprisal (multiplication)')
    ax.set_ylabel('R-value')
        
axes[0].legend([c[0].upper() + c[1:] for c in conds], frameon=False)
#axes[1].legend([c[0].upper() + c[1:] for c in conds], frameon=False,
 #              bbox_to_anchor=[1.0,1]) # , 'Interaction', 'Node + surprisal + interaction'
axes[1].get_yaxis().set_visible(False)

axes[0].set_title('Node * surprisal')
axes[1].set_title('Node * third variable')

plt.tight_layout()
sns.despine()
fig.savefig(f'{resultsdir}/comparison_{systematic.shape[0]}it{"_".join(conds)}.svg')

# %% let's do some stats between these for multiplication = 1
#mult = mult_values.index(0.5)
comp1 = 'node split'

fig,axes=plt.subplots(ncols=3, figsize=(9,3), sharey=True)

for ix,comp2 in enumerate(['node', 'node random split', 'node + surprisal + interaction']):
    ax = axes[ix]
    ax.margins(x=0)
    
    for name, data in zip(['Systematic', 'Random'], [systematic, other_variable]):
        statslist = []
        
        print(name)
        print(f'{comp2} vs {comp1}')
        
        idx1 = testdict[comp1]
        idx2 = testdict[comp2]
        
        for multi,mult in enumerate(mult_values):
        
            d1 = data[:,idx1,multi]
            d2 = data[:,idx2,multi]
            
            comp = stats.ttest_rel(d1,d2)
            print(comp)
            
            statslist.append(comp.statistic)
        
        ax.plot(mult_values, statslist)   
        if ix == 2:
            ax.legend(['Surprisal', 'Third variable'], frameon=False,bbox_to_anchor=[2,1])
             
                    
    ax.set_xlabel('Multiplication of surprisal')
    if ix == 0:
        ax.set_ylabel('T-value')
    else:
        ax.get_yaxis().set_visible(False)
    
    ax.set_title(f'{comp1}\nvs\n{comp2}')
    
     
for ax in axes:
    ax.axhline(y=1.96, color='red', ls='--', alpha=0.5, lw=0.8)
    ax.axhline(y=-1.96, color='red', ls='--', alpha=0.5, lw=0.8)

sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/{comp1}-vs-models-tvals.svg')

# %% the effect of surprisal
comp1 = 'node'

fig,axes=plt.subplots(ncols=2, figsize=(9,3), sharey=True)

for ix,comp2 in enumerate(['node + surprisal', 'node + surprisal + interaction']):
    ax = axes[ix]
    ax.margins(x=0)
    
    for name, data in zip(['Systematic', 'Random'], [systematic, other_variable]):
        statslist = []
        
        print(name)
        print(f'{comp2} vs {comp1}')
        
        idx1 = testdict[comp1]
        idx2 = testdict[comp2]
        
        for multi,mult in enumerate(mult_values):
        
            d1 = data[:,idx1,multi]
            d2 = data[:,idx2,multi]
            
            comp = stats.ttest_rel(d1,d2)
            print(comp)
            
            statslist.append(comp.statistic)
        
        ax.plot(mult_values, statslist)   
        if ix == len(axes)-1:
            ax.legend(['Surprisal', 'Third variable'], frameon=False,bbox_to_anchor=[2,1])
             
                    
    ax.set_xlabel('Multiplication of surprisal')
    if ix == 0:
        ax.set_ylabel('T-value')
    else:
        ax.get_yaxis().set_visible(False)
    
    ax.set_title(f'{comp1}\nvs\n{comp2}')
    
     
for ax in axes:
    ax.axhline(y=1.96, color='red', ls='--', alpha=0.5, lw=0.8)
    ax.axhline(y=-1.96, color='red', ls='--', alpha=0.5, lw=0.8)

sns.despine()
plt.tight_layout()

#fig.savefig(f'{resultsdir}/{comp1}-vs-models-surprisal-tvals.svg')


# %% the betas
# load them - model 1:1 for node split, also show node random split
betafiles = [file for file in allfiles if file.startswith(f'systematic_betas_{str(it)}it')]
ovbetafiles = [file for file in allfiles if file.startswith(f'other_variable_betas_{str(it)}it')]

# open the files
subetas = []

for betafile in betafiles:
    with open(f'{resultsdir}/{betafile}', 'rb') as f:
        data = pickle.load(f)
        
    subetas.append(data)
    
ovbetas = []

for ovbetafile in ovbetafiles:
    with open(f'{resultsdir}/{ovbetafile}', 'rb') as f:
        data = pickle.load(f)
    
    ovbetas.append(data)
    
# %% flatten the beta files
subetas_flat = [dictfile for listfile in subetas for dictfile in listfile]
subetas_dict = dict.fromkeys(subetas_flat[0])

for key in subetas_dict.keys():
    
    subetas_dict[key] = []
    
    for iteration in subetas_flat:
        subetas_dict[key].append(iteration[key])

# %% plot the difference with sd
sfreq=120
lags = lag_span(-0.2, 1.0, sfreq)

# %%
fig,axes = plt.subplots(nrows=5, figsize=(5,10))

for ix, multfact in enumerate(mult_values):
    if ix%4 != 0:
        continue

    if ix == 0:
        ax = axes[ix]
        
    else:
        ax = axes[int(ix/4)]
        
    split = [b[2] for b in subetas_dict[ix]]
    split = np.asarray(split)
    high = split[:,:,0]
    low = split[:,:,1]
    
    avg_high = np.mean(high, axis=0)
    std_high = np.std(high,axis=0)
    avg_low = np.mean(low, axis=0)
    std_low = np.std(low, axis=0)

    ax.margins(x=0)
    ax.plot(lags/sfreq, avg_high)
    ax.plot(lags/sfreq, avg_low)
    
    ax.set_title(f'Multiplication factor {np.round(multfact, decimals=1)}')    
    
    ax.fill_between(lags/sfreq, avg_high-std_high, avg_high+std_high, alpha=0.6) 
    ax.fill_between(lags/sfreq, avg_low-std_low, avg_low+std_low, alpha=0.6) 
    
    if ix < 16:
        ax.get_xaxis().set_visible(False)
    
    ax.set_ylabel('Coeff. (a.u.)')
    ax.set_xlabel('Time (s)')

ax.legend(['Node count/high surprisal', 'Node count/low surprisal'], frameon=False)
sns.despine()
plt.tight_layout()
    
fig.savefig(f'{resultsdir}/nodesplit_betas_5-factors.svg')

# %% surprisal & interaction hallucinated responses
surprisal_betas = np.asarray([b[1] for b in subetas_dict[19]])
mean_surprisal_betas = np.mean(surprisal_betas, axis=0)
std_surprisal_betas = np.std(surprisal_betas,axis=0)

fig,axes=plt.subplots(ncols=3,figsize=(9,3), gridspec_kw={'width_ratios': [1,1,2]})

# on the first column we plot the simple R-values for node and node + surprisal
conds = ['node','node + surprisal']
ax = axes[0]
ax.margins(x=0)
cond_ix = [testdict[c] for c in conds]
to_viz = systematic[:,cond_ix,:]
means = np.mean(to_viz, 0)
std = np.std(to_viz,0)
    
ax.plot(mult_values,means.T)
        
for line in list(range(0,to_viz.shape[1])):
    ax.fill_between(mult_values, means.T[:,line]-std.T[:,line], 
                    means.T[:,line]+std.T[:,line], alpha=0.2, zorder=-1)
    
ax.set_xlabel('Influence of surprisal (multiplication)')
ax.set_ylabel('R-value')
    
axes[0].legend([c[0].upper() + c[1:] for c in conds], frameon=False,
               loc='lower center')
axes[0].set_title('Reconstruction accuracy')

# on the second column we plot the difference between 'node' and 'node + surprisal'
# % the effect of surprisal
comp1 = 'node + surprisal'
comp2 = 'node'

ax = axes[1]
ax.margins(x=0)

for name, data in zip(['Systematic', 'Random'], [systematic, other_variable]):
    statslist = []
    
    print(name)
    print(f'{comp2} vs {comp1}')
    
    idx1 = testdict[comp1]
    idx2 = testdict[comp2]
    
    for multi,mult in enumerate(mult_values):
    
        d1 = data[:,idx1,multi]
        d2 = data[:,idx2,multi]
        
        comp = stats.ttest_rel(d1,d2)
        print(comp)
        
        statslist.append(comp.statistic)
    
    ax.plot(mult_values, statslist)   
    
ax.legend(['Surprisal', 'Third variable'], frameon=False)                
ax.set_xlabel('Multiplication of surprisal')
ax.set_ylabel('T-value')
ax.set_title('Statistical comparison')
    
ax.axhline(y=1.96, color='red', ls='--', alpha=0.5, lw=0.8)
ax.axhline(y=-1.96, color='red', ls='--', alpha=0.5, lw=0.8)

# second column: the response to surprisal
ax = axes[2]
ax.margins(x=0)
ax.plot(lags/sfreq, mean_surprisal_betas)
ax.legend(['Node', 'Surprisal'], frameon=False)

ax.fill_between(lags/sfreq, mean_surprisal_betas[:,0]-std_surprisal_betas[:,0], mean_surprisal_betas[:,0]+std_surprisal_betas[:,0], alpha=0.6) 
ax.fill_between(lags/sfreq, mean_surprisal_betas[:,1]-std_surprisal_betas[:,1], mean_surprisal_betas[:,1]+std_surprisal_betas[:,1], alpha=0.6) 
ax.set_ylabel('Coeff. (a.u.)')
ax.set_xlabel('Time (s)')
ax.set_title('TRF waveforms\nNode + surprisal')

sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/effect_of_surprisal.svg')

# %% let's look at whether there is an hallucinated response in split models
fig,ax=plt.subplots(ncols=2, figsize=(9,3))

for axi,split_type in zip(ax,[3,5]):
    bets = np.asarray([b[split_type] for b in subetas_dict[19]])
    mns = np.mean(bets, axis=0)
    stds= np.std(bets,axis=0)

    axi.plot(lags/sfreq, mns)
    axi.legend(['Node 1', 'Node 2', 'Surprisal'], frameon=False)

    axi.fill_between(lags/sfreq, mns[:,0]-stds[:,0], mns[:,0]+stds[:,0], alpha=0.6) 
    axi.fill_between(lags/sfreq, mns[:,1]-stds[:,1], mns[:,1]+stds[:,1], alpha=0.6) 
    axi.fill_between(lags/sfreq, mns[:,2]-stds[:,2], mns[:,2]+stds[:,2], alpha=0.6) 
     
    axi.set_ylabel('Coeff. (a.u.)')
    axi.set_xlabel('Time (s)')
    axi.set_title('TRF waveforms\nNode + surprisal')

# there is a response

# %% interaction term
fig,axes=plt.subplots(ncols=3,figsize=(9,3), gridspec_kw={'width_ratios': [1,1,2]})

# on the first column we plot the simple R-values for node and node + surprisal
conds = ['node','interaction', 'node + surprisal + interaction']
cond_abbrev = ['node', 'int', 'n+s+int']
ax = axes[0]
ax.margins(x=0)
cond_ix = [testdict[c] for c in conds]
to_viz = systematic[:,cond_ix,:]
means = np.mean(to_viz, 0)
std = np.std(to_viz,0)

colors = sns.color_palette('tab10')
           
for line in list(range(0,to_viz.shape[1])):
    ax.plot(mult_values,means.T[:,line], 
            color=colors[cond_ix[line]],
            label=conds[line])
    
ax.legend([c[0].upper() + c[1:] for c in cond_abbrev], frameon=False,
          bbox_to_anchor = [0.5,1])

for line in list(range(0,to_viz.shape[1])): 
    ax.fill_between(mult_values, means.T[:,line]-std.T[:,line], 
                    means.T[:,line]+std.T[:,line], alpha=0.2, zorder=-1,
                    color=colors[cond_ix[line]])
    
ax.set_xlabel('Influence of surprisal (multiplication)')
ax.set_ylabel('R-value')
    

ax.set_title('Reconstruction accuracy')

# on the second column we plot the difference between 'node' and 'node + surprisal'
# % the effect of surprisal
comp1 = 'interaction'
comp2 = 'node'

ax = axes[1]
ax.margins(x=0)

for name, data in zip(['Systematic', 'Random'], [systematic, other_variable]):
    statslist = []
    
    print(name)
    print(f'{comp2} vs {comp1}')
    
    idx1 = testdict[comp1]
    idx2 = testdict[comp2]
    
    for multi,mult in enumerate(mult_values):
    
        d1 = data[:,idx1,multi]
        d2 = data[:,idx2,multi]
        
        comp = stats.ttest_rel(d1,d2)
        print(comp)
        
        statslist.append(comp.statistic)
    
    ax.plot(mult_values, statslist)   
    
ax.legend(['Surprisal', 'Third variable'], frameon=False)                
ax.set_xlabel('Multiplication of surprisal')
ax.set_ylabel('T-value')
ax.set_title(f'Statistical comparison\n{comp1} vs {comp2}')
    
ax.axhline(y=1.96, color='red', ls='--', alpha=0.5, lw=0.8)
ax.axhline(y=-1.96, color='red', ls='--', alpha=0.5, lw=0.8)

# third column: the response to interaction
mdl = 'node + surprisal + interaction'
testix = testdict[mdl]
bets = np.asarray([b[testix] for b in subetas_dict[19]])
mns = np.mean(bets, axis=0)
stds = np.std(bets,axis=0)

ax = axes[2]
ax.margins(x=0)
ax.plot(lags/sfreq, mns)
ax.legend(['Node', 'Surprisal', 'Interaction'], frameon=False)

for trf_i in list(range(mns.shape[1])):

    ax.fill_between(lags/sfreq, mns[:,trf_i]-stds[:,0], mns[:,trf_i]+stds[:,0], alpha=0.6) 
    
ax.set_ylabel('Coeff. (a.u.)')
ax.set_xlabel('Time (s)')
ax.set_title('TRF waveforms\nNode + surprisal')

sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/interaction_effects.svg')

# %% plot of example data
# this is the same code as used in the actual simulations
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
kernel_width = 0.5 if resultsdir.endswith('500ms') else 0.8 # 500 milliseconds of response
mult = 2.0 # multiply surprisal by two 
kernel = sf.impulse_response(sfreq, kernel_width)

systematic = np.zeros((n_iterations, 8, len(mult_values)))
other_variable = systematic.copy()

# get values for the split predictor
brackets = 2 * np.random.randn(1, len(tmp_samp)) + 5
brackets = np.abs(brackets)
surprisal = np.random.lognormal(mean=2.3, sigma=0.41, size=(1,len(tmp_samp))) 

some_other_variable = np.random.lognormal(mean=2.3, sigma=0.41,size=(1,len(tmp_samp)))
    
# data - with linear moveable kernel, adjusting the influence of surprisal   
for var,varname in zip([surprisal, some_other_variable], 
                             ['systematic', 'random']):
    # surprisal kernels
    # in the case of some_other_variable we will use this to generate the data
    surprisal_kernels = [moveable_kernel(surp, kernel,multiplication=mult) for surp in var[0,:]]
    data_list = []
    
    for i,(br,sk) in enumerate(zip(brackets[0,:], surprisal_kernels)):
        response = scipy.signal.convolve(np.expand_dims(br,0), sk)
        data_list.append(response)

data_list_long = [np.pad(d, (0, 300-len(d)), 'constant', constant_values=(0)) for d in data_list]
data_frame = pd.DataFrame(data_list_long, index=np.squeeze(var))
data_frame = data_frame.sort_index(ascending=False)

# %% here we plot the data
fig,axes = plt.subplots(ncols=2, figsize=(9,4))

colormap = sns.color_palette("Spectral", n_colors=1000)
    
for i,response in enumerate(data_list):    
    axes[0].plot(response,color=colormap[i])

axes[0].set_xlabel('Samples (120 Hz)')
axes[0].set_ylabel('Coeff. (a.u.)')
axes[0].set_title('Responses modeled')

sns.heatmap(data_frame.iloc[:,:150], ax=axes[1], 
            cmap=sns.diverging_palette(210, 15, l=40, s=100, as_cmap=True), vmin=-10, vmax=10)
plt.locator_params(axis='both', nbins=10) 

ticklocs = plt.yticks()[0]
ticklabs = plt.yticks()[1]
axes[1].set_yticks(ticklocs, labels=['32.9', '15.2', '12.4', '10.6', '9.1', '7.7', '6.3'])


plt.xticks(ticks=axes[1].get_xticks(),rotation='horizontal')

axes[1].set_title(f'{"".join([varname[0].upper(),varname[1:]])} time-shift, {mult}x')
axes[1].set_ylabel('Surprisal (bits)')
axes[1].set_xlabel('Samples from "word onsets"')

sns.despine()
plt.tight_layout()
plt.show()

#fig.savefig('/home/lacnsg/sopsla/Documents/code-repositories/trf-simulations/set3-figures/log-500ms/example_data.svg')
