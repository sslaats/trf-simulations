#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:27:31 2022

@author: sopsla
"""
import numpy as np
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import mne
from pyeeg.utils import lag_matrix

def get_max_power_frequency(data, n_samples, fs):
    fourier_transform = np.abs(scipy.fft.rfft(data))
    power_spectrum = np.square(fourier_transform)
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    return frequency[list(power_spectrum).index(max(power_spectrum))]

def plot_powerspectrum(data, n_samples, fs, ax=None, logscale=True):
    fourier_transform = np.abs(scipy.fft.rfft(data))
    power_spectrum = np.square(fourier_transform)
    
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    if ax==None:
        plt.subplots()
        plt.plot(frequency[:int(round(len(frequency)/2))], power_spectrum[:int(round(len(power_spectrum)/2))])
    else:
        ax.plot(frequency[:int(round(len(frequency)/2))], power_spectrum[:int(round(len(power_spectrum)/2))])
        
    if logscale:
        plt.yscale('log')
    else:
        plt.yscale('linear')
        
    plt.ylabel(" Magnitude (energy)") #"Power ({0})".format('log' if logscale else 'linear'))
    plt.xlabel("f (Hz)")
    
def generate_kernel(loc, width, type='gaussian', tmin=-0.1, tmax=0.7, freqs=[3., 6.], fs=120):
    """
    Generate the kernel response (aka impulse response of the LTI system, i.e. the ground truth TRF).
    The kernel can be of two kind of shapes:
        - type == 'gaussian' will produce a bell/gaussian shaped kernel
        - type == 'sines' will produce sum of sines with respective frequencies defined by `freqs`, multiplied by a hanning window located at `loc` and spanning `width`
    
    Returns
    -------
    tlags : ndarray
        Time lags vector
    kernel : ndarray
        Same shape as tlags, the kernel coefficients.
    """
    dt = 1/fs
    tlags = np.arange(tmin, tmax, dt)
    lags = np.round(tlags*fs).astype(int)
    iscale = int(fs * width)
    iloc = np.argmin(abs(tlags - loc)) - iscale//2
    ker = np.zeros_like(tlags)
    indices = [i for i in (np.arange(iscale) + iloc) if i<len(ker)]
    ker[indices] = np.hanning(iscale)[:len(indices)]
    if type=='gaussian':
        return tlags, ker
    elif type=='sines':
        # Use two sines: 3Hz and 6Hz
        freqs = np.atleast_1d(freqs)
        sines = np.zeros_like(tlags)
        for f in freqs:
            sines += np.sin(2*np.pi*f*tlags + np.random.rand()*np.pi)
        return tlags, ker * sines
    else:
        raise ValueError("type must be 'gaussian' or 'sines'")
    
def impulse_response(sfreq, kernel_width, superimposed=False, krn_type='sine', freq=6):
    """
    sfreq : int (Hz)
    kernel_width : float (s)

    """
    
    n_samples = int(kernel_width * sfreq)
    hw = np.hanning(n_samples)
    half_index = int(n_samples/2)
    hw[half_index:] = hw[half_index:] / 2
    
    if krn_type == 'sine':
        krn = np.squeeze(hw) * np.sin(freq * np.pi * 
                                      np.array(range(0, n_samples)) / n_samples)
    
        if superimposed:
            krn = krn * np.sin(freq*2*np.pi*np.array(range(0, n_samples)) / n_samples)
            
    elif krn_type == 'gaussian':
        raise NotImplementedError()
    
    krn = np.concatenate([np.zeros((24)), krn])
    return krn

def simulate_TRF(data, predictors, lags, alpha, fit_intercept=False):
    """
    Quick trf function for simulation purposes
    """
    
    y = data.T
    x = predictors
    
    # lag the predictor matrix
    X = lag_matrix(x, lags)
    
    nan_rows = np.isnan(X.mean(1))
    y = y[~nan_rows]
    X = X[~nan_rows]
        
    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0],1)), X])
        
    XtX = X.T @ X
    Xty = X.T @ y
    
    beta = np.linalg.inv(XtX + np.eye(X.shape[1])*alpha) @ Xty

    if fit_intercept:
        beta = beta[1:]

    if len(x.shape) > 1:
        beta = beta.reshape((len(lags), np.shape(x)[-1]))
    
    return beta

def simulate_prediction(beta, data, predictors, lags, prediction_type='corrcoef'):
    """
    Quick predict function for simulation purposes
    """
    y = data
    x = predictors
    
    if len(beta.shape) > 1:
        beta = beta.reshape((len(lags)*np.shape(beta)[-1]))
    X = lag_matrix(x, lags)
    nan_rows = np.isnan(X.mean(1))
    X = X[~nan_rows]
    y = y[~nan_rows]
    
    yhat = X @ beta
    
    if prediction_type == 'corrcoef':
        R = np.diag(np.corrcoef(yhat, y, rowvar=False), k=1)
    elif prediction_type == 'variance':
        R = np.sum(yhat**2) / np.sum(y**2)
    elif prediction_type == 'corrunbiased':
        R = np.diag(np.corrcoef(yhat, y, rowvar=False), k=1) * len(yhat) / len(yhat!=0)
        
    return R

def simulate_permutation(beta_1, beta_2, no_subjects):
    ttest = lambda x, y: mne.stats.cluster_level.ttest_1samp_no_p(x-y)
    
    threshold = stats.t.ppf(1-0.5/2, df=no_subjects-1)
    
    cluster_stats = mne.stats.permutation_cluster_test([beta_1, beta_2],
                                                          n_permutations=1024,
                                                          threshold=threshold,
                                                          tail = 1,
                                                          stat_fun=ttest,
                                                          max_step=1,
                                                          n_jobs=1,
                                                          buffer_size=None,
                                                          adjacency=None,
                                                      verbose=False)
    
    print("There are %d significant clusters\n"%(cluster_stats[2]<0.05).sum())
    return cluster_stats

def simulate_stimulus(isi, n_stim, data_duration, fs=120, smoothing=False):
    """
    Generates two impulse train time series: one with constant values and one with normally distributed
    values at each spike onset.
    Each impulse is seperated by `isi` seconds.
    If `smoothing` is `True`, the impulse trains are convolved by a smoothing kernel (hanning window) of 200ms.

    Returns
    -------
    t : ndarray
        Time vector.
    comb : ndarray
        Impulse train with constant value of ones, same shape as `t`.
    valued : ndarray
        Impulse train with normally distributed values at onsets, same shape as `t`.
    tend : float
        Last onset time.
    """
    t = np.arange(0, data_duration, 1/fs)
    tspikes = np.arange(1, n_stim+1,) * isi
    comb = np.zeros_like(t)
    comb[np.asarray(tspikes*fs, dtype=int)] = 1
    onset_values = np.random.rand(n_stim,)
    valued = comb.copy()
    valued[np.asarray(tspikes*fs, dtype=int)] = onset_values
    
    # Blur
    if smoothing:
        comb = np.convolve(comb, np.hanning(int(0.2*fs)), 'same')
        valued = np.convolve(valued, np.hanning(int(0.2*fs)), 'same')
    
    return t, comb, valued, tspikes[-1]

def simulate_response(kernel, stim, snr, tmax=0.8, tmin=-0.2, fs=120):
    """
    Generate the noisy and noiseless responses given a stimulus and a kernel.
    The model is simply a convolution of the stimulus with the kernel coefficients.
    The noisy output adds some white measurment noise to the convolved response.

    Parameters
    ----------
    kernel : ndarray (nlags, 1)
        Kernel coefficients (see `generate_kernel`)
    stim : ndarray (ntimes, npreds)
        Stimulus representation (see `generate_stim`)
    snr : float
        SNR for the noisy output
    tmin, tmax : float, float
        How the kernel aligns with respect to the stimulus during convolution.
    fs : float
        Sampling rate.

    Returns
    -------
    yresp : ndarray (ntimes, 1)
        Clean output from the convolution between `kernel` and `stim`.
    y : ndarray (ntimes, 1)
        Noisy output (yresp + noise), with noise scaled to have a broadband SNR of `snr`.
    """
    yresp = np.convolve(stim, np.pad(kernel, (int(fs*(abs(max(tmax+tmin, 0))))-1,
                                           int(fs*(abs(min(tmax+tmin, 0))))), mode='constant'), 'same')
    noise = np.random.randn(*yresp.shape)
    y =  yresp + noise * (np.std(yresp)/10**(snr/10))
    return yresp, y

 
def spectrum_noise(spectrum_func, samples=1024, rate=44100):
    """ 
    make noise with a certain spectral density
    code from https://www.socsci.ru.nl/wilberth/python/noise.html
    """
    freqs = np.fft.rfftfreq(samples, 1.0/rate)            # real-fft frequencies (not the negative ones)
    spectrum = np.zeros_like(freqs, dtype='complex')      # make complex numbers for spectrum
    spectrum[1:] = spectrum_func(freqs[1:])               # get spectrum amplitude for all frequencies except f=0
    phases = np.random.uniform(0, 2*np.pi, len(freqs)-1)  # random phases for all frequencies except f=0
    spectrum[1:] *= np.exp(1j*phases)                     # apply random phases
    noise = np.fft.irfft(spectrum)                        # return the reverse fourier transform
    noise = np.pad(noise, (0, samples - len(noise)), 'constant') # add zero for odd number of input samples
 
    return noise
 
def pink_spectrum(f, f_min=0, f_max=np.inf, att=np.log10(2.0)*10):
    """
    Define a pink (1/f) spectrum
        f     = array of frequencies
        f_min = minimum frequency for band pass
        f_max = maximum frequency for band pass
        att   = attenuation per factor two in frequency in decibel.
                Default is such that a factor two in frequency increase gives a factor two in power attenuation.
   
   code from https://www.socsci.ru.nl/wilberth/python/noise.html 
   """
    # numbers in the equation below explained:
    #  0.5: take the square root of the power spectrum so that we get an amplitude (field) spectrum 
    # 10.0: convert attenuation from decibel to bel
    #  2.0: frequency factor for which the attenuation is given (octave)
    s = f**-( 0.5 * (att/10.0) / np.log10(2.0) )  # apply attenuation
    s[np.logical_or(f < f_min, f > f_max)] = 0    # apply band pass
    return s
    