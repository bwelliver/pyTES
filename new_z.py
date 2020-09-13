import os
import time
from os.path import isabs, dirname, basename
import argparse

import numpy as np
import pandas as pan

import iv_processor
from numba import jit, prange
import multiprocessing
from joblib import Parallel, delayed

import ROOT as rt
from readROOT import readROOT
from writeROOT import writeROOT


def load_data(input_path):
    """Load data files for Z measurement."""
    chlist = 'ChList'
    channels = readROOT(input_path, tree=None, branches=None, method='single', tobject=chlist)
    channels = channels['data'][chlist]
    branches = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s']
    branches = branches + ['Waveform' + '{:03d}'.format(int(i)) for i in channels]
    print('Branches to be read are: {}'.format(branches))
    tree = 'data_tree'
    method = 'chain'
    data = readROOT(input_path, tree, branches, method)
    data = data['data']
    data['Channel'] = channels
    return data


def convert_to_numpy(data):
    """Convert root output data dictionary to 2d numpy arrays."""
    for key, value in data.items():
        if isinstance(value, dict):
            data[key], nsamples = iv_processor.convert_dict_to_ndarray(value)
    return data


@jit(nopython=True, parallel=True)
def lock_in_amplifier(signal, freq, fs, exfreq):
    """Compute lock-in amplifier components."""
    result = np.zeros(3)
    if np.remainder(freq, exfreq) == 0 and np.remainder(np.floor(freq/exfreq), 2) == 1:
        t = (2*np.pi*np.linspace(0, signal.shape[1], signal.shape[1])/fs) * freq  # check this
        #xv = np.mean(signal*np.sin(t))
        #yv = np.mean(signal*np.cos(t))
        # note: check: np.mean(np.dot(signal, sint))/signal.shape[1] might be equiv.
        xv = np.mean(np.dot(signal, np.sin(t)))/signal.shape[1]
        yv = np.mean(np.dot(signal, np.cos(t)))/signal.shape[1]
        mod = 2*np.sqrt(xv*xv + yv*yv)
        phase = np.arctan2(yv, xv)
        result[0] = mod
        result[1] = phase
        result[2] = freq
    return result


def lock_in(v_in, v_out, dt, number_samples, exfreq):
    """Try to do a lock-in amplifier."""
    T = number_samples * dt
    fs = np.round(1/dt)
    freqs = np.linspace(1/T, fs, np.int(number_samples))
    for idx in prange(freqs.size):
        freqs[idx] = np.round(freqs[idx])
    nyfreqs = freqs[0:int(freqs.size/2)]
    lock_vin_mod = np.zeros(freqs.size)
    lock_vin_phase = np.zeros(freqs.size)
    lock_vout_mod = np.zeros(freqs.size)
    lock_vout_phase = np.zeros(freqs.size)
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(lock_in_amplifier)(v_in, freq, fs) for freq in nyfreqs)
    for idx in range(nyfreqs.size):
        freq = nyfreqs[idx]
        if np.remainder(freq, exfreq) == 0 and np.remainder(np.floor(freq/exfreq), 2) == 1:
            # odd multiple
            modphase = lock_in_amplifier(v_in, freq, fs)
            lock_vin_mod[idx] = modphase[0]
            lock_vin_phase[idx] = modphase[1]
            modphase = lock_in_amplifier(v_out, freq, fs)
            lock_vout_mod[idx] = modphase[0]
            lock_vout_phase[idx] = modphase[1]
    return lock_vin_mod, lock_vin_phase, lock_vout_mod, lock_vout_phase


def main(filename):
    data = load_data(filename)
    data = convert_to_numpy(data)
    dt = data['SamplingWidth_s'][0]
    number_samples = data['NumberOfSamples'][0]
    exfreq = 17
    locked = lock_in(data['Waveform000'], data['Waveform001'], dt, number_samples, exfreq)
    return locked
