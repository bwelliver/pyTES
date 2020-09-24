import os
import time
from os.path import isabs, dirname, basename
import argparse

import numpy as np
import pandas as pan
import h5py

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


@jit(nopython=True)
def lock_in_amplifier(signal, freq, fs):
    """Compute lock-in amplifier components."""
    result = np.zeros(2)
    # This is really 2*pi*f * t where t spans the duration of time the signal is
    t = (2*np.pi*np.linspace(0, signal.shape[1], signal.shape[1])/fs) * freq  # check this
    #xv = np.mean(signal*np.sin(t))
    #yv = np.mean(signal*np.cos(t))
    # note: check: np.mean(np.dot(signal, sint))/signal.shape[1] might be equiv.
    xv = np.mean(np.dot(signal, np.sin(t)))/signal.shape[1]
    yv = np.mean(np.dot(signal, np.cos(t)))/signal.shape[1]
    result[0] = 2*np.sqrt(xv*xv + yv*yv)
    result[1] = np.arctan2(yv, xv)
    #result[2] = freq
    return result


@jit(nopython=True, parallel=True)
def lock_in(v_in, v_out, dt, number_samples, exfreq):
    """Try to do a lock-in amplifier."""
    T = number_samples * dt  # the total time of the window
    fs = np.round(1/dt)  # the sampling rate given by 1/sample_width
    # Construct the vector of frequencies we sample at
    freqs = np.linspace(1/T, fs, np.int(number_samples))
    # We need to round each frequency to whole values?
    for idx in prange(freqs.size):
        freqs[idx] = np.round(freqs[idx])
    nyfreqs = freqs[0:int(freqs.size/2)]
    lockin_values = np.zeros((freqs.size, 4))  # Nx4 array
    
    lock_vin_mod = np.zeros(freqs.size)
    lock_vin_phase = np.zeros(freqs.size)
    lock_vout_mod = np.zeros(freqs.size)
    lock_vout_phase = np.zeros(freqs.size)
    #num_cores = multiprocessing.cpu_count()
    #results = Parallel(n_jobs=num_cores)(delayed(lock_in_amplifier)(v_in, freq, fs, exfreq) for freq in nyfreqs)
    for idx in prange(nyfreqs.size):
        freq = nyfreqs[idx]
        if np.remainder(freq, exfreq) == 0 and np.remainder(np.floor(freq/exfreq), 2) == 1:
            # odd multiple
            modphase = lock_in_amplifier(v_in, freq, fs)
            lock_vin_mod[idx] = modphase[0]
            lock_vin_phase[idx] = modphase[1]
            modphase = lock_in_amplifier(v_out, freq, fs)
            lock_vout_mod[idx] = modphase[0]
            lock_vout_phase[idx] = modphase[1]
    return lock_vin_mod, lock_vin_phase, lock_vout_mod, lock_vout_phase, freqs


def compute_zmeas(fvin, fvout, squid_params, input_gain=1, output_gain=1):
    """Compute the measured Z for a given set of observations."""
    Rsh = squid_params['Rsh']
    M = squid_params['M']
    Rfb = squid_params['Rfb']
    Zbias = squid_params['Zbias']
    z_meas = (M*Rsh*Rfb*(fvin/input_gain))/(Zbias*(fvout/output_gain))
    return z_meas


def get_lockin_averages(filename):
    data = load_data(filename)
    data = convert_to_numpy(data)
    dt = data['SamplingWidth_s'][0]
    number_samples = data['NumberOfSamples'][0]
    exfreq = 17
    locked = lock_in(data['Waveform000'], data['Waveform001'], dt, number_samples, exfreq)
    return locked


def transform_to_rect(polar_results):
    """Transform to rectangular parameters."""
    fvin = polar_results[0]*(np.cos(polar_results[1]) + 1j*np.sin(polar_results[1]))
    fvout = polar_results[2]*(np.cos(polar_results[3]) + 1j*np.sin(polar_results[3]))
    return fvin, fvout


def main():
    normal_file = '/Users/bwelliver/tmp/SlowDigit_Norm_250uA_1000nA_Square/root//SlowDigitNorm250uA1000nASquare_1.root'
    sc_file = '/Users/bwelliver/tmp/SlowDigit_SC_0uA_1uA_Suqare/root//SlowDigitSC0uA1uASquare_1.root'
    print('Starting normal lock-in')
    norm_locked = get_lockin_averages(normal_file)
    print('Starting sc lock-in')
    sc_locked = get_lockin_averages(sc_file)
    print('Computing other quantities now...')
    norm_fvin, norm_fvout = transform_to_rect(norm_locked)
    sc_fvin, sc_fvout = transform_to_rect(sc_locked)
    exfreq = 17
    freqs = norm_locked[4]
    cut = np.logical_and(np.remainder(freqs, exfreq) == 0, np.remainder(np.floor(freqs/exfreq), 2) == 1)
    squid_params = {'Rsh': 20.2e-3, 'M': -1.3145, 'Rfb': 100000, 'Zbias': 10200}
    norm_zmeas = compute_zmeas(norm_fvin, norm_fvout, squid_params, input_gain=1)
    sc_zmeas = compute_zmeas(sc_fvin, sc_fvout, squid_params, input_gain=1)
    return norm_fvin, norm_fvout, sc_fvin, sc_fvout, freqs, cut, norm_zmeas, sc_zmeas
