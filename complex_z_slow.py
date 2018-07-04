import argparse
import glob

import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt
from scipy import fftpack
from scipy.signal import hann
from scipy import signal

from readROOT import readROOT

eps = np.finfo(float).eps
ln = np.log
mp.rcParams['agg.path.chunksize'] = 10000


def gen_plot_bar(x, y, xlab, ylab, title, fName, dx=1, logx='log', logy='log'):
    '''Create bar plot'''
    N = y.size
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.bar(x, y, align='center', width=dx)
    ax.xticks([i*8*dx for i in range(N/8)]+[N*dx-dx/2])
    ax.xlim(-dx/2, N*dx-dx/2)
    ax.set_title(title)
    ax.grid(True)
    fig.savefig(fName, dpi=100)
    plt.close('all')
    return None


def gen_plot_line2(x, y, xlab, ylab, title, fName, logx='log', logy='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(16, 9))
    ax = fig2.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', linewidth=1)
    ax.set_xscale(logx)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(logy)
    ax.set_title(title)
    ax.grid(True)
    fig2.savefig(fName, dpi=100)
    plt.close('all')
    return None


def gen_plot_line(x, y, xlab, ylab, title, fName, logx='log', logy='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(16, 9))
    ax = fig2.add_subplot(111)
    ax.plot(x, y, linestyle='-', linewidth=0.5)
    ax.set_xscale(logx)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(logy)
    ax.set_title(title)
    ax.grid(True)
    fig2.savefig(fName, dpi=200)
    plt.close('all')
    return None


def process_waveform(dWaveform, procType='mean'):
    '''Take an input waveform dictionary and process it by collapse to 1 point
    An incoming waveform is a dictionary with keys = event number and values = waveform of duration 1s
    This function will collapse the waveform to 1 value based on procType and return it as a numpy array
    We will also return a bonus array of the rms for each point too
    '''
    # We can have events missing so better to make the vectors equal to the max key
    npWaveform = np.empty(len(dWaveform.keys()))
    npWaveformRMS = np.empty(len(dWaveform.keys()))
    if procType == 'mean':
        for ev, waveform in dWaveform.items():
            npWaveform[ev] = np.mean(waveform)
            npWaveformRMS[ev] = np.std(waveform)
    elif procType == 'median':
        for ev, waveform in dWaveform.items():
            npWaveform[ev] = np.median(waveform)
            q75, q25 = np.percentile(waveform, [75, 25])
            npWaveformRMS[ev] = q75 - q25
    else:
        raise Exception('Please enter mean or median for process')
    return npWaveform, npWaveformRMS


def concatenate_waveform(waveform_array):
    '''Concatenate a bunch of wave form dictionaries to a single numpy array'''
    # This might use too much memory
    data = np.concatenate([v for k,v in sorted(waveform_array.items())])
    return data


def time_to_freq(t):
    '''Generate an appropriate frequency vector given a time vector
    N = number of samples
    dt = sample spacing
    '''
    N = t.size
    dt = t[-1] - t[-2]
    #f = np.linspace(0.0, 1.0/(2.0*dt), N//2)
    print('Number of points with sample spacing: {}'.format([N, dt]))
    f = fftpack.fftfreq(N)/dt
    return f


def compute_fft(time, data):
    '''Given time and data compute fft parameters'''
    w = hann(time.size)
    freq = time_to_freq(time)
    #fdata = fftpack.fft(data*w)
    fdata = np.fft.fft(data*w)
    return freq, fdata


def compute_welch(time, data, number_segments=10):
    fs = 1/(time[-1] - time[-2])
    # Compute number of points per segment so we have N segments
    nperseg = int(time.size//number_segments)
    f, Pxx_den = signal.welch(data, fs, window='hann', nperseg=nperseg)
    return f, Pxx_den


def compute_noise_spectra(input_directory, squid_run):
    '''Main function to compute noise spectra information from'''
    
    # This is tricky because they must be chained together if we have multiple partials
    list_of_files = glob.glob('{}/*{}*.root'.format(input_directory, squid_run))
    tree = 'data_tree'
    branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform']
    method = 'chain'
    data = readROOT(list_of_files, tree, branches, method)
    data = data['data']
    # Now we need to unfold the data into an array
    # Note Units: waveform data are in Volts
    waveforms = {ch: {} for ch in np.unique(data['Channel'])}
    number_channels = np.unique(data['Channel']).size
    for event, channel in enumerate(data['Channel']):
        waveforms[channel][event//number_channels] = data['Waveform'][event]
    # We now have waveforms separated into dictionaries based on channel number
    # waveforms[2][0] is a numpy array for the first event on channel 2. waveforms[2][1] is the next event
    # Let us collapse this to a concatenated array
    data_array = {}
    time_array = {}
    for channel in waveforms.keys():
        data_array[channel] = concatenate_waveform(waveforms[channel])
        # We don't need to add Timestamp_s[0] + Timestamp_mus[0]/1e6 because we don't need absolute time
        time_array[channel] = np.asarray([i*data['SamplingWidth_s'][0] for i in range(data_array[channel].size)])
        print('The second entry in this channels time array is: {}'.format(time_array[channel][1]))
        print('There are {} total data points'.format(data_array[channel].size))
    # Now we have dictionaries that should have, for a given channel, the time and voltage as single arrays. Let's plot and see
    
    for channel in data_array.keys():
        xlab = 'Time (mus)'
        ylab = 'Signal (mV)'
        title = 'Digitizer Channel ' + str(channel) + ' Signal vs Time'
        fName = '/Users/bwelliver/cuore/bolord/noise_spectra/test/ch_' + str(channel) + '_output_vs_time.png'
        gen_plot_line(time_array[channel]*1e6, data_array[channel], xlab, ylab, title, fName, logx='linear', logy='linear')

    # Now let's try our hand at fft
    for channel in data_array.keys():
        print('Computing fft...')
        freq, fdata = compute_fft(time_array[channel], data_array[channel])
        print('Making plot')
        xlab = 'Frequency (Hz)'
        ylab = 'V'
        title = 'Digitizer Channel ' + str(channel) + ' FFT Signal vs Frequency'
        fName = '/Users/bwelliver/cuore/bolord/noise_spectra/test/ch_' + str(channel) + '_fft_log.png'
        fcut = freq >= 0
        gen_plot_line(freq, np.abs(fdata), xlab, ylab, title, fName, logx='log', logy='log')
        # Try to compute a psd directly
        psd = ((np.abs(fdata)/fdata.size)**2)/(2*(freq[1]-freq[0]))
        xlab = 'Frequency (Hz)'
        ylab = 'PSD V^2 / Hz'
        title = 'Digitizer Channel ' + str(channel) + ' PSD vs Frequency'
        fName = '/Users/bwelliver/cuore/bolord/noise_spectra/test/ch_' + str(channel) + '_psd_log.png'
        gen_plot_line(freq, psd, xlab, ylab, title, fName, logx='log', logy='log')
        #gen_plot_bar(freq, np.abs(fdata), xlab, ylab, title, fName, dx=1000, logx='log', logy='log')
    # Try the welch method
    for channel in data_array.keys():
        print('Computing using welch')
        freq, fdata = compute_welch(time_array[channel], data_array[channel], number_segments=20)
        print('Making plot')
        xlab = 'Frequency (Hz)'
        ylab = 'PSD V^2/Hz'
        title = 'Digitizer Channel ' + str(channel) + ' FFT Signal vs Frequency'
        fName = '/Users/bwelliver/cuore/bolord/noise_spectra/test/ch_' + str(channel) + '_welch_psd_log.png'
        gen_plot_line(freq, fdata, xlab, ylab, title, fName, logx='log', logy='log')
    return None
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDir', help='Specify the full path to the directory with the SQUID root files you wish to use are')
    parser.add_argument('-r', '--squidRun', help='Specify the SQUID run number you wish to grab files from')
    parser.add_argument('-o', '--outputFile', help='Specify output root file. If not a full path, it will be output in the same directory as the input SQUID file')    
    args = parser.parse_args()
    
    # Need to modify this per use? This will be a dictionary that maps currents to a list of partials
    # Note that if a current scan has a zombie in it that we must split the range up into separate scans if we want to use
    # the data otherwise we will get a discontinuity.
    current_partials = {'30': [i+1 for i in range(9)]+[11], 
                        '20': [i+13 for i in range(10)], 
                        '15': [i+24 for i in range(10)], 
                        '10': [i+35 for i in range(5)],
                        
                       }
    compute_noise_spectra(input_directory=args.inputDir, squid_run=args.squidRun)