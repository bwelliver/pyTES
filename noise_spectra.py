import os
import argparse
import re
from os.path import isabs
from os.path import dirname
from os.path import basename

import argparse
import glob

import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt
from scipy import fftpack
from scipy.signal import hann
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt

from readROOT import readROOT

eps = np.finfo(float).eps
ln = np.log
mp.rcParams['agg.path.chunksize'] = 10000

def mkdpaths(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    return True


def natural_sort_key(string, _dre=re.compile(r'(\d+)')):
    '''Defines a natural sorting key for use with sorting file lists'''
    key = [int(text) if text.isdigit() else text.lower() for text in _dre.split(string)]
    return key


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
    ax.set_ylim((1e-14, 1e-7))
    fig2.savefig(fName, dpi=100)
    plt.close('all')
    return None


def gen_plot_line(x, y, xlab, ylab, title, fName, peaks=None, ylim=(1e-15, 0), logx='log', logy='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(32, 8))
    ax = fig2.add_subplot(111)
    ax.plot(x, y, linestyle='-', linewidth=1)
    if peaks is not None:
        # Let's get the highest 15 peaks
        ax.plot(peaks, y[peaks], 'x')
        for i, label in enumerate(peaks):
            ax.text(peaks[i], y[peaks[i]]*1.5, str(label) + ' Hz - ' + '{:2.3f} pV^2/Hz'.format(y[peaks[i]]*1e12))
    ax.set_xscale(logx)
    fontsize=18
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    ax.set_yscale(logy)
    ax.set_ylim((ylim))
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(which='minor')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fontsize)
    fig2.savefig(fName, dpi=200)
    plt.close('all')
    return None


def gen_plot_line_both(ax, freq, fdata, channel):
    '''Overlay both plots'''
    ax.plot(freq, fdata, linestyle='-', linewidth=0.5, label="Channel {}".format(channel))
    return ax


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
    print('Welch input using sampling frequency of {} Hz'.format(fs))
    print('Welch using nperseg={}'.format(nperseg))
    f, Pxx_den = signal.welch(data, fs, window='hann', nperseg=nperseg)
    return f, Pxx_den


def compute_noise_spectra(input_directory, squid_run, mode='old', number_segments=10):
    '''Main function to compute noise spectra information from'''
    
    # This is tricky because they must be chained together if we have multiple partials
    list_of_files = glob.glob('{}/*{}*.root'.format(input_directory, squid_run))
    list_of_files.sort(key=natural_sort_key)
    tree = 'data_tree'
    # New mode:
    if mode == 'new':
        chlist = 'ChList'
        branches = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s']
        channels = readROOT(list_of_files[0], None, None, method='single', tobject=chlist)
        channels = channels['data'][chlist]
        branches = branches + ['Waveform' + '{:03d}'.format(int(i)) for i in channels]
    else:
        branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform']
    method = 'chain'
    data = readROOT(list_of_files, tree, branches, method)
    data = data['data']
    # Make output directory
    outdir = '/Users/bwelliver/cuore/bolord/noise_spectra/sr_' + str(squid_run)
    mkdpaths(outdir)
    
    # New mode:
    # Waveforms come in their own things now.
    # Waveform%d[ev] = np.array
    if mode == 'new':
        data_array = {ch: None for ch in channels}
        time_array = {ch: None for ch in channels}
        for channel in data_array.keys():
            data_array[channel] = concatenate_waveform(data['Waveform00' + str(int(channel))])
            time_array[channel] = np.asarray([i*data['SamplingWidth_s'][0] for i in range(data_array[channel].size)])
            print('The second entry in this channels time array is: {}'.format(time_array[channel][1]))
            print('There are {} total data points'.format(data_array[channel].size))
    else:
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
        print('Making output vs time for channel {}'.format(channel))
        xlab = 'Time (mus)'
        ylab = 'Signal (mV)'
        title = 'Digitizer Channel ' + str(channel) + ' Signal vs Time for SR ' + str(squid_run)
        fName = outdir + '/ch_' + str(channel) + '_output_vs_time.png'
        gen_plot_line(time_array[channel]*1e6, data_array[channel], xlab, ylab, title, fName, ylim=(-0.1, 0.1), logx='linear', logy='linear')

    # Now let's try our hand at fft
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    for channel in data_array.keys():
        print('Computing fft...')
        freq, fdata = compute_fft(time_array[channel], data_array[channel])
        print('Making plot')
        xlab = 'Frequency (Hz)'
        ylab = 'V'
        title = 'Digitizer Channel ' + str(channel) + ' FFT Signal vs Frequency for SR ' + str(squid_run)
        fName = outdir + '/ch_' + str(channel) + '_fft_log.png'
        fcut = freq >= 0
        gen_plot_line(freq, np.abs(fdata), xlab, ylab, title, fName, logx='log', logy='log')
        # Try to compute a psd directly
        psd = ((np.abs(fdata)/fdata.size)**2)/(2*(freq[1]-freq[0]))
        xlab = 'Frequency (Hz)'
        ylab = 'PSD V^2 / Hz'
        title = 'Digitizer Channel ' + str(channel) + ' PSD vs Frequency for SR ' + str(squid_run)
        fName = outdir + '/ch_' + str(channel) + '_psd_log.png'
        gen_plot_line(freq, psd, xlab, ylab, title, fName, ylim=(1e-15, 0), logx='log', logy='log')
        #gen_plot_bar(freq, np.abs(fdata), xlab, ylab, title, fName, dx=1000, logx='log', logy='log')
        ax = gen_plot_line_both(ax, freq, psd, channel)
    xlab = 'Frequency (Hz)'
    ylab = 'PSD V^2/Hz'
    title = 'Digitizer Channels ' + ' FFT Signal vs Frequency for SR ' + str(squid_run)
    fName = outdir + '/both_channels_psd_log.png'
    ax.set_xscale('log')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale('log')
    ax.set_ylim((1e-15, 0))
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    print('Saving both manual psd...')
    fig.savefig(fName, dpi=200)
    plt.close('all')
    # Try the welch method
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    for channel in data_array.keys():
        print('Computing using welch with {} segments'.format(number_segments))
        freq, fdata = compute_welch(time_array[channel], data_array[channel], number_segments=number_segments)
        # Find peaks toooooooo
        #peaks = find_peaks_cwt(fdata, np.asarray([i+0.1 for i in range(10)]), noise_perc=10, min_snr=20)
        peaks = None
        print('Making plot')
        xlab = 'Frequency (Hz)'
        ylab = 'PSD V^2/Hz'
        title = 'Digitizer Channel ' + str(channel) + ' FFT Signal vs Frequency for SR ' + str(squid_run)
        fName = outdir + '/ch_' + str(channel) + '_welch_psd_log.png'
        gen_plot_line(freq, fdata, xlab, ylab, title, fName, peaks=peaks, logx='log', logy='log')
        ax = gen_plot_line_both(ax, freq, fdata, channel)
    xlab = 'Frequency (Hz)'
    ylab = 'PSD V^2/Hz'
    title = 'Digitizer Channels ' + ' FFT Signal vs Frequency for SR ' + str(squid_run)
    fName = outdir + '/both_channels_welch_psd_log.png'
    ax.set_xscale('log')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale('log')
    ax.set_ylim((1e-15, 1e-1))
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    print('Saving both welch psd...')
    fig.savefig(fName, dpi=200)
    plt.close('all')
    # Get both overlapped
    #gen_plot_line_both(outdir, time_array, data_array, number_segments=50, logx='log', logy='log')
    # Try a spectrogram
#    print('Trying to do a spectrogram')
#    last_channel = list(data_array.keys())[-1]
#    fsample = 250000 # for run 902 it is 250000 Hz
#    nperseg = int(data_array[last_channel].size//2)
#    f, t, Sxx = signal.spectrogram(data_array[last_channel], fsample, nperseg=nperseg)
#    plt.pcolormesh(t, f, np.log(Sxx))
#    #print(Sxx)
#    print('Max Sxx is: {}'.format(Sxx.max()))
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
#    plt.savefig(outdir + '/spectrogram_channel_7.png', dpi=200)
#    plt.ylim((0,5))
#    plt.savefig(outdir + '/spectrogram_channel_7_lowF.png', dpi=200)
#    
#    
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
#    plt.yscale('log')
#    plt.ylim((2e-2, 125e3))
#    plt.savefig(outdir + '/spectrogram_channel_7_log.png', dpi=200)
#    plt.close('all')
#    
#    print('Trying to do a spectrogram')
#    f, t, Sxx = signal.spectrogram(data_array[5], 250e3)
#    plt.pcolormesh(t, f, np.log(Sxx))
#    #print(Sxx)
#    print('Max Sxx is: {}'.format(Sxx.max()))
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
#    plt.savefig(outdir + '/spectrogram_channel_5.png', dpi=200)
#    
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
#    plt.yscale('log')
#    plt.ylim((2e-2, 125e3))
#    plt.savefig(outdir + '/spectrogram_channel_5_log.png', dpi=200)
#    plt.close('all')
    return None
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDir', help='Specify the full path to the directory with the SQUID root files you wish to use are')
    parser.add_argument('-r', '--squidRun', help='Specify the SQUID run number you wish to grab files from')
    parser.add_argument('-o', '--outputFile', help='Specify output root file. If not a full path, it will be output in the same directory as the input SQUID file')
    parser.add_argument('-n', '--newMode', default=None, help='Specify if you want to use the new mode for root files or not')
    parser.add_argument('-s', '--numSegments', default=10, type=int, help='Specify the number of segments for Welch Method')
    args = parser.parse_args()
    if args.newMode is not None:
        root_mode = 'new'
    else:
        root_mode = 'old'
    compute_noise_spectra(input_directory=args.inputDir, squid_run=args.squidRun, mode=root_mode, number_segments=args.numSegments)