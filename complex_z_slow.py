import argparse
import glob
import os

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


def mkdpaths(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    return None


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
    if x is None:
        ax.plot(y, linestyle='-', linewidth=0.5)
    else:
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


def gen_plot_points(x, y, xlab, ylab, title, fName, logx='linear', logy='linear'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(16, 9))
    ax = fig2.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xscale(logx)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(logy)
    #ax.set_ylim([-1, 1])
    #ax.set_xlim([-1, 1])
    ax.grid()
    ax.set_title(title)
    fig2.savefig(fName, dpi=100)
    #plt.show()
    #plt.draw()
    plt.close('all')
    return None


def make_fft_plots(input_directory, current, frequency, fVin, fVout):
    '''Helper function to generate fft plots'''
    mkdpaths(input_directory + '/' + 'plots/')
    xLabel = 'Frequency (Hz)'
    yLabel = 'FFT Input Voltage [V]'
    title = 'Amplitude of FFT Input Voltage vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_input_voltage_fft_log.png'
    gen_plot_line(frequency, np.abs(fVin), xLabel, yLabel, title, file_name, logx='log', logy='log')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'FFT Output Voltage [V]'
    title = 'Amplitude of FFT Output Voltage vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_output_voltage_fft_log.png'
    gen_plot_line(frequency, np.abs(fVout), xLabel, yLabel, title, file_name, logx='log', logy='log')
    
    xLabel = 'None (NaN)'
    yLabel = 'FFT Output Voltage [V]'
    title = 'Amplitude of FFT Output Voltage vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_output_voltage_fft_sample_log.png'
    gen_plot_line(None, np.abs(fVout), xLabel, yLabel, title, file_name, logx='log', logy='log')
    return None


def make_z_plots(input_directory, current, square_frequency, frequency, z, fVin):
    '''Helper function to generate fft plots'''
    mkdpaths(input_directory + '/' + 'plots/')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'Impedance [Ohms]'
    title = 'Amplitude of Complex Impedence vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_z_fft_log.png'
    gen_plot_line(frequency, np.abs(z), xLabel, yLabel, title, file_name, logx='log', logy='log')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'Phase of Impedence'
    title = 'Phase of Complex Impedence vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_z_unwrap_phase_log.png'
    gen_plot_line(frequency, np.unwrap(np.angle(z)), xLabel, yLabel, title, file_name, logx='log', logy='linear')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'Phase of Impedence'
    title = 'Phase of Complex Impedence vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_z_phase_log.png'
    gen_plot_line(frequency, np.angle(z), xLabel, yLabel, title, file_name, logx='log', logy='linear')
    
    
    xLabel = 'Phase of Impedence'
    yLabel = 'Magnitude of Impedence'
    title = 'Magnitude of Complex Impedence vs Phase for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_z_magphase_log.png'
    gen_plot_line(np.unwrap(np.angle(z)), np.abs(z), xLabel, yLabel, title, file_name, logx='linear', logy='log')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'Im Z'
    title = 'Imaginary Part of Z vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_imz_log.png'
    gen_plot_line(frequency, z.imag, xLabel, yLabel, title, file_name, logx='log', logy='log')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'Re Z'
    title = 'Real Part of Z vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_rez_log.png'
    gen_plot_line(frequency, z.real, xLabel, yLabel, title, file_name, logx='log', logy='log')
    
    xLabel = 'ReZ'
    yLabel = 'ImZ'
    title = 'Imaginary vs Real parts of Complex Impedence for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_reZimZ.png'
    gen_plot_points(z.real, z.imag, xLabel, yLabel, title, file_name, logx='linear', logy='linear')
    
    
    # Generate only odd harmonics of square_frequency
    fMax = 1e4
    f0 = float(square_frequency)
    fRange = int(fMax/f0/2)
    fcut = frequency >= f0
    vcut = np.abs(fVin) > 10
    for index in range(fRange):
        odd_harmonic = (2*index + 1)*f0
        fcut = np.logical_or(fcut, np.logical_and(frequency >= odd_harmonic - 0.5, frequency <= odd_harmonic + 0.5))
    fcut = np.logical_and(vcut, fcut)
    harmonics = frequency[fcut]
    z_harmonics = z[fcut]
    xLabel = 'ReZ'
    yLabel = 'ImZ'
    title = 'Imaginary vs Real parts of Complex Impedence for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_reZimZ_harmonics.png'
    gen_plot_points(z_harmonics.real, z_harmonics.imag, xLabel, yLabel, title, file_name, logx='linear', logy='linear')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'Im Z'
    title = 'Imaginary Part of Z vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_imz_harmonics_log.png'
    gen_plot_line(harmonics, z_harmonics.imag, xLabel, yLabel, title, file_name, logx='log', logy='linear')
    
    xLabel = 'Frequency (Hz)'
    yLabel = 'Re Z'
    title = 'Real Part of Z vs Frequency for Ib = ' + current + ' uA'
    file_name = input_directory + '/' + 'plots/' + 'ib_' + current + '_uA_rez_harmonics_log.png'
    gen_plot_line(harmonics, z_harmonics.real, xLabel, yLabel, title, file_name, logx='log', logy='linear')
    return None


def get_squid_parameters(channel):
    '''Return SQUID Parameters based on a given channel'''
    
    squid_dictionary = {
        2: {
            'Li': 6e-9,
            'Min': 1/26.062,
            'Mf': 1/33.27,
            'Rfb': 1e4,
            'Rsh': 21e-3,
            'Rbias': 1e4,
            'Cbias': 150e-12
        }
    }
    # Compute auxillary SQUID parameters based on ratios
    squid_dictionary[2]['M'] = -squid_dictionary[2]['Min']/squid_dictionary[2]['Mf']
    squid_dictionary[2]['Lf'] = squid_dictionary[2]['M']*squid_dictionary[2]['M']*squid_dictionary[2]['Li']
    return squid_dictionary[channel]


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
    fdata = fftpack.fft(data*w)
    #fdata = np.fft.fft(data*w)
    #fdata = np.fft.fft(data)
    return freq, fdata


def compute_welch(time, data, number_segments=20):
    fs = 1/(time[-1] - time[-2])
    # Compute number of points per segment so we have N segments
    nperseg = int(time.size//number_segments)
    f, Pxx_den = signal.welch(data, fs, window='hann', nperseg=nperseg)
    return f, Pxx_den


def get_filenames(input_directory, squid_run, partial_list):
    '''Function that will construct a list of file names given input directory, squid run and a set of partials'''
    
    # This assumes partials take the format of '{}/*{}*p*{}.root'.format(input_directory, squid_run, '%.5d'%(3))
    list_of_files = []
    for partial in partial_list:
        partial_string = '%.5d'%(partial)
        matching_files = glob.glob('{}/*{}*p*{}.root'.format(input_directory, squid_run, partial_string))
        if len(matching_files) > 1:
            print('WARNING: More than one partial matched search query')
        for file in matching_files:
            list_of_files.append(file)
    # Check for uniqueness and sort
    list_of_files = sorted(list(set(list_of_files)))
    return list_of_files


def load_data(list_of_files):
    '''Grab the data from a root file chain'''
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
    return data_array, time_array


def get_fft(time, data, input_channel, output_channel, method='fft'):
    '''Compute fft data from various methods'''
    # Now let's try our hand at fft
    if method == 'fft':
        freq, data[input_channel] = compute_fft(time[input_channel], data[input_channel])
        freq, data[output_channel] = compute_fft(time[output_channel], data[output_channel])
        fcut = freq >= 0
        freq = freq[fcut]
        data[input_channel] = data[input_channel][fcut]
        data[output_channel] = data[output_channel][fcut]
    elif method == 'welch':
        # TODO: Warning this only gets us the magnitude we lose phase information
        freq, data[input_channel] = compute_welch(time[input_channel], data[input_channel], number_segments=20)
        freq, data[output_channel] = compute_welch(time[output_channel], data[output_channel], number_segments=20)
        fcut = freq >= 0
        freq = freq[fcut]
        data[input_channel] = data[input_channel][fcut]
        data[output_channel] = data[output_channel][fcut]
    return freq, data[input_channel], data[output_channel]


def process_from_rootfiles_averageZ(list_of_files, input_directory, current, squid_channel=2):
    '''Given list of root files get fft values and compute Z and average the Z'''
    print('Loading data...')
    for index, file in enumerate(list_of_files):
        data_dictionary, time_dictionary = load_data([file])
        print('Computing fft...')
        freq, fVin, fVout = get_fft(time_dictionary, data_dictionary, input_channel=5, output_channel=7, method='fft')
        if index == 0:
            z = compute_z2(fVin, fVout, freq, squid_channel=2)
        else:
            tz = compute_z2(fVin, fVout, freq, squid_channel=2)
            z += tz
    # Data is done so divide
    z = z/len(list_of_files)
    return z, freq, fVin, fVout


def process_from_rootfiles_average(list_of_files, input_directory, current):
    '''Given list of root files we must read the chain in and process the somewhat large amount of data.
    This function will ultimately generate FFT data from some input data and return it
    '''
    
    # Step 1 - Get the data from the root file in a neat dictionary
    print('Loading data...')
    for index, file in enumerate(list_of_files):
        data_dictionary, time_dictionary = load_data([file])
        # Step 2 - We should do some FFT stuff to this set of data
        print('Computing fft...')
        if index == 0:
            freq, fVin, fVout = get_fft(time_dictionary, data_dictionary, input_channel=5, output_channel=7, method='fft')
        else:
            tfreq, tfVin, tfVout = get_fft(time_dictionary, data_dictionary, input_channel=5, output_channel=7, method='fft')
            freq += tfreq
            fVin += tfVin
            fVout += tfVout
    # Data is done so divide
    N = len(list_of_files)
    freq = freq/N
    fVin = fVin/N
    fVout = fVout/N
    return freq, fVin, fVout



def process_from_rootfiles(list_of_files):
    '''Given list of root files we must read the chain in and process the somewhat large amount of data.
    This function will ultimately generate FFT data from some input data and return it
    This will basically compute fft on a per root file basis and average them together at the end
    '''
    
    # Step 1 - Get the data from the root file in a neat dictionary
    print('Loading data...')
    data_dictionary, time_dictionary = load_data(list_of_files)
    # Step 2 - We should do some FFT stuff to this set of data
    print('Computing fft...')
    freq, fVin, fVout = get_fft(time_dictionary, data_dictionary, input_channel=5, output_channel=7, method='fft')
    return freq, fVin, fVout


def compute_z2(fVin, fVout, frequency, squid_channel=2):
    '''Compute complex impedence based on fft transformed data and squid parameters'''
    squid_parameters = get_squid_parameters(squid_channel)
    Rbias = squid_parameters['Rbias']
    Cbias = squid_parameters['Cbias']
    Rfb = squid_parameters['Rfb']
    Lfb = squid_parameters['Lf']
    M = squid_parameters['M']
    Rsh = squid_parameters['Rsh']
    # Zbias = Rb - j/wC
    #Zbias = squid_data['Rbias'] - 1j/(2*np.pi*fft_data['frequency']*squid_data['Cbias'])
    Zbias = Rbias / (1 + 2*np.pi*frequency*1j*Rbias*Cbias)
    #Zbias = squid_data['Rbias'] / (1 + 2*np.pi * fft_data['frequency'] * 1j * squid_data['Rbias'] * squid_data['Cbias'])
    #Zbias = squid_data['Rbias']
    Zfb = Rfb + 2*np.pi*1j*frequency*Lfb
    #Zfb = squid_data['Rfb'] + 2*np.pi*1j * fft_data['frequency'] * squid_data['Lf']
    # M*Rfb/vout = 1/iTES and Rsh/Zbias is ?
    z = (fVin/fVout)*(M*Zfb*Rsh)/Zbias
    #z = (fft_data['fv_in']/fft_data['fv_out'])*(squid_data['M']  * Zfb * squid_data['Rsh'])/(Zbias)
    #z = fft_data['fv_in']/(squid_data['Rbias'] * (fft_data['fv_out']/squid_data['Rfb']/squid_data['M'])) - squid_data['Rsh'] - 1j*2*np.pi*fft_data['frequency']*squid_data['Li']
    return z


def compute_z(fVin, fVout, frequency, squid_channel=2):
    '''Function that will compute the complex impedence for a given SQUID channel'''
    squid_parameters = get_squid_parameters(squid_channel)
    # Electrical model for the Ztes:
    # We have an input bias --> iBias = Vbias/Zbias
    # We have an output voltage which can map to iTES --> iTES = Vout/(M*Rf) or maybe Rf should be Zf = Rf + jwLf
    # Ztes nominally then is vTES/iTES which is to say Vbias/(Zbias*iTES) * T(w) ?
    # Compute the various things then...
    Rbias = squid_parameters['Rbias']
    Cbias = squid_parameters['Cbias']
    Rfb = squid_parameters['Rfb']
    Lfb = squid_parameters['Lf']
    M = squid_parameters['M']
    Rsh = squid_parameters['Rsh']
    
    Zbias = Rbias + 1/(2*np.pi*1j*frequency*Cbias)
    Zfb = Rfb #+ 2*np.pi*1j*frequency*Lfb
    iTES = fVout/(Rfb*M)
    vTES = Rsh*((fVin/Zbias) - iTES)
    zTES = vTES / iTES
    return zTES


def compute_complex_impedance(input_directory, squid_run, partial_dictionary, square_frequency=1):
    '''Main function to compute noise spectra information from'''
    
    # The way this will work is that we will operate on one current at a time the whole way through.
    # First load the data array and compute the fft information. Here we do it either through fft or welch
    # From the fft information then use the electrical model to compute Z(f)
    # From Z(f) use the square wave frequency to select points around it
    # Then fit and plot
    # Store sub-sampled f and Z data as current dependent dictionary
    
    
    for current, partial_list in partial_dictionary.items():
        list_of_files = get_filenames(input_directory, squid_run, partial_list)
        # Next we need to process root files in our list
        print('Starting data processing for current {}'.format(current))
        #freq, fVin, fVout = process_from_rootfiles(list_of_files)
        #freq, fVin, fVout = process_from_rootfiles_average(list_of_files, input_directory, current)
        # Try to average not fft but z
        # Now make some generic fft plots
        #print('Making fft plots...')
        #make_fft_plots(input_directory, current, freq, fVin, fVout)
        # FFT plots are basic and done now let us obtain SQUID parameters needed for Z
        # Here we can compute basic quantities: vTES, iTES and then Z or Ztes
        #z = compute_z2(fVin, fVout, freq, squid_channel=2)
        z, freq, fVin, fVout = process_from_rootfiles_averageZ(list_of_files, input_directory, current, squid_channel=2)
        make_z_plots(input_directory, current, square_frequency, freq, z, fVin)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDir', 
                        help='Specify the full path to the directory with the SQUID root files you wish to use are')
    parser.add_argument('-r', '--squidRun', 
                        help='Specify the SQUID run number you wish to grab files from')
    parser.add_argument('-o', '--outputFile', 
                        help='Specify output root file. If not a full path, it will be output in the same directory as the input SQUID file')
    parser.add_argument('-f', '--squareFrequency', 
                        help='Specify the frequency of the square wave being applied')
    args = parser.parse_args()
    
    # Need to modify this per use? This will be a dictionary that maps currents to a list of partials
    # Note that if a current scan has a zombie in it that we must split the range up into separate scans if we want to use
    # the data otherwise we will get a discontinuity.
    current_partials = {'30': [i+1 for i in range(2)], 
                        '20': [i+13 for i in range(5)], 
                        '15': [i+24 for i in range(10)], 
                        '10': [i+37 for i in range(5)],
                        '8': [i+47 for i in range(10)],
                        '7.7': [i+60 for i in range(9)],
                        '7.4': [i+70 for i in range(7)],
                        '7.2': [i+82 for i in range(10)],
                        '0': [i+93 for i in range(10)]
                       }
    
    current_partials = {'8': [i+47 for i in range(1)]
                       }
    
    compute_complex_impedance(input_directory=args.inputDir, squid_run=args.squidRun, partial_dictionary=current_partials, square_frequency=args.squareFrequency)