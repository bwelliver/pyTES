import argparse
import glob
import re
from os.path import splitext


import numpy as np
import pandas as pd
import matplotlib as mp
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def ratio_model_function(f, *args):
    '''Complex version of ratio fit'''
    # rn, rl, lin = [0.71431989, 24.46217e-3 + 21e-3, *args]
    # rn, rl, lin = [0.71431989, *args]
    rn, rl, lin = args
    zl = 1j * 2 * np.pi * f * lin
    zn = rn + rl + zl
    zsc = rl + zl
    return zn/zsc


def ratio_model_function_wrapper(f, *args):
    '''flat version of ratio fit'''
    ratio = ratio_model_function(f, *args)
    return np.append(ratio.real, ratio.imag)


def natural_sort_key(string, _dre=re.compile(r'(\d+)')):
    '''Defines a natural sorting key for use with sorting file lists'''
    key = [int(text) if text.isdigit() else text.lower() for text in _dre.split(string)]
    return key


def invert_ratio(data, invert=False):
    '''Function that will invert the real and imaginary parts of the complex frequency response'''
    if invert is True:
        data.update((f, 1/resp) for f, resp in data.items())
    return data


def gen_plot_points_fit(xdata, ydata, xfit, yfit, results, perr, labels, **kwargs):
    '''Create generic plots that may be semilogx (default)'''
    xlabel = labels['xlabel']
    ylabel = labels['ylabel']
    title = labels['title']
    figname = labels['figname']

    figsize = kwargs.get('figsize', (12, 12))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.plot(xfit, yfit, 'r-', marker='None', linewidth=2)
    ax.set_xlabel(xlabel, fontsize=18, horizontalalignment='right', x=1.0)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)

    # Parse other relevant kwargs
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    if kwargs.get('minorticks', None) is not None:
        ax.minorticks_on()
        ax.grid(which='minor')
    ax.tick_params(axis='both', which='major', labelsize=22)
    # Set up text strings for fit
    rn, rl, lin = results
    rn_err, rl_err, lin_err = perr
    tRn = r'$R_{n} = %.5f \pm %.5f \mathrm{m\Omega}$' % (rn*1e3, rn_err*1e3)
    tRl = r'$R_{L} = %.5f \pm %.5f \mathrm{m\Omega}$' % (rl*1e3, rl_err*1e3)
    tLin = r'$L_{in} = %.5f \pm %.5f \mathrm{\mu H}$' % (lin*1e6, lin_err*1e6)
    text_string = tRn + '\n' + tRl + '\n' + tLin
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.2, 0.2, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    set_aspect = kwargs.get('set_aspect', None)
    if set_aspect is not None:
        ax.set_aspect(set_aspect, 'datalim')
    fig.savefig(figname, dpi=150, bbox_inches='tight')
    # plt.show()
    # plt.draw()
    plt.close('all')
    return True


def gen_plot_points(xdata, ydata, labels, **kwargs):
    '''Create generic plots that may be semilogx (default)'''
    xlabel = labels['xlabel']
    ylabel = labels['ylabel']
    title = labels['title']
    figname = labels['figname']

    figsize = kwargs.get('figsize', (12, 12))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xlabel(xlabel, fontsize=18, horizontalalignment='right', x=1.0)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)

    # Parse other relevant kwargs
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    if kwargs.get('minorticks', None) is not None:
        ax.minorticks_on()
        ax.grid(which='minor')
    ax.tick_params(axis='both', which='major', labelsize=22)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    set_aspect = kwargs.get('set_aspect', None)
    if set_aspect is not None:
        ax.set_aspect(set_aspect, 'datalim')
    fig.savefig(figname, dpi=150, bbox_inches='tight')
    # plt.show()
    # plt.draw()
    plt.close('all')
    return True


def generate_model_diagnostic_plots(output_directory, ratio, model_function, results, perr):
    '''Generate diagnostic plots for the model fits'''
    # Split into arrays
    freq = np.fromiter(ratio.keys(), dtype='float')
    ratio = np.fromiter(ratio.values(), dtype='c16')
    model_freq = np.linspace(freq.min(), freq.max(), int(1e5))
    model_ratio = model_function(model_freq, *results)
    if len(results) == 2:
        results = [0.71431989, *results]
        perr = [26.98e-3, *perr]
    if len(results) == 1:
        results = [0.71431989, 24.46217e-3 + 21e-3, *results]
        perr = [26.98e-3, 0.61583e-3, *perr]
    # Plot the real, imaginary, and magnitude vs frequency
    labels = {'xlabel': 'Frequency [Hz]',
              'ylabel': 'Re Zn/Zsc',
              'title': 'Power Spectrum of Model Impedance Ratio Real',
              'figname': output_directory + '/real_ratio_model_tones.png'
              }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points_fit(freq, ratio.real, model_freq, model_ratio.real, results, perr, labels, **formargs)

    labels = {'xlabel': 'Frequency [Hz]',
              'ylabel': 'Im Zn/Zsc',
              'title': 'Power Spectrum of Model Impedance Ratio Imaginary',
              'figname': output_directory + '/imag_ratio_model_tones.png'
              }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points_fit(freq, ratio.imag, model_freq, model_ratio.imag, results, perr, labels, **formargs)

    labels = {'xlabel': 'Frequency [Hz]',
              'ylabel': 'Abs Zn/Zsc',
              'title': 'Power Spectrum of Model Impedance Ratio Magnitude',
              'figname': output_directory + '/abs_ratio_model_tones.png'
              }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points_fit(freq, np.absolute(ratio), model_freq, np.absolute(model_ratio), results, perr, labels, **formargs)

    # Make Nyquist plot (Im vs Re)
    labels = {'xlabel': 'Re Zn/Zsc',
              'ylabel': 'Im Zn/Zsc',
              'title': 'Nyquist Plot of Model Impedance Ratio',
              'figname': output_directory + '/nyquist_ratio_model.png'
              }
    formargs = {'figsize': (16, 16),
                'xscale': 'linear',
                'yscale': 'linear',
                'set_aspect': 'equal'}
    gen_plot_points_fit(ratio.real, ratio.imag, model_ratio.real, model_ratio.imag, results, perr, labels, **formargs)
    return True


def generate_diagnostic_plots(output_directory, ratio, mode='ratio'):
    '''Catch all function to generate specific plots'''
    # Split into arrays
    freq = np.fromiter(ratio.keys(), dtype='float')
    ratio = np.fromiter(ratio.values(), dtype='c16')

    # Plot the real, imaginary, and magnitude vs frequency
    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re Zn/Zsc',
                  'title': 'Power Spectrum of Measured Impedance Ratio Real',
                  'figname': output_directory + '/psd_real_znsc_tones.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re G',
                  'title': 'Power Spectrum of Transfer Function Real',
                  'figname': output_directory + '/psd_real_g_tones.png'
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points(freq, np.real(ratio), labels, **formargs)

    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im Zn/Zsc',
                  'title': 'Power Spectrum of Measured Impedance Ratio Imaginary',
                  'figname': output_directory + '/psd_imag_znsc_tones.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im G',
                  'title': 'Power Spectrum of Transfer Function Imaginary',
                  'figname': output_directory + '/psd_imag_g_tones.png'
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points(freq, np.imag(ratio), labels, **formargs)

    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs Zn/Zsc',
                  'title': 'Power Spectrum of Measured Impedance Ratio Magnitude',
                  'figname': output_directory + '/psd_abs_znsc_tones.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs G',
                  'title': 'Power Spectrum of Transfer Function Magnitude',
                  'figname': output_directory + '/psd_abs_g_tones.png'
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points(freq, np.absolute(ratio), labels, **formargs)

    # Make Nyquist plot (Im vs Re)
    if mode == 'ratio':
        labels = {'xlabel': 'Re Zn/Zsc',
                  'ylabel': 'Im Zn/Zsc',
                  'title': 'Nyquist Plot of Impedance Ratio',
                  'figname': output_directory + '/nyquist_ratio.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Re G',
                  'ylabel': 'Im G',
                  'title': 'Nyquist Plot of Transfer Function',
                  'figname': output_directory + '/nyquist_g.png'
                  }
    formargs = {'figsize': (16, 16), 'xscale': 'linear', 'yscale': 'linear', 'set_aspect': 'equal'}
    gen_plot_points(np.real(ratio), np.imag(ratio), labels)
    return True


# File IO
def get_frequency_list(filename):
    '''Load a fast digitizer txt and get the sweept frequency values
    The first few lines are header so we will skip them
    But also we can have more than one "end of header" string so search backwards.
    This is longer but won't make copies of the array. Note that data starts 2 lines after end of header is printed
    '''
    print('The filename is: {}'.format(filename))
    file_ext = splitext(filename)[1]
    lines_to_skip = 1 if file_ext == '.txt' else 2
    # print('Skipping {} lines'.format(lines_to_skip))
    with open(filename, 'r') as file:
        lines = file.readlines()
    eoh = 0
    for index, line in reversed(list(enumerate(lines))):
        if line.find('***End_of_Header') > -1:
            eoh = index
            break
    # print('End of header line at: {} and the line is: {}'.format(eoh, lines[eoh]))
    tones = []
    start = eoh + lines_to_skip
    for line in lines[start:]:
        line = line.strip('\n').split('\t')
        tones.append(float(line[0]))
    return np.array(tones)


def get_data_pandas(filename, freq=None):
    '''Load data into memory via pandas and return what we need, namely the specific
    tone and response value. This is faster than readlines()
    '''
    print('The filename is: {}'.format(filename))
    # file_ext = splitext(filename)[1]
    # lines_to_skip = 1 if file_ext == '.txt' else 2
    data = pd.read_csv(filename, delimiter='\t')
    cols = data.columns.tolist()
    df = data[cols[0]][1] - data[cols[0]][0]
    cut = (data[cols[0]] > freq - df/2) & (data[cols[0]] < freq + df/2)
    tones = data[cols[0]][cut].to_numpy()
    zdata = data[cols[1]][cut].to_numpy()
    return tones, zdata


def get_data(filename, freq=None):
    '''Load a fast digitizer lvm and get signal and response columns
    The first few lines are header so we will skip them
    But also we can have more than one "end of header" string so search backwards.
    This is longer but won't make copies of the array. Note that data starts 2 lines after end of header is printed
    '''
    print('The filename is: {}'.format(filename))
    file_ext = splitext(filename)[1]
    lines_to_skip = 1 if file_ext == '.txt' else 2
    # print('Skipping {} lines'.format(lines_to_skip))
    with open(filename, 'r') as file:
        lines = file.readlines()
    eoh = 0
    for index, line in reversed(list(enumerate(lines))):
        if line.find('***End_of_Header') > -1:
            eoh = index
            break
    # print('End of header line at: {} and the line is: {}'.format(eoh, lines[eoh]))
    tones = []
    zdata = []
    start = eoh + lines_to_skip
    # Determine df
    df = []
    for line in lines[start:start+2]:
        line = line.strip('\n').split('\t')
        df.append(float(line[0]))
    df = df[1] - df[0]
    if freq is None:
        for line in lines[start:]:
            line = line.strip('\n').split('\t')
            tones.append(float(line[0]))
            zdata.append(float(line[1]))
    else:
        for line in lines[start:]:
            line = line.strip('\n').split('\t')
            lfreq = float(line[0])
            if freq - df/2 < lfreq < freq + df/2:
                tones.append(lfreq)
                zdata.append(float(line[1]))
                break
    return np.asarray(tones), np.asarray(zdata)


def parse_lvm_file(infile, intype='response', freq=None):
    '''Function to parse a LVM file containing complex Z data
    There are two bits of information needed: the frequency and the data
    Frequency can be obtained from file name via the pattern '*_frequencyHz*'
    '''

    if intype == 'response':
        freq, zdata = get_data_pandas(infile, freq)
        return freq, zdata
    if intype == 'frequency_list':
        freq = get_frequency_list(infile)
        return freq
    return None


def get_list_of_files(input_directory, subdir, run, temperature, current):
    '''Get the list of files
    In general useful information is encoded in the file names and also user specified
    input_directory/*run{}/*T{}mK*/*{}uA*.txt''
    '''
    list_of_files = glob.glob('{}/*run{}/*T{}mK*/{}/*_{}uA*.txt'.format(input_directory, run, temperature, subdir, current))
    list_of_files.sort(key=natural_sort_key)
    return list_of_files


def split_files_by_type(list_of_files):
    '''Split master list of files into frequency files and data files'''
    frequency_files = []
    new_list_of_files = []
    for file in list_of_files:
        isfrequency = file.find('tones') > -1 or file.find('freq') > -1
        if isfrequency is True:
            frequency_files.append(file)
        else:
            new_list_of_files.append(file)
    return frequency_files, new_list_of_files


def get_tones(frequency_files):
    '''Get the tones to be used in this sweep from the corresponding tone file
    '''
    # Step 1: Find the right tone file for this current
    tone_file = frequency_files[0]
    # Step 2: Load up the file
    tones = parse_lvm_file(tone_file, intype="frequency_list")
    return tones


def parse_file(file_name, tones):
    '''Parse data from the specified file'''
    # We have a file that has some data in it...what data is it and where does it go?
    is_real = file_name.find('real') != -1
    frequency_index = int(file_name.strip('.txt').split('_')[-1])
    if frequency_index >= tones.size:
        print('Frequency index of {} is found but frequency list is only {} entries long. Skipping...'.format(frequency_index, len(tones)))
        return None
    # Load the file...we know what frequency we are looking for so use that to our advantage
    desired_tone = tones[frequency_index]
    frequencies, response = parse_lvm_file(file_name, freq=desired_tone)
    # It is possible the file contains multiple records for a given frequency. If so average them together
    tone = frequencies[0]
    response = np.mean(response)
    # Now we have a single point (f, r). If it is imaginary multiply the response by 1j.
    if not is_real:
        response *= 1j
    return tone, response


def get_response(data_files, tones):
    '''Get the response data to be used in this sweep from corresponding files'''
    # Here we need to return set of ratios. The general format is freq -> ratio
    # Each file is full of data we do not need.
    # We will store these in a dictionary such that d[f] = response
    data = {}
    for file in data_files:
        tone, response = parse_file(file, tones)
        if tone in data.keys():
            data[tone] += response
        else:
            data[tone] = response
    # Now we have a data dictionary with keys of frequencies and values are complex response
    return data


def get_tones_and_response(input_directory, subdir, run, temperature, current, invert=True):
    '''Function that will return a dictionary of tones and corresponding ratios
    '''
    # The value of current tells what files go grab so in principle we do NOT need to pass it
    # to other functions since the list we have to work with should only contain the correct currents
    list_of_files = get_list_of_files(input_directory, subdir, run, temperature, current)
    frequency_files, data_files = split_files_by_type(list_of_files)
    tones = get_tones(frequency_files)
    response = get_response(data_files, tones)
    response = invert_ratio(response, invert=invert)
    return tones, response


def get_ratio(input_directory, subdir, run, temperature, sc, normal):
    '''Process steps to return a dictionary comprised of keys that are the tones and values
    that are the complex response at that particular tone
    '''
    # Step 1: Get the SC and normal tones and ratios
    sc_tones, sc_response = get_tones_and_response(input_directory, subdir, run, temperature, current=sc)
    n_tones, n_response = get_tones_and_response(input_directory, subdir, run, temperature, current=normal)
    if np.any(sc_tones != n_tones):
        print('Warning: SC tones and Normal tones do not agree!')
        raise Exception('SC and Normal tone lists are not the same!')
    if np.any(sc_tones != list(sc_response.keys())):
        print('Warning! Response dictionary keys not the same as tone list!')
        raise Exception('Warning! Response dictionary keys not the same as tone list!')
    if sc_response.keys() != n_response.keys():
        raise Exception('Warning! SC response keys are different from Normal response keys!')
    # Step 2: Compute the ratio of the normal response to the sc response
    ratio = {f: n_response[f]/sc_response[f] for f in sc_response}
    return ratio, n_response


def fit_ratio_model(ratio, model_func, **kwargs):
    '''Function to perform fitting of the frequency response ratio to a model'''

    tones = np.fromiter(ratio.keys(), dtype=float)
    ratio = np.fromiter(ratio.values(), dtype=np.complex128)
    flatratio = np.append(ratio.real, ratio.imag)
    result, pcov = curve_fit(model_func, tones, flatratio, **kwargs)
    perr = np.sqrt(np.diag(pcov))
    return result, perr


def get_transfer_function(n_response, results):
    '''Using the results and the normal mode response compute the empirical SQUID transfer function
    G(f)
    '''
    if len(results) == 2:
        Rn, Rl, L = [714.31989e-3, *results]
    elif len(results) == 1:
        Rn, Rl, L = [714.31989e-3, 24.46217e-3 + 21e-3, *results]
    else:
        Rn, Rl, L = results
    Zbias = 10000
    Rf = 10000
    M = -1.26314
    Rs = 21e-3
    dc_factor = (Rs * Rf * M)/Zbias
    # zcirc_normal = Rn + Rl + (2 * 1j * np.pi) * L * np.fromiter(n_response.keys(), dtype=float)
    # Keeping in mind n_response is a dictionary with keys = tones and values = complex response
    # we compute g to be a similar thingsa
    # zcirc_normal(f) = Rn + Rl + 2j*pi*f*L
    g = {}
    for tone, response in n_response.items():
        g[tone] = (response * dc_factor) / (Rn + Rl + (2j * np.pi * L * tone))
    return g


def compute_transfer_function(input_directory, output_directory, subdir, run, temperature, sc, normal):
    '''Function to handle computation and diagnostic plots related
    to the transfer function.
    Outputs:
        G: the transfer function
        Rn: the TES normal resistance
        Rl: the Thevenin equivalent load resistance (R_shunt + R_para)
        L: the TES input line inductance
    These outputs are fit results
    '''
    # Step 1: Get the ratio dictionary
    ratio, n_response = get_ratio(input_directory, subdir, run, temperature, sc, normal)
    # Step 2: Diagnostic plots of the ratio?
    print('Generating diagnostic plots in {}'.format(output_directory))
    generate_diagnostic_plots(output_directory, ratio, mode='ratio')
    # Step 3: Fit to a model [rn, rl, lin]
    print('Attempting to fit ratio model')
    fitargs = {'p0': (10e-3, 0.1e-8), 'method': 'lm'}
    # fitargs = {'p0': (10e-3, 0.1e-8), 'bounds': ((0, 0), (np.inf, np.inf)), 'method': 'trf'}
    results, perr = fit_ratio_model(ratio, ratio_model_function_wrapper, **fitargs)
    if len(results) == 2:
        print('The fit results are: Rn = {} mOhm, Rl = {} mOhm, Lin = {} uH'.format('fixed', results[0]*1e3, results[1]*1e6))
    elif len(results) == 1:
        print('The fit results are: Rn = {} mOhm, Rl = {} mOhm, Lin = {} uH'.format('fixed', 'fixed', results[0]*1e6))
    else:
        print('The fit results are: Rn = {} mOhm, Rl = {} mOhm, Lin = {} uH'.format(results[0]*1e3, results[1]*1e3, results[2]*1e6))
    # Step 4: Diagnostic plots of the model
    generate_model_diagnostic_plots(output_directory, ratio, ratio_model_function, results, perr)

    # Step 5: Now we can create G(w)
    g = get_transfer_function(n_response, results)
    generate_diagnostic_plots(output_directory, g, mode='transfer')
    if len(results) == 2:
        Rn, Rl, L = [714.31989e-3, *results]
    elif len(results) == 1:
        Rn, Rl, L = [714.31989e-3, 24.46217e-3 + 21e-3, *results]
    else:
        Rn, Rl, L = results
    return (g, Rn, Rl, L)


def process_complex_impedance(indir, outdir, subdir, only_transfer, run, temperature, sc, normal, bias):
    '''Main function that implements complex impedance computations'''

    # Step 1: Generate the transfer function if we don't have it already
    # IMPORTANT: Since the ratio function is basically (Rn + Rl + 2jpifL)/(Rl + 2jpifL)
    # an infinite number of solutions exist of the form a*(Rn + Rl + 2jpifL) / a*(Rl + 2jpifL)
    # (i.e., a*Rn, a*Rl, a*Lin). We *MUST* provide at least 1 of these as fixed values to get an
    # appropriate scaling factor.
    G, Rn, Rl, L = compute_transfer_function(indir, outdir, subdir, run, temperature, sc, normal)

    # Step 2: Compute the complex impedance
    if not only_transfer:
        return bias
    # compute_z(input_directory=indir, output_directory=outdir, transfer_function=G, normalR=Rn, loadR=Rl, inL=L)
    return None


def get_args():
    '''Get input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDirectory',
                        help='Specify the full path of the directory that contains all the files you wish to use')
    parser.add_argument('-o', '--outputDirectory',
                        help='Specify the full path of the output directory to put plots and root files.\
                        If it is not a full path, a plots and root subdirectory will be added in the input directory')
    parser.add_argument('-d', '--subDirectory',
                        help='Specify a subdirectory name inside the inputDirectory/run$/T$mK/ root directory to get files from')
    parser.add_argument('-g', '--onlyTransfer', action='store_true',
                        help='Only compute the transfer function step. Stops computation prior to impedance step')
    parser.add_argument('-T', '--temperature',
                        help='Specify the temperature in mK')
    parser.add_argument('-r', '--run', type=int,
                        help='Specify the run number')
    parser.add_argument('-s', '--sc', default=0,
                        help='Specify the superconducting mode bias current in uA. Default is 0')
    parser.add_argument('-n', '--normal',
                        help='Specify the normal mode bias current in uA')
    parser.add_argument('-b', '--bias',
                        help='Specify the bias mode bias current in uA')
    args = parser.parse_args()
    plotDir = '{}/run{}/T{}mK/{}'.format(args.inputDirectory, args.run, args.temperature, args.subDirectory)
    args.outputDirectory = args.outputDirectory if args.outputDirectory else plotDir
    return args


if __name__ == '__main__':
    ARGS = get_args()
    process_complex_impedance(ARGS.inputDirectory, ARGS.outputDirectory, ARGS.subDirectory, ARGS.onlyTransfer, ARGS.run, ARGS.temperature, ARGS.sc, ARGS.normal, ARGS.bias)
