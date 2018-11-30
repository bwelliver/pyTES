import argparse

import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt

from scipy import fftpack
from scipy.signal import hann
from scipy.optimize import curve_fit


# Fitting stuff
def poly_w_fit(f, a, n, x0):
    '''A simple polynomial fit in frequency'''
    return a*np.power(f, n) + x0


def complex_ratio_fit(f, Rn, Rl, L, C):
    '''Complex version of ratio fit'''
    #Rn = 0.543
    ZL = 1j*2*np.pi*f*L
    #ZLs = 1j*2*np.pi*f*Ls
    ZC = 1/(1j*2*np.pi*f*C)
    
    Z1 = 1/(1/Rn + 1/ZC) + Rl + ZL
    Z2 = Rl + ZL
    ratio = Z1/Z2
    return ratio


def ratio_fit(f, Rn, Rl, L, C):
    '''flat version of ratio fit'''
    #Rn = 0.543
    ratio = complex_ratio_fit(f, Rn, Rl, L, C)
    
    q = np.real(ratio)
    q = np.append(q, np.imag(ratio))
    return q


def complex_tes_one_block(f, a, b, C, T):
    '''Simple 1 block model for Ztes
    f = actual frequency data
    Independent parameters shall be as follows:
    I = current on TES
    a = alpha (T/R*dR/dT)
    b = beta (I/R)*dR/dI
    R = TES resistance
    g = TES thermal conductivity
    C = TES heat capacity
    T = TES temperature
    Rl = TES circuit load resistance
    Lin = TES input branch inductance
    Also to keep in mind
    L = TES loop gain
    t = TES time constant
    Z(w) = Rl + jwL + Ztes(w)
    '''
    I = 2.21e-6
    R = 0.096
    g = 0.887e-12
    
    #T = 9e-3
    P = I**2 * R
    L = P*a/(g*T)
    t = C/g
    ti = t/(1-L)
    Ztes = R*(1+b) + (R*L/(1-L))*(2+b)/(1+ 1j*2*np.pi*f*ti)
    return Ztes

def tes_one_block(f, a, b, C, T):
    '''Simple 1 block model for Ztes
    f = actual frequency data
    Independent parameters shall be as follows:
    I = current on TES
    a = alpha (T/R*dR/dT)
    b = beta (I/R)*dR/dI
    R = TES resistance
    g = TES thermal conductivity
    C = TES heat capacity
    T = TES temperature
    Also to keep in mind
    L = TES loop gain
    t = TES time constant
    '''
    #I = 15e-6
    #R = 40e-3
    #g = 60e-12
    Ztes = complex_tes_one_block(f, a, b, C, T)
    q = np.real(Ztes)
    q = np.append(q, np.imag(Ztes))
    return q


def gen_plot_points_fit(z, z_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear'):
    """Create generic plots that may be semilogx (default)
    I, a, b, R, g, C, T, Rl, Lin
    """
    a, b, C, T = result
    aerr, berr, Cerr, Terr = perr
    
    fig2 = plt.figure(figsize=(16, 16))
    ax = fig2.add_subplot(111)
    ax.plot(np.real(z), np.imag(z), marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    # Now the fit
    ax.plot(np.real(z_model), np.imag(z_model), 'r-', marker='None', linewidth=2)
    ax.set_xscale(xlog)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(ylog)
    #ax.set_ylim([-1, 1])
    #ax.set_xlim([-1, 1])
    ax.grid()
    ax.set_title(title)
    
    # Set up text strings for my fit
    #tI = r'$I_{0} = %.5f \pm %.5f \mathrm{\mu A}$'%(I*1e6, Ierr*1e6)
    ta = r'$\alpha = %.5f \pm %.5f$'%(a, aerr)
    tb = r'$\beta = %.5f \pm %.5f$'%(b, berr)
    #tR = r'$R_{0} = %.5f \pm %.5f \mathrm{m\Omega}$'%(R*1e3, Rerr*1e3)
    #tg = r'$G = %.5f \pm %.5f \mathrm{pW/K}$'%(g*1e12, gerr*1e12)
    tC = r'$C = %.5f \pm %.5f \mathrm{pJ/K}$'%(C*1e12, Cerr*1e12)
    tT = r'$T_{0} = %.5f \pm %.5f \mathrm{mK}$'%(T*1e3, Terr*1e3)
    #tRl = r'$R_{L} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rl*1e3, Rlerr*1e3)
    #tLin = r'$L_{in} = %.5f \pm %.5f \mathrm{nH}$'%(Lin*1e9, Linerr*1e9)
    #text_string = tI + '\n' + tR + '\n' + tg + '\n' + ta + '\n' + tb + '\n' + tC + '\n' + tT
    text_string = ta + '\n' + tb + '\n' + tC + '\n' + tT
        
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.7, 0.2, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    
    
    fig2.savefig(fName, dpi=200)
    #plt.show()
    #plt.draw()
    plt.close('all')
    return None


def gen_plot_points_fit_ratio(ratio, ratio_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear'):
    """Create generic plots that may be semilogx (default)
    I, a, b, R, g, C, T, Rl, Lin
    """
    Rn, Rl, Lin, Ls, C = result
    Rnerr, Rlerr, Linerr, Lserr, Cerr = perr
    
    fig2 = plt.figure(figsize=(16, 16))
    ax = fig2.add_subplot(111)
    ax.plot(np.real(ratio), np.imag(ratio), marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    # Now the fit
    ax.plot(np.real(ratio_model), np.imag(ratio_model), 'r-', marker='None', linewidth=2)
    ax.set_xscale(xlog)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(ylog)
    #ax.set_ylim([-1, 1])
    #ax.set_xlim([-1, 1])
    ax.grid()
    ax.set_title(title)
    
    # Set up text strings for my fit
    tRn = r'$R_{n} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rn*1e3, Rnerr*1e3)
    tRl = r'$R_{L} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rl*1e3, Rlerr*1e3)
    tLin = r'$L_{in} = %.5f \pm %.5f \mathrm{\mu H}$'%(Lin*1e6, Linerr*1e6)
    tLs = r'$L_{s} = %.5f \pm %.5f \mathrm{\mu H}$'%(Ls*1e6, Lserr*1e6)
    tC = r'$C_{in} = %.5f \pm %.5f \mathrm{nF}$'%(C*1e9, Cerr*1e9)
    text_string = tRn + '\n' + tRl + '\n' + tLin + '\n' + tLs + '\n' + tC
        
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.7, 0.2, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    fig2.savefig(fName, dpi=200)
    #plt.show()
    #plt.draw()
    plt.close('all')
    return None


def gen_plot_points_fit_ratio_components(tone, ratio, ratio_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear', component='real'):
    """Create generic plots that may be semilogx (default)
    I, a, b, R, g, C, T, Rl, Lin
    """
    Rn, Rl, Lin, Ls, C = result
    Rnerr, Rlerr, Linerr, Lserr, Cerr = perr
    
    fig2 = plt.figure(figsize=(16, 16))
    ax = fig2.add_subplot(111)
    if component == 'real':
        ax.plot(tone, np.real(ratio), marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
        # Now the fit
        ax.plot(tone, np.real(ratio_model), 'r-', marker='None', linewidth=2)
    if component == 'imag':
        ax.plot(tone, np.imag(ratio), marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
        # Now the fit
        ax.plot(tone, np.imag(ratio_model), 'r-', marker='None', linewidth=2)
    ax.set_xscale(xlog)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(ylog)
    #ax.set_ylim([-1, 1])
    #ax.set_xlim([-1, 1])
    ax.grid()
    ax.set_title(title)
    
    # Set up text strings for my fit
    tRn = r'$R_{n} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rn*1e3, Rnerr*1e3)
    tRl = r'$R_{L} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rl*1e3, Rlerr*1e3)
    tLin = r'$L_{in} = %.5f \pm %.5f \mathrm{\mu H}$'%(Lin*1e6, Linerr*1e6)
    tLs = r'$L_{s} = %.5f \pm %.5f \mathrm{\mu H}$'%(Ls*1e6, Lserr*1e6)
    tC = r'$C_{in} = %.5f \pm %.5f \mathrm{nF}$'%(C*1e9, Cerr*1e9)
    text_string = tRn + '\n' + tRl + '\n' + tLin + '\n' + tLs + '\n' + tC
        
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.7, 0.2, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    fig2.savefig(fName, dpi=200)
    #plt.show()
    #plt.draw()
    plt.close('all')
    return None


def gen_plot_points_fit_poly(tone, y, y_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear'):
    """Create generic plots that may be semilogx (default)
    I, a, b, R, g, C, T, Rl, Lin
    """
    #Rn, Rl, Lin, Lsin, Lsin2 = result
    #Rnerr, Rlerr, Linerr, Lsinerr, Lsinerr2 = perr
    
    fig2 = plt.figure(figsize=(16, 16))
    ax = fig2.add_subplot(111)
    ax.plot(tone, y, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    # Now the fit
    ax.plot(tone, y_model, 'r-', marker='None', linewidth=2)
    ax.set_xscale(xlog)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(ylog)
    #ax.set_ylim([-1, 1])
    #ax.set_xlim([-1, 1])
    ax.grid()
    ax.set_title(title)
    
    # Set up text strings for my fit
    #tRn = r'$R_{n} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rn*1e3, Rnerr*1e3)
    #tRl = r'$R_{L} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rl*1e3, Rlerr*1e3)
    #tLin = r'$L_{in} = %.5f \pm %.5f \mathrm{\mu H}$'%(Lin*1e6, Linerr*1e6)
    #tLsin = r'$L_{sin} = %.5f \pm %.5f \mathrm{\mu H}$'%(Lsin*1e6, Lsinerr*1e6)
    #tLsin2 = r'$C_{sin} = %.5f \pm %.5f \mathrm{nF}$'%(Lsin2*1e9, Lsinerr2*1e9)
    #text_string = tRn + '\n' + tRl + '\n' + tLin + '\n' + tLsin + '\n' + tLsin2
        
    #props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #ax.text(0.7, 0.2, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    fig2.savefig(fName, dpi=200)
    #plt.show()
    #plt.draw()
    plt.close('all')
    return None


def gen_plot_points(x, y, xlab, ylab, title, fName, xlim=None, ylim=None, xlog='linear', ylog='linear', figSize=(16,16)):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=figSize)
    ax = fig2.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xscale(xlog)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(ylog)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title)
    fig2.savefig(fName, dpi=150)
    #plt.show()
    #plt.draw()
    plt.close('all')
    return None


def get_data(filename):
    '''Load a fast digitizer lvm and get signal and response columns
    The first few lines are header so we will skip them
    But also we can have more than one "end of header" string so search backwards.
    This is longer but won't make copies of the array. Note that data starts 2 lines after end of header is printed
    '''
    print('The filename is: {}'.format(filename))
    with open(filename, 'r') as f:
        lines = f.readlines()
    eoh = 0
    for index, line in reversed(list(enumerate(lines))):
        if line.find('***End_of_Header') > -1:
            eoh = index
            break
    print('End of header line at: {} and the line is: {}'.format(eoh, lines[eoh]))
    f = []
    Z = []
    for line in lines[eoh+2:]:
        line = line.strip('\n').split('\t')
        f.append(float(line[0]))
        Z.append(float(line[1]))
    return np.asarray(f), np.asarray(Z)


def get_time_data(filename):
    '''Load a fast digitizer lvm and get signal and response columns
    The first few lines are header so we will skip them
    But also we can have more than one "end of header" string so search backwards.
    This is longer but won't make copies of the array. Note that data starts 2 lines after end of header is printed
    '''
    print('The filename is: {}'.format(filename))
    with open(filename, 'r') as f:
        lines = f.readlines()
    eoh = 0
    for index, line in reversed(list(enumerate(lines))):
        if line.find('***End_of_Header') > -1:
            eoh = index
            break
    print('End of header line at: {} and the line is: {}'.format(eoh, lines[eoh]))
    t = []
    vIn = []
    vOut = []
    for line in lines[eoh+2:]:
        line = line.strip('\n').split('\t')
        t.append(float(line[0]))
        vIn.append(float(line[1]))
        vOut.append(float(line[2]))
    return np.asarray(t), np.asarray(vIn), np.asarray(vOut)


def parse_lvm_file(inFile, intype='freq'):
    '''Function to parse a LVM file containing complex Z data
    There are two bits of information needed: the frequency and the data
    Frequency can be obtained from file name via the pattern '*_frequencyHz*'
    '''
    #frequency = re.search('[0-9]*Hz', inFile).group(0).replace('Hz','')
    # Next we need to load the data into a numpy array
    if intype == 'freq':
        freq, Z = get_data(inFile)
    # Now construct into a dictionary and return it
    #lvm_dictionary = {'freq': time, 'fV': v_in}
        return freq, Z
    elif intype == 'time':
        t, vIn, vOut = get_time_data(inFile)
        return t, vIn, vOut
    return None


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


def compute_fft(data):
    '''Given an input data dictionary of time, v_in, v_out, compute the fft info'''
    wIn = hann(data['time'].size)
    wOut = hann(data['time'].size)
    freq = time_to_freq(data['time'])
    fv_in = fftpack.fft(data['v_in']*wIn)
    fv_out = fftpack.fft(data['v_out']*wOut)
    return {'frequency': freq, 'fv_in': fv_in, 'fv_out': fv_out}


def get_data_from_file(inputDirectory, real_suffix=None, imag_suffix=None, timeseries_suffix=None):
    '''Function that grabs data from a file, both imaginary and real frequency responses, and timeseries if they exist'''
    # Obtain the desired type of data
    data = {}
    if real_suffix is not None:
        refreq, reRatio = parse_lvm_file(inputDirectory + '/' + real_suffix)
        data['real_freq'] = refreq
        data['reRatio'] = reRatio
    if imag_suffix is not None:
        imagfreq, imRatio = parse_lvm_file(inputDirectory + '/' + imag_suffix)
        data['imag_freq'] = imagfreq
        data['imRatio'] = imRatio
    if timeseries_suffix is not None:
        t, vIn, vOut = parse_lvm_file(inputDirectory + '/' + timeseries_suffix, intype='time')
        data['time'] = t
        data['v_in'] = vIn
        data['v_out'] = vOut
    return data


def invert_ratio(data, invert=False):
    '''Function that will invert the real and imaginary parts of the complex frequency response'''
    if invert == True:
        ratio = data['reRatio'] + 1j*data['imRatio']
        ratio = 1/ratio
        data['reRatio'] = ratio.real
        data['imRatio'] = ratio.imag
    return data


def get_tone_list(base_tone, max_frequency, tone_step_size):
    '''Generate a list of desired tones'''
    # Note that if we have a multi-tone sweep we have harmonics of the form:
    # f0 + n*fStep < max_frequency
    # If we have a square wave only the odd harmonics of the base tone are to be selected:
    # f0 + (2n)*f0 < max_frequency
    # So here we have fStep = 2*f0
    num_tones = int((max_frequency - base_tone)/tone_step_size)
    tone_list = [base_tone + n*tone_step_size for n in range(num_tones)]
    return tone_list


def select_tones(tone_list, frequency, df):
    '''Create a cut to select only the desired tones from the frequency sweep'''
    frequency_cut = np.zeros(frequency.size, dtype=bool)
    for f in tone_list:
        loop_cut = np.logical_and(frequency >= f - df, frequency <= f + df)
        frequency_cut = np.logical_or(frequency_cut, loop_cut)
    return frequency_cut


def fit_z_model_and_plot(inputDirectory, tones, z, p0=None, lbounds=None, ubounds=None, method='lm'):
    '''Helper function that attempts to fit'''
    zz = np.real(z)
    zz = np.append(zz, np.imag(z))
    print('Attempting to fit...')
    if method == 'trf':
        print('The initial guess is {}'.format(p0))
        if lbounds is not None or ubounds is not None:
            result, pcov = curve_fit(tes_one_block, tones, zz, p0=p0, bounds=(lbounds, ubounds), method=method, max_nfev=1e4)
        else:
            result, pcov = curve_fit(tes_one_block, tones, zz, p0=p0, method=method, max_nfev=1e4)
    if method == 'lm':
        result, pcov = curve_fit(tes_one_block, tones, zz, p0=p0, method=method)
    perr = np.sqrt(np.diag(pcov))
    z_model = complex_tes_one_block(tones, *result)
    xlab = 'real Z [Ohms]'
    ylab = 'imag Z [Ohms]'
    title = 'Nyquist plot of Z Model'
    fName = inputDirectory + '/nyquist_zmodel_tones.png'
    gen_plot_points_fit(z, z_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear')
    print('Fit and plot done')
    return result, perr, z_model

def fit_ratio_model_and_plot(inputDirectory, tones, ratio, p0=None, lbounds=None, ubounds=None, method='lm'):
    '''Helper function that attempts to fit ratio model for transfer function'''
    ratioratio = np.real(ratio)
    ratioratio = np.append(ratioratio, np.imag(ratio))
    print('Attempting to fit...')
    if method == 'trf':
        result, pcov = curve_fit(ratio_fit, tones, ratioratio, p0=p0, bounds=(lbounds, ubounds), method=method, max_nfev=1e4)
    if method == 'lm':
        result, pcov = curve_fit(ratio_fit, tones, ratioratio, p0=p0, method=method)
    perr = np.sqrt(np.diag(pcov))
    if len(result) < 3:
        Rn = 0.543
        Rnerr = 0
        result = [Rn, result[0], result[1]]
        perr = [Rnerr, perr[0], perr[1]]
    #result = [0.547, 32.5e-3, 2.089e-7]
    ratio_model = complex_ratio_fit(tones, *result)
    # Insert missing things
    result = [result[0], result[1], result[2], 0, result[3]]
    perr = [perr[0], perr[1], perr[2], 0, result[3]]
    print('The values for the fit are: Rn = {} mOhm, Rl = {} mOhm, L = {} uH, Ls = {} uH, C = {} nF'.format(result[0]*1e3, result[1]*1e3, result[2]*1e6, result[3]*1e6, result[4]*1e9))
    xlab = 'Real Zn/Zsc'
    ylab = 'Imag Zn/Zsc'
    title = 'Nyquist plot of Ratio Model'
    fName = inputDirectory + '/nyquist_ratio_model_tones.png'
    gen_plot_points_fit_ratio(ratio, ratio_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear')
    
    # Make component plots
    xlab = 'Frequency [Hz]'
    ylab = 'Real Part of Impedance Ratio'
    title = 'Real Plot of Ratio Model'
    fName = inputDirectory + '/real_ratio_model_tones.png'
    gen_plot_points_fit_ratio_components(tones, ratio, ratio_model, result, perr, xlab, ylab, title, fName, xlog='log', ylog='linear', component='real')
    
    xlab = 'Frequency [Hz]'
    ylab = 'Imag Part of Impedance Ratio'
    title = 'Imag Plot of Ratio Model'
    fName = inputDirectory + '/imag_ratio_model_tones.png'
    gen_plot_points_fit_ratio_components(tones, ratio, ratio_model, result, perr, xlab, ylab, title, fName, xlog='log', ylog='linear', component='imag')
    
    print('Fit and plot done')
    return result, perr, ratio_model


runType = 'transfer'

def compute_z(G, Rn, Rl, L):
    '''Compute the complex impedance given the transfer function'''
    inputDirectory = '/Users/bwelliver/cuore/bolord/complex_z/test_sd'
    imag_suffix = 'BaseTemperature_Imag.lvm'
    real_suffix = 'BaseTemperature_Real.lvm'
    # Get the data for each type and we will combine it
    data = get_data_from_file(inputDirectory, real_suffix, imag_suffix)

    # If the ratio is not representative of fVin/fVout let's invert it so it is
    data = invert_ratio(data, True)

    xlab = 'Real Ratio'
    ylab = 'Imaginary Ratio'
    title = 'Nyquist plot of fVin/fVout'
    fName = inputDirectory + '/nyquist_ratio_test.png'
    gen_plot_points(data['reRatio'], data['imRatio'], xlab, ylab, title, fName, xlog='linear', ylog='linear')
    # Now we need to filter the appropriate tones
    f0 = 7
    fStep = 7 #if square, fStep = 2*f0
    fMax = 30.0020e3
    df = 0.1
    tone_list = get_tone_list(f0, fMax, fStep)
    fcut = select_tones(tone_list, data['real_freq'], df)
    fcut = np.logical_and(fcut, data['real_freq'] < 10e3)
    fcut = np.logical_and(fcut, data['real_freq'] > 0)
    # Now select only specific indices.
    df = data['real_freq'][1]
    base_index = int(f0/df)
    index_step = int(fStep/df)
    nRange = int((fMax-f0)/fStep)
    index_list = [base_index + i*index_step for i in range(nRange)]
    fcut[index_list] = True
    # OK now select the desired frequencies from our total cut (un-meaned)
    tones = data['real_freq'][fcut]
    ratio_tones = data['reRatio'][fcut] + 1j*data['imRatio'][fcut]
    # attempt to correct T(w) to Z(w)

    xlab = 'Real Ratio'
    ylab = 'Imag Ratio'
    title = 'Nyquist plot of Frequency Response'
    fName = inputDirectory + '/nyquist_ratio_tones.png'
    xlim=[-0.1, 0.15]
    ylim = [-0.01, 0.15]
    gen_plot_points(ratio_tones.real, ratio_tones.imag, xlab, ylab, title, fName, xlim=None, ylim=None, xlog='linear', ylog='linear')


    # SQUID Parameters
    Rsh = 21e-3
    Li = 6e-9
    M = -1.27664
    Rbias = 10e3
    Cbias = 100e-12
    Lbias = 1e-3
    Zbias = Rbias + 1/(1j*2*np.pi*tones*Cbias) # This is impedance to GROUND!!!!!!
    Zbias = Rbias
    Rfb = 10e3
    Lfb = M*M*Li
    Zfb = Rfb + 2*np.pi*1j*tones*Lfb
    Zfb = Rfb
    Rp = 11.5e-3
    G_tones = G[fcut]
    #z = compute_complex_z(ratio_tones, tones, squid)
    z = (M*Rfb*Rsh*(ratio_tones)/Zbias)/G_tones - Rl - 2*np.pi*1j*tones*L

    # Plot Things
    xlab = 'Real Z [Ohms]'
    ylab = 'Imag Z [Ohms]'
    title = 'Nyquist plot of Z'
    fName = inputDirectory + '/nyquist_z_tones.png'
    xlim=[-0.1, 0.15]
    ylim = [-0.01, 0.15]
    #gen_plot_points((1/cZ_tones).real, (1/cZ_tones).imag, xlab, ylab, title, fName, log='linear')
    gen_plot_points(z.real, z.imag, xlab, ylab, title, fName, xlim=None, ylim=None, xlog='linear', ylog='linear')

    # Make fft plots
    xlab = 'Frequency [Hz]'
    ylab = 'Abs Z [Ohm]'
    title = 'Power spectrum of Z'
    fName = inputDirectory + '/psd_z_tones.png'
    gen_plot_points(tones, np.abs(z), xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Real Z [Ohm]'
    title = 'Power spectrum of Real Z'
    fName = inputDirectory + '/psd_real_z_tones.png'
    gen_plot_points(tones, z.real, xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Im Z [Ohm]'
    title = 'Power spectrum of Im Z'
    fName = inputDirectory + '/psd_imag_z_tones.png'
    gen_plot_points(tones, z.imag, xlab, ylab, title, fName, xlog='log', ylog='linear')


    # Attempt to fit
    # f, I, a, b, R0, g, C, T, Rl, Lin
    loI = 1e-6
    loR = 21e-3
    loG = 1e-12
    loa = 1
    lob = 1e-3
    loC = 1e-12
    loT = 7e-3

    lbounds = [loa, lob, loC, loT]
    #ubounds = [1, 1e6, 1e6, 1, 1, 1, 40e-3]
    hiI = 15e-6
    hiR = 200e-3
    hiG = 1e-9
    hia = 500
    hib = 10
    hiC = 500e-12
    hiT = 34e-3
    ubounds = [hia, hib, hiC, hiT]

    I0 = 2.21e-6
    R0 = 100e-3
    G0 = 55e-12
    a0 = 20
    b0 = 1
    C0 = 20e-12
    T0 = 9e-3
    x0 = [a0, b0, C0, T0]
    result, perr, z_model = fit_z_model_and_plot(inputDirectory, tones, z, p0=x0, lbounds=lbounds, ubounds=ubounds, method='trf')


###########################
# Next let's load a timeseries file and fft it ourselves
if runType == 'all' or runType == 'time':
    inputDirectory = '/Users/bwelliver/cuore/bolord/complex_z/test_sd'
    timeseries_suffix = 'BaseTemperature_time.lvm'
    data = get_data_from_file(inputDirectory, real_suffix=None, imag_suffix=None, timeseries_suffix=timeseries_suffix)

    fdata = compute_fft(data)

    refreq = fdata['frequency']
    ratio = fdata['fv_in']/fdata['fv_out']

    # Now these are the same for the freq but the components should be all here so let us plot
    xlab = 'Real Ratio'
    ylab = 'Imag Z'
    title = 'Nyquist plot of fVin/fVout'
    fName = inputDirectory + '/nyquist_ratio_test_fromTime.png'
    gen_plot_points(ratio.real, ratio.imag, xlab, ylab, title, fName, xlog='linear', ylog='linear')

    # Now we need to filter the appropriate tones
    f0 = 16
    fStep = 16 #if square, fStep = 2*f0
    fMax = 50000
    df = 0.1
    tone_list = get_tone_list(f0, fMax, fStep)
    fcut = select_tones(tone_list, refreq, df)
    fcut = np.logical_and(fcut, refreq < 25e3)
    fcut = np.logical_and(fcut, refreq > 0)

    # OK now select the desired frequencies from our total cut (un-meaned)
    tones = refreq[fcut]
    ratio_tones = ratio[fcut]

    # SQUID Parameters
    Rsh = 21e-3
    Li = 6e-9
    M = -1.27664
    Rbias = 10e3
    Cbias = 100e-12
    Lbias = 1e-3
    Zbias = Rbias + 1/(1j*2*np.pi*tones*Cbias) # This is impedance to GROUND!!!!!!
    Zbias = Rbias
    Rfb = 10e3
    Lfb = M*M*Li
    Zfb = Rfb + 2*np.pi*1j*tones*Lfb
    Zfb = Rfb
    Rp = 11.5e-3

    z = M*Rfb*Rsh*(ratio_tones)/Zbias - Rsh - Rp - 2*np.pi*1j*tones*Li

    xlab = 'real Z [Ohms]'
    ylab = 'imag Z [Ohms]'
    title = 'Nyquist plot of Z'
    fName = inputDirectory + '/nyquist_z_tones_fromTime.png'
    #gen_plot_points((cZ_tones).real, (1/cZ_tones).imag, xlab, ylab, title, fName, log='linear')
    gen_plot_points(z.real, z.imag, xlab, ylab, title, fName, xlim=xlim, ylim=ylim, xlog='linear', ylog='linear')

    # Make fft plots
    xlab = 'Frequency [Hz]'
    ylab = 'Abs Z [Ohm]'
    title = 'Power spectrum of Z'
    fName = inputDirectory + '/psd_z_tones_fromTime.png'
    gen_plot_points(tones, np.abs(z), xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Real Z [Ohm]'
    title = 'Power spectrum of Real Z'
    fName = inputDirectory + '/psd_real_z_tones_fromTime.png'
    gen_plot_points(tones, z.real, xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Im Z [Ohm]'
    title = 'Power spectrum of Im Z'
    fName = inputDirectory + '/psd_imag_z_tones_fromTime.png'
    gen_plot_points(tones, z.imag, xlab, ylab, title, fName, xlog='log', ylog='linear')

def compute_transfer_function(input_directory, output_directory, frequency_parameters, invert=True)
    '''Compute the transfer function'''
    ############# Let us look at the transfer function
    
    # TODO: Make this more automatic
    sc_imag_suffix = 'BaseTemperature_Imag_sc.lvm'
    sc_real_suffix = 'BaseTemperature_Real_sc.lvm'
    # Get the data for each type and we will combine it
    data = get_data_from_file(inputDirectory, sc_real_suffix, sc_imag_suffix)
    # If the ratio is not representative of fVin/fVout let's invert it so it is
    scdata = invert_ratio(data, invert)
    n_imag_suffix = 'BaseTemperature_Imag_n.lvm'
    n_real_suffix = 'BaseTemperature_Real_n.lvm'
    # Get the data for each type and we will combine it
    data = get_data_from_file(inputDirectory, n_real_suffix, n_imag_suffix)
    # If the ratio is not representative of fVin/fVout let's invert it so it is
    ndata = invert_ratio(data, True)

    # Now we have sc and normal data frequency response.
    # The following is trueish:
    # (Zmeas,n)/(Zmeas,sc) = Ratio = (Rn + Rl + jwL)/(Rl + jwL)
    # G(w) = (Zmeas,n)/(Rn + Rl + jwL)
    # We will fit the ratio to get values for Rl and L
    # In theory we can just take the ratio because only the DC bit will be special
    ratio = (ndata['reRatio'] + 1j*ndata['imRatio'])/(scdata['reRatio'] + 1j*scdata['imRatio'])
    # Get specific tones
    # Now we need to filter the appropriate tones
    f0 = frequency_parameters['base']
    fStep = frequency_parameters['step'] #if square, fStep = 2*f0
    fMax = frequency_parameters['maximum']
    df = frequency_parameters['df']
    tone_list = get_tone_list(f0, fMax, fStep)
    fcut = select_tones(tone_list, ndata['real_freq'], df)
    # TODO fix this part someday with better precision so we can go above 10 kHz
    fcut = np.logical_and(fcut, ndata['real_freq'] < 10e3)
    fcut = np.logical_and(fcut, ndata['real_freq'] > 0)
    # Now select only specific indices...work around for precision issues in logfile
    df = ndata['real_freq'][1]
    base_index = int(f0/df)
    index_step = int(fStep/df)
    nRange = int((fMax-f0)/fStep)
    index_list = [base_index + i*index_step for i in range(nRange)]
    fcut[index_list] = True
    # OK now select the desired frequencies from our total cut (un-meaned)
    tones = ndata['real_freq'][fcut]
    ratio_tones = ratio[fcut]
    # Generate diagnostic plots of the ratios prior to fitting?
    # Plot the ratios
    xlab = 'Frequency [Hz]'
    ylab = 'Abs Zn/Zsc'
    title = 'Power spectrum of Measured Impedance Ratio'
    fName = inputDirectory + '/psd_znzsc_tones.png'
    gen_plot_points(tones, np.abs(ratio_tones), xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Real Zn/Zsc'
    title = 'Power spectrum of Real Measured Impedance Ratio'
    fName = inputDirectory + '/psd_real_znzsc_tones.png'
    gen_plot_points(tones, ratio_tones.real, xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Im Zn/Zsc'
    title = 'Power spectrum of Im Measured Impedance Ratio'
    fName = inputDirectory + '/psd_imag_znzsc_tones.png'
    gen_plot_points(tones, ratio_tones.imag, xlab, ylab, title, fName, xlog='log', ylog='linear')
    
    # Attempt to fit
    x0 = [0.54, 33e-3, 0.8e-6, 1e-12]
    lbounds = [0.52, 21e-3, 0, 0]
    ubounds = [0.58, 100e-3, 1000e-6, 1000e-6]
    result, perr, ratio_model = fit_ratio_model_and_plot(inputDirectory, tones, ratio_tones, p0=x0, lbounds=lbounds, ubounds=ubounds, method='trf')
    
    # Now we can create G(w)
    Rn, Rl, L = result[0], result[1], result[2]
    Zcirc_normal = Rn + Rl + 2*1j*np.pi*ndata['real_freq']*L
    Zbias = 10000
    Rf = 10000
    M = -1.27664
    Rs = 21e-3
    dc_factor = (Rs*Rf*M)/Zbias
    G = (ndata['reRatio'] + 1j*ndata['imRatio'])*dc_factor/Zcirc_normal
    G_tones = G[fcut]
    
    
    xlab = 'Real G(w)'
    ylab = 'Im G(w)'
    title = 'Nyquist plot of G(w)'
    fName = inputDirectory + '/nyquist_g_tones.png'
    gen_plot_points(G_tones.real, G_tones.imag, xlab, ylab, title, fName, xlim=None, ylim=None, xlog='linear', ylog='linear')

    # Make fft plots
    xlab = 'Frequency [Hz]'
    ylab = 'Abs G'
    title = 'Power spectrum of Abs(G)'
    fName = inputDirectory + '/psd_g_tones.png'
    gen_plot_points(tones, np.abs(G_tones), xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Real G'
    title = 'Power spectrum of Real G'
    fName = inputDirectory + '/psd_real_g_tones.png'
    gen_plot_points(tones, G_tones.real, xlab, ylab, title, fName, xlog='log', ylog='linear')

    xlab = 'Frequency [Hz]'
    ylab = 'Im G'
    title = 'Power spectrum of Im G'
    fName = inputDirectory + '/psd_imag_g_tones.png'
    gen_plot_points(tones, G_tones.imag, xlab, ylab, title, fName, xlog='log', ylog='linear')
    
    return G, Rn, Rl, L
if runType == 'transfer' or runType == 'both':
    G, Rn, Rl, L = compute_transfer_function()
    if runType == 'both':
        compute_z(G, Rn, Rl, L)

        
def process_complex_impedance(indir, outdir, only_transfer):
    '''Main function that implements complex impedance computations'''
    
    # Step 1: Generate the transfer function if we don't have it already
    G, Rn, Rl, L = compute_transfer_function(input_directory=indir, output_directory=outdir)
    
    # Step 2: Compute the complex impedance
    compute_z(input_directory=indir, output_directory=outdir, transfer_function=G, normalR=Rn, loadR=Rl, inL=L)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDirectory', help='Specify the full path of the directory that contains all the files you wish to use')
    parser.add_argument('-o', '--outputDirectory', help='Specify the full path of the output directory to put plots and root files. If it is not a full path, a plots and root subdirectory will be added in the input directory')
    parser.add_argument('-T' '--onlyTransfer', action='store_true', help='Only compute the transfer function step. Stops computation prior to impedance step')
    
    args = parser.parse_args()
    outPath = args.outputDirectory if args.outputDirectory else args.inputDirectory
    
    process_complex_impedance(args.inputDirectory, args.outputDirectory, args.onlyTransfer)
    