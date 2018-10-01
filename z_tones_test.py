import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt

from scipy import fftpack
from scipy.signal import hann
from scipy.optimize import curve_fit


# Fitting stuff
def complex_tes_one_block(f, I, a, b, R, g, C, T, Rl, Lin):
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
    #I = 15e-6
    #R = 40e-3
    #g = 60e-12
    P = I**2 * R
    L = P*a/(g*T)
    t = C/g
    ti = t/(1-L)
    Ztes = R*(1+b) + (R*L/(1-L))*(2+b)/(1+ 1j*2*np.pi*f*ti)
    Z = Ztes + Rl + 2*np.pi*1j*f*Lin
    return Z

def tes_one_block(f, I, a, b, R, g, C, T, Rl, Lin):
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
    P = I**2 * R
    L = P*a/(g*T)
    t = C/g
    ti = t/(1-L)
    Ztes = R*(1+b) + (R*L/(1-L))*(2+b)/(1+ 1j*2*np.pi*f*ti)
    Z = Ztes + Rl + 2*np.pi*1j*f*Lin
    
    q = np.real(Z)
    q = np.append(q, np.imag(Z))
    return q

def gen_plot_points_fit(z, z_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear'):
    """Create generic plots that may be semilogx (default)
    I, a, b, R, g, C, T, Rl, Lin
    """
    I, a, b, R, g, C, T, Rl, Lin = result
    Ierr, aerr, berr, Rerr, gerr, Cerr, Terr, Rlerr, Linerr = perr
    
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
    tI = r'$I_{0} = %.5f \pm %.5f \mathrm{\mu A}$'%(I*1e6, Ierr*1e6)
    ta = r'$\alpha = %.5f \pm %.5f$'%(a, aerr)
    tb = r'$\beta = %.5f \pm %.5f$'%(b, berr)
    tR = r'$R_{0} = %.5f \pm %.5f \mathrm{m\Omega}$'%(R*1e3, Rerr*1e3)
    tg = r'$G = %.5f \pm %.5f \mathrm{pW/K}$'%(g*1e12, gerr*1e12)
    tC = r'$C = %.5f \pm %.5f \mathrm{pJ/K}$'%(C*1e12, Cerr*1e12)
    tT = r'$T_{0} = %.5f \pm %.5f \mathrm{mK}$'%(T*1e3, Terr*1e3)
    tRl = r'$R_{L} = %.5f \pm %.5f \mathrm{m\Omega}$'%(Rl*1e3, Rlerr*1e3)
    tLin = r'$L_{in} = %.5f \pm %.5f \mathrm{nH}$'%(Lin*1e9, Linerr*1e9)
    text_string = tI + '\n' + ta + '\n' + tb + '\n' + tR + '\n' + tg + '\n' + tC + '\n' + tT + '\n' + tRl + '\n' + tLin
        
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.7, 0.2, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    
    
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


def fit_z_model_and_plot(tones, z, p0=None, lbounds=None, ubounds=None, method='lm'):
    '''Helper function that attempts to fit'''
    zz = np.real(z)
    zz = np.append(zz, np.imag(z))
    print('Attempting to fit...')
    if method == 'trf':
        result, pcov = curve_fit(tes_one_block, tones, zz, p0=x0, bounds=(lbounds, ubounds), method=method, max_nfev=1e4)
    if method == 'lm':
        result, pcov = curve_fit(tes_one_block, tones, zz, p0=x0, method=method)
    perr = np.sqrt(np.diag(pcov))
    z_model = complex_tes_one_block(tones, *result)
    xlab = 'real Z [Ohms]'
    ylab = 'imag Z [Ohms]'
    title = 'Nyquist plot of Z Model'
    fName = inputDirectory + '/nyquist_zmodel_tones.png'
    gen_plot_points_fit(z, z_model, result, perr, xlab, ylab, title, fName, xlog='linear', ylog='linear')
    print('Fit and plot done')
    return result, perr, z_model


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
f0 = 16
fStep = 16 #if square, fStep = 2*f0
fMax = 50000
df = 0.1
tone_list = get_tone_list(f0, fMax, fStep)
fcut = select_tones(tone_list, data['real_freq'], df)
fcut = np.logical_and(fcut, data['real_freq'] < 25e3)
fcut = np.logical_and(fcut, data['real_freq'] > 0)
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
Zbias = 1/(2*np.pi*1j*tones*Cbias)
Rfb = 10e3
Lfb = M*M*Li
Zfb = Rfb + 2*np.pi*1j*tones*Lfb
Zfb = Rfb
Rp = 11.5e-3

#z = compute_complex_z(ratio_tones, tones, squid)
z = M*Rfb*Rsh*(ratio_tones)/Zbias - Rsh - Rp - 2*np.pi*1j*tones*Li

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
loI = 0
loa = -10
lob = -10
loR0 = 1e-3
log = 1e-15
loC = -500e-12
loT = 2e-3
loRl = 1e-4
loLin = 1e-9

lbounds = [loI, loa, lob, loR0, log, loC, loT, loRl, loLin]
#ubounds = [1, 1e6, 1e6, 1, 1, 1, 40e-3]
hiI = 50e-6
hia = 1e4
hib = 10
hiR0 = 50e-3
hig = 100e-12
hiC = 500e-12
hiT = 50e-3
hiRl = 1e-1
hiLin = 1e-5
ubounds = [hiI, hia, hib, hiR0, hig, hiC, hiT, hiRl, hiLin]
I0 = 13e-6
a0 = 20
b0 = 1
R0 = 50e-3
g0 = 60e-12
C0 = 20e-12
T0 = 33e-3
Rl0 = 33e-3
Lin0 = 6e-9
x0 = [I0, a0, b0, R0, g0, C0, T0, Rl0, Lin0]
result, perr, z_model = fit_z_model_and_plot(tones, z, p0=x0, lbounds=lbounds, ubounds=ubounds, method='trf')


###########################
# Next let's load a timeseries file and fft it ourselves

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
