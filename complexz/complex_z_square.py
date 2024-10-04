import glob
from os.path import isabs
from os.path import dirname
from os.path import basename
import socket
import getpass
import argparse
import re
import datetime
import time

import ROOT as rt
import numpy as np
import pandas as pan

from scipy import fftpack
from scipy.signal import flattop
from scipy.signal import hann
from scipy.signal import boxcar
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import minimize


from readROOT import readROOT
from writeROOT import writeROOT
import cloneROOT as cr

import matplotlib as mp
from matplotlib import pyplot as plt


# Overload
nsum = np.sum
ln = np.log
eps = np.finfo(float).eps


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
    
    P = I**2 * R
    L = P*a/(g*T)
    t = C/g
    ti = t/(1-L)
    Ztes = R*(1+b) + (R*L/(1-L))*(2+b)/(1+ 1j*2*np.pi*f*ti)
    Z = Ztes + Rl + 2*np.pi*1j*f*Lin
    
    q = np.real(Z)
    q = np.append(q, np.imag(Z))
    return q


def residuals(params, f, Z):
    '''Residuals'''
    P, a, b, R, g, C, T = params
    diff = tes_one_block(f, P, a, b, R, g, C, T) - Z
    z1d = diff.real
    z1d = np.append(z1d, diff.imag)
    return z1d


def nll_tes_one_block(params, f, Z):
    '''A fit function for the tes one block'''
    I, a, b, R, g, C, T, Rl, Lin = params
    if I <= 0 or a <= 0 or b <= 0 or R <= 1e-3 or g <= 0 or C <= 0 or T <= 0 or Rl <= 0 or Lin <= 0:
        return np.inf
    model = tes_one_block(f, I, a, b, R, g, C, T, Rl, Lin)
    # This is complex values so it is a little tricky
    #lnl = -nsum(ln(model + np.min(model) + eps))
    lnl = nsum((model - Z)**2)
    #print("Neg LL: {}".format(lnl))
    return lnl


def get_squid_parameters(channel):
    '''SQUID Parameters based on a given channel'''
    
    squid_dictionary = {2: {
        'Li': 6e-9,
        'Min': 1/26.062,
        'Mf': 1/33.27,
        'Rfb': 1e4,
        'Rsh': 21e-3,
        'Rbias': 1e4,
        'Cbias': 150e-12
    }}
    squid_dictionary[2]['M'] = -squid_dictionary[2]['Min']/squid_dictionary[2]['Mf']
    squid_dictionary[2]['Lf'] = squid_dictionary[2]['M']*squid_dictionary[2]['M']*squid_dictionary[2]['Li']
    return squid_dictionary[channel]


def gen_plot_line(x, y, xlab, ylab, title, fName, logx='log', logy='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(16, 9))
    ax = fig2.add_subplot(111)
    #ax.plot(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', linewidth=1)
    ax.plot(x, y, linestyle='-', linewidth=0.5)
    ax.set_xscale(logx)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(logy)
    ax.set_title(title)
    ax.grid(True)
    fig2.savefig(fName, dpi=100)
    plt.close('all')
    return None


def gen_plot_points(x, y, xlab, ylab, title, fName, log='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(16, 16))
    ax = fig2.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(log)
    ax.set_ylim([-1, 1])
    ax.set_xlim([-1, 1])
    ax.grid()
    ax.set_title(title)
    fig2.savefig(fName, dpi=100)
    #plt.show()
    #plt.draw()
    plt.close('all')
    return None


def gen_plot_points_fit(z, z_model, result, perr, xlab, ylab, title, fName, log='log'):
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
    ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(log)
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


def get_vTES(iBias, vOut, Rfb, M, Rsh):
    '''computes the TES voltage in Volts'''
    vTES = Rsh*(iBias - vOut/Rfb/M)
    return vTES


def get_iTES(vOut, Rfb, M):
    '''Computes the TES current and TES current RMS in Amps'''
    iTES = vOut/Rfb/M
    return iTES


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
    

def getFFT(x):
    '''Compute the fft of the quantities for use in complex impedence
    N = number of samples
    T = sample spacing
    1/2T will be frequency spacing
    '''
    N = x.size
    return N


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
    t = []
    v_in = []
    v_out = []
    for line in lines[eoh+2:]:
        line = line.strip('\n').split('\t')
        t.append(float(line[0]))
        v_in.append(float(line[1]))
        v_out.append(float(line[3]))
    return np.asarray(t), np.asarray(v_in), np.asarray(v_out)


def parse_lvm_file(inFile):
    '''Function to parse a LVM file containing complex Z data
    There are two bits of information needed: the frequency and the data
    Frequency can be obtained from file name via the pattern '*_frequencyHz*'
    '''
    #frequency = re.search('[0-9]*Hz', inFile).group(0).replace('Hz','')
    # Next we need to load the data into a numpy array
    time, v_in, v_out = get_data(inFile)
    # Now construct into a dictionary and return it
    lvm_dictionary = {'time': time, 'v_in': v_in, 'v_out': v_out}
    return lvm_dictionary


def get_unique_currents(file_list):
    '''Given a list of files obtain the unique set of current values'''
    list_of_currents = []
    for file in file_list:
        list_of_currents.append(re.search('([-\d.]+uA)', file).group(0).replace('uA', ''))
    # strings are hashable so the old list->set->list trick works
    list_of_currents = list(set(list_of_currents))
    return list_of_currents


def compute_fft(data):
    '''Given an input data dictionary of time, v_in, v_out, compute the fft info'''
    wIn = hann(data['time'].size)
    wOut = hann(data['time'].size)
    freq = time_to_freq(data['time'])
    fv_in = fftpack.fft(data['v_in']*wIn)
    fv_out = fftpack.fft(data['v_out']*wOut)
    return {'frequency': freq, 'fv_in': fv_in, 'fv_out': fv_out}


def get_current_data(input_directory, current):
    '''For a current, process partials'''
    list_of_partials = glob.glob(input_directory + '/*' + '_' + current + 'uA*.lvm')
    # Now extract data from the partial file
    average_fft_dictionary = {}
    for index, partial in list(enumerate(list_of_partials)):
        lvm_dictionary = parse_lvm_file(partial)
        fft_dictionary = compute_fft(lvm_dictionary)
        for key, value in fft_dictionary.items():
            if index == 0:
                average_fft_dictionary[key] = value
            else:
                average_fft_dictionary[key] += value
    # Now simply divide by the number of partials
    for key in average_fft_dictionary.keys():
        average_fft_dictionary[key] = average_fft_dictionary[key]/(index+1)
    # Now in principle this dictionary contains the average we want so return it
    return average_fft_dictionary
            
        
    


def get_frequency_data(inputDirectory):
    '''Function to parse LVM files in a directory
    Z_7.2uA_32mK_Square_17Hz_20s_p01.lvm
    File names should have specific keys to parse
    '_juA_TmK_Square_fSqHz_tAcqs_pn.lvm'
    '''
    # That is, should have a current, time length, temperature, and partial number.
    # For a given directory probably only current and partials should change. Since data size can be large we should
    # process one current at a time. As we load the partials for a current, process them into FFT and average these together
    # The end result is a dictionary with key of current, and contents of FFT data.
    
    list_of_files = glob.glob(inputDirectory + '/' + '*.lvm')
    # From list of files get list of currents
    list_of_currents = get_unique_currents(list_of_files)
    # Next we want to process all files for a specific current, one at a time
    current_dictionary = {}
    for current in list_of_currents:
        current_dictionary[current] = get_current_data(inputDirectory, current)
    # At this point we should have a data dictionary with keys = strings of current
    return current_dictionary


def compute_complex_z(fft_data, squid_data):
    '''Compute complex impedence based on fft transformed data and squid parameters'''
    # Zbias = Rb - j/wC
    #Zbias = squid_data['Rbias'] - 1j/(2*np.pi*fft_data['frequency']*squid_data['Cbias'])
    Zbias = squid_data['Rbias'] / (1 + 2*np.pi * fft_data['frequency'] * 1j * squid_data['Rbias'] * squid_data['Cbias'])
    #Zbias = squid_data['Rbias']
    Zfb = squid_data['Rfb'] + 2*np.pi*1j * fft_data['frequency'] * squid_data['Lf']
    # M*Rfb/vout = 1/iTES and Rsh/Zbias is ?
    z = (fft_data['fv_in']/fft_data['fv_out'])*(squid_data['M']  * Zfb * squid_data['Rsh'])/(Zbias)
    #z = fft_data['fv_in']/(squid_data['Rbias'] * (fft_data['fv_out']/squid_data['Rfb']/squid_data['M'])) - squid_data['Rsh'] - 1j*2*np.pi*fft_data['frequency']*squid_data['Li']
    return z


def save_z_to_root(output_directory, f_dictionary, z_dictionary, data_dictionary):
    '''Function to save our frequency and complex impedence data to root file
    Here we will let the bias currents be the name of TTrees
    Each Tree will contain 3 branches: 'Frequency', 'ReZ', 'ImZ'
    dict = {'TDirectory': {
                'topLevel': {},
                'newDir': {
                    'TTree': {
                        'newTree': {
                            'TBranch': {
                                'branchA': branchA'
                                }
                            }
                        }
                    }
                }
            'TTree': {
                'tree1': {
                    'TBranch': {
                        'branch1': branch1, 'branch2': branch2
                        }
                    }
                }
            }
    '''
    data = {'TDirectory': {'computed_z': {'TTree': {}}, 'fft': {'TTree': {}}}}
    #data = {'TTree': {}}
    for current in f_dictionary.keys():
        data['TDirectory']['computed_z']['TTree'][current] = {'TBranch': {'Frequency': f_dictionary[current], 'ReZ': np.real(z_dictionary[current]), 'ImZ': np.imag(z_dictionary[current])}}
        data['TDirectory']['fft']['TTree'][current] = {'TBranch': {'Frequency': data_dictionary[current]['frequency'], 'ReVin': np.real(data_dictionary[current]['fv_in']), 'ImVin': np.imag(data_dictionary[current]['fv_in']), 'ReVout': np.real(data_dictionary[current]['fv_out']), 'ImVout': np.imag(data_dictionary[current]['fv_out']) }}
    print('data dictionary for root file is: {}'.format(data))
    outFile = output_directory + '/root/complexZ.root'
    writeROOT(outFile, data)
    return None


def do_complex_z(inputDirectory, outputDirectory, makeRoot):
    '''
    Main function for the complex impedence computation routine. This function will go through and 
    generate a complex impedence plot for the data in a given directory.
    '''

    # Based on input directory we need to parse into a dictionary we can use
    data_dictionary = get_frequency_data(inputDirectory)
    squid_data = get_squid_parameters(2)
    z_dictionary = {}
    f_dictionary = {}
    # Next we need to parse the data into something more useful. For a given current let us perform an FFT and make a figure
    # The value for a given current key is a dictionary of fft quantities {frequncy: .., fv_in: .., fv_out: ..}
    
    for current_key, data in data_dictionary.items():
        fcut = data['frequency'] >= 0
        
        gen_plot_line(data['frequency'][fcut], np.abs(data['fv_in'][fcut]), 'f (Hz)', 'Input Voltage fft', 'Input Voltage FFT', '/Users/bwelliver/cuore/bolord/complex_z/test/vin_fft_' + current_key + 'uA_log.png', logx='log', logy='log')
        
        gen_plot_line(data['frequency'][fcut], np.abs(data['fv_out'][fcut]), 'f (Hz)', 'Output Voltage fft', 'Output Voltage FFT', '/Users/bwelliver/cuore/bolord/complex_z/test/vout_fft_' + current_key + 'uA_log.png', logx='log', logy='log')
        
        z = compute_complex_z(data, squid_data)
        
        gen_plot_line(data['frequency'][fcut], np.abs(z[fcut]), 'f (Hz)', 'Abs Complex Z', 'Complex Z', '/Users/bwelliver/cuore/bolord/complex_z/test/zabs_fft_' + current_key + 'uA_log.png', logx='log', logy='log')
        
        gen_plot_points(np.real(z[fcut]), np.imag(z[fcut]), 'Re(Z)', 'Im(Z)', 'Z Re vs Im', '/Users/bwelliver/cuore/bolord/complex_z/test/z_re_im_' + current_key + 'uA.png', log='linear')
        # Next logic test is to loop over the z dictionary and cut near the frequency to construct a bunch of points
        z_array = np.empty(0)
        f_array = np.empty(0)
        zmean_array = np.empty(0)
        fmean_array = np.empty(0)
        f0 = 17
        fMax = 1e4
        fRange = int(fMax/2/15)
        vcut = np.logical_and(np.abs(data['fv_in']) > 2, data['frequency'] < fMax)
        # At the end of this for loop f_array and z_array will contain, for the current_key, the values of z and f to use
        # in making the complex impedence plots. So save these
        for i in range(fRange):
            f = (2*i+1)*f0
            fcut = np.logical_and(data['frequency'] > f - 0.005, data['frequency'] < f + 0.005)
            cut = np.logical_and(vcut, fcut)
            if np.sum(cut) < 1:
                continue
            f_array = np.append(f_array, data['frequency'][cut])
            z_array = np.append(z_array, z[cut])
            zmean_array = np.append(zmean_array, np.mean(z[cut]))
            fmean_array = np.append(fmean_array, np.mean(data['frequency'][cut]))
        z_dictionary[current_key] = z_array
        f_dictionary[current_key] = f_array
        gen_plot_line(f_array, np.abs(z_array), 'f (Hz)', 'Abs Complex Z', 'Complex Z', '/Users/bwelliver/cuore/bolord/complex_z/test/zarry_abs_fft_' + current_key + 'uA_log.png', logx='log', logy='log')
        
        gen_plot_points(np.real(z_array), np.imag(z_array), 'Re(Z)', 'Im(Z)', 'Z Re vs Im', '/Users/bwelliver/cuore/bolord/complex_z/test/zarray_re_im_' + current_key + 'uA.png', log='linear')
        
        gen_plot_line(fmean_array, np.abs(zmean_array), 'f (Hz)', 'Abs Complex Z', 'Complex Z', '/Users/bwelliver/cuore/bolord/complex_z/test/zmeanArray_abs_fft_' + current_key + 'uA_log.png', logx='log', logy='log')
        
        gen_plot_points(np.real(zmean_array), np.imag(zmean_array), 'Re(Z)', 'Im(Z)', 'Z Re vs Im', '/Users/bwelliver/cuore/bolord/complex_z/test/zmeanArray_re_im_' + current_key + 'uA.png', log='linear')
    if makeRoot == True:
        save_z_to_root(outputDirectory, f_dictionary, z_dictionary, data_dictionary)
    # Do a fit now
    for current in z_dictionary.keys():
        #Assumes ydata = f(xdata, *params) + eps
        #z = tes_one_block(f, P, a, b, R, g, C, T, Rl, Lin)
        f = f_dictionary[current]
        z = z_dictionary[current]
        lbounds = [1e-9, 1, 1e-1, 1e-3, 1e-14, 1e-14, 2e-3, 1e-4, 1e-9]
        #ubounds = [1, 1e6, 1e6, 1, 1, 1, 40e-3]
        ubounds = [30e-6, 1e6, 1e1,  1,     700e-12, 20e-12,  1e-1,  1e-1, 1e-7]
        x0 =      [1e-6,   5e2, 5e-1, 66e-3, 100e-12, 3e-12,   32e-3, 1e-3, 8e-9]
        
        ifake = 1e-6
        afake = 500
        bfake = 5e-1
        rfake = 66e-3
        gfake = 344e-12
        cfake = 5e-12
        tfake = 32e-3
        rlfake = 1e-3
        linfake = 8e-9
        #f = np.asarray([i+1 for i in range(int(1e4-1))])
        #z = complex_tes_one_block(f, ifake, afake, bfake, rfake, gfake, cfake, tfake, rlfake, linfake)
        zz = np.real(z)
        zz = np.append(zz, np.imag(z))
        
        result, pcov = curve_fit(tes_one_block, f, zz, p0=x0, bounds=(lbounds,ubounds))
        perr = np.sqrt(np.diag(pcov))
        #print('The fit results for current {} are {}'.format(current, result))
        # Try nll
        #x0 = [1e-6, 1e4, 1e3, 66e-3, 1e-3, 1e-9, 32e-3]
        #print('The initial guess is {0}'.format(x0))
        
        #result = minimize(nll, x0=np.array(x0), args=(Rm,Tm,fDict[func2fit]), method='Nelder-Mead', options={'fatol':1e-11})
        #result = minimize(nll_tes_one_block, x0=np.array(x0), args=(f,zz), method='Nelder-Mead', options={'fatol': 1e-11, 'xatol': 1e-11, 'adaptive': True})
        #result = result.x
        #perr = result/100
        
        #params, cov, infodict, mesg, ier = leastsq(residuals, x0, args=(f, z), full_output=True)
        #result = params
        print('The fit results for current {} are {}'.format(current, result))
        print('The fit errors for current {} are {}'.format(current, perr))
        # Try to plot?
        z_model = complex_tes_one_block(f, result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8])
        #print('ReZ is {}'.format(np.real(z_model)))
        xlab = 'Re(Z)'
        ylab = 'Im(Z)'
        title = 'Real Z vs Imag Z with fit'
        
        gen_plot_points_fit(z, z_model, result, perr, xlab, ylab, title, '/Users/bwelliver/cuore/bolord/complex_z/test/zfit_re_im_' + current + 'uA.png', log='linear')
        
        gen_plot_line(f, np.abs(z_model), 'f (Hz)', 'Abs Complex Z Model', 'Complex Z Model', '/Users/bwelliver/cuore/bolord/complex_z/test/zarry_model_abs_fft_' + current + 'uA_log.png', logx='log', logy='log')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDirectory', help='Specify the full path of the directory that contains all the files you wish to use')
    parser.add_argument('-o', '--outputDirectory', help='Specify the full path of the output directory to put plots and root files. If it is not a full path, a plots and root subdirectory will be added in the input directory')
    parser.add_argument('-r', '--makeRoot', action='store_true', help='Specify whether to write complex impedence data to a root file')
    
    args = parser.parse_args()
    outPath = args.outputDirectory if args.outputDirectory else args.inputDirectory
    
    do_complex_z(inputDirectory=args.inputDirectory, outputDirectory=outPath, makeRoot=args.makeRoot)
    
#    # SQUID parameters...mutual inductance, feedback resistor, bias resistor, bias capacitor, shunt
#    M = -1.272
#    Rfb = 1e4
#    Li = 6e-9
#    Lf = M*M*Li
#    Rbias = 10000.0
#    Cbias = 150e-12
#    Rsh = 21e-3
#    
#    # Specify input parameters for TES state
#    Rn = 19e-3 # bias dependent Rn
#    i_dc = 28.25e-6 # dc bias current
#        
#    # Get relevant data into arrays
#    t, vIn, vOut = readLVM(inFile)
#    tcut = t > 0
#    f = time_to_freq(t[tcut])
#    Zbias = Rbias / (1 + 2*np.pi*f*1j*Rbias*Cbias) # This is the input impedance --> Rbias + (1)/(2*np.pi*1j*f*Cbias)
#    Zfb = Rfb + 1j*2*np.pi*f*Lf
#    # Z(w) = Ztes + Rsh + jwL + Rpar
#    # Also Z(w) = Vth/Ites
#    
#    # Vth/Zth = Vsource/Zsource
#    # Zth = Rsh
#    # Zsource = Rsh + Zbias
#    
#    # Get vTES
#    # The input bias current will be AC and dependent upon the DC offset and whatever goes in
#    mFilter = True
#    if mFilter is True:
#        newVin = np.zeros(0)
#        newVout = np.zeros(0)
#        newf = np.zeros(0)
#        newt = np.zeros(0)
#        newZbias = np.zeros(0)
#        newZfb = np.zeros(0)
#        ss = 10
#        lb = 0
#        ub = ss
#        while lb < vIn.size - 1 and ub - lb > 1:
#            tVin = vIn[lb:ub]
#            tVout = vOut[lb:ub]
#            #tf = f[lb:ub]
#            tt = t[lb:ub]
#            #tZbias = Zbias[lb:ub]
#            #tZfb = Zfb[lb:ub]
#            newVin = np.append(newVin, np.median(tVin))
#            newVout = np.append(newVout, np.median(tVout))
#            #newf = np.append(newf, np.median(tf))
#            newt = np.append(newt, np.median(tt))
#            #newZbias = np.append(newZbias, np.median(tZbias))
#            #newZfb = np.append(newZfb, np.median(tZfb))
#            lb += 1
#            ub = ub + 1 if ub + ss <= vIn.size - 1 else vIn.size - 1
#        vIn = newVin
#        vOut = newVout
#        t = newt
#        tcut = t > 0
#        f = time_to_freq(t[tcut])
#        Zbias = Rbias / (1 + 2*np.pi*f*1j*Rbias*Cbias) # This is the input impedance --> Rbias + (1)/(2*np.pi*1j*f*Cbias)
#        Zfb = Rfb + 1j*2*np.pi*f*Lf
#    # Test:
#    fvIn = fftpack.fft(vIn[tcut] + i_dc*Rbias)
#    fvOut = fftpack.fft(vOut[tcut])
#    
#    fvTh = fvIn*Rsh/Zbias
#    
#    fvRatio = fvIn/fvOut
#    
#    z = (fvRatio)*(M*Rsh*Zfb)/(Zbias)
#    # sanity check
#    tn = t - t[0]
#    
#    cut = np.logical_and(f > 1e3, f < 1e7)
#    fcut = f > 0
#    
#    #gen_plot_points(t, vIn, 't', 'vIn', 'Input Voltage vs Time', '/Users/bwelliver/cuore/bolord/run10/z/vIn_vs_t.png', log='linear')
#    # Make plots?
#    gen_plot_line(f[fcut], np.abs(fvIn[fcut]), 'f (Hz)', 'vIn fft', 'vIn Voltage FFT', '/Users/bwelliver/cuore/bolord/run10/z/vin_fft_log.png', logx='log', logy='log')
#    gen_plot_line(f[fcut], np.abs(fvOut[fcut]), 'f (Hz)', 'vOut fft', 'vOut Voltage FFT', '/Users/bwelliver/cuore/bolord/run10/z/vout_fft_log.png', logx='log', logy='log')
#    gen_plot_line(f[fcut], np.abs(fvTh[fcut]), 'f (Hz)', 'vTh fft', 'vTh Voltage FFT', '/Users/bwelliver/cuore/bolord/run10/z/vth_fft_log.png', logx='log', logy='log')
#    gen_plot_line(f[fcut], np.abs(fvRatio[fcut]), 'f (Hz)', 'vIn/vOut', 'vIn/vOut FFT', '/Users/bwelliver/cuore/bolord/run10/z/vratio_fft_log.png', logx='log', logy='log')
#    
#    gen_plot_line(f, np.abs(z), 'f (Hz)', 'z fft', 'z FFT', '/Users/bwelliver/cuore/bolord/run10/z/z_fft_log.png', logx='log', logy='log')
#    
#    gen_plot_points(np.real(fvRatio[cut]), np.imag(fvRatio[cut]), 'Re(vRatio)', 'Im(vRatio)', 'vRatio Real vs Imag', '/Users/bwelliver/cuore/bolord/run10/z/vRatio_re_im.png', log='linear')
#    
#    cut = np.logical_and(cut, np.abs(z) < 10)
#    gen_plot_points(np.real(z[cut]), np.imag(z[cut]), 'Re(Z)', 'Im(Z)', 'Z Re vs Im', '/Users/bwelliver/cuore/bolord/run10/z/z_re_im_log.png', log='linear')
#    #print(tData['time'])
#    # We also need to obtain the UnixTimestamps from the ROOT file.
#    #tree = 'BridgeLog'
#    #branch = 'Time_sec'
#    
#    print('Noise thermometry data has been parsed into unix timestamps.')

