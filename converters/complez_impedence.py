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
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from readROOT import readROOT
import cloneROOT as cr

import matplotlib as mp
from matplotlib import pyplot as plt


def gen_plot_line(x, y, xlab, ylab, title, fName, logx='log', logy='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(16, 9))
    ax = fig2.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=1, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', linewidth=1)
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
    ax.plot(x, y, marker='.', markersize=1, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_yscale(log)
    ax.set_title(title)
    fig2.savefig(fName, dpi=100)
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


def readLVM(filename):
    '''Load a fast digitizer lvm and get signal and response columns
    The first few lines are header so we will skip them
    '''
    
    with open(filename, 'r') as f:
        lines = f.readlines()[22:]
    
    t = []
    vIn = []
    vOut = []
    for line in lines:
        line = line.strip('\n').split('\t')
        t.append(float(line[0]))
        vIn.append(float(line[1]))
        vOut.append(float(line[3]))
    return np.asarray(t), np.asarray(vIn), np.asarray(vOut)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', help='Specify the full path of the file you wish to use')
    args = parser.parse_args()
    
    inFile = args.inputFile
    
    # SQUID parameters...mutual inductance, feedback resistor, bias resistor, bias capacitor, shunt
    M = -1.272
    Rfb = 1e4
    Li = 6e-9
    Lf = M*M*Li
    Rbias = 10000.0
    Cbias = 150e-12
    Rsh = 21e-3
    
    # Specify input parameters for TES state
    Rn = 19e-3 # bias dependent Rn
    i_dc = 28.25e-6 # dc bias current
        
    # Get relevant data into arrays
    t, vIn, vOut = readLVM(inFile)
    tcut = t > 0
    f = time_to_freq(t[tcut])
    Zbias = Rbias / (1 + 2*np.pi*f*1j*Rbias*Cbias) # This is the input impedance --> Rbias + (1)/(2*np.pi*1j*f*Cbias)
    Zfb = Rfb + 1j*2*np.pi*f*Lf
    # Z(w) = Ztes + Rsh + jwL + Rpar
    # Also Z(w) = Vth/Ites
    
    # Vth/Zth = Vsource/Zsource
    # Zth = Rsh
    # Zsource = Rsh + Zbias
    
    # Get vTES
    # The input bias current will be AC and dependent upon the DC offset and whatever goes in
    mFilter = True
    if mFilter is True:
        newVin = np.zeros(0)
        newVout = np.zeros(0)
        newf = np.zeros(0)
        newt = np.zeros(0)
        newZbias = np.zeros(0)
        newZfb = np.zeros(0)
        ss = 10
        lb = 0
        ub = ss
        while lb < vIn.size - 1 and ub - lb > 1:
            tVin = vIn[lb:ub]
            tVout = vOut[lb:ub]
            #tf = f[lb:ub]
            tt = t[lb:ub]
            #tZbias = Zbias[lb:ub]
            #tZfb = Zfb[lb:ub]
            newVin = np.append(newVin, np.median(tVin))
            newVout = np.append(newVout, np.median(tVout))
            #newf = np.append(newf, np.median(tf))
            newt = np.append(newt, np.median(tt))
            #newZbias = np.append(newZbias, np.median(tZbias))
            #newZfb = np.append(newZfb, np.median(tZfb))
            lb += 1
            ub = ub + 1 if ub + ss <= vIn.size - 1 else vIn.size - 1
        vIn = newVin
        vOut = newVout
        t = newt
        tcut = t > 0
        f = time_to_freq(t[tcut])
        Zbias = Rbias / (1 + 2*np.pi*f*1j*Rbias*Cbias) # This is the input impedance --> Rbias + (1)/(2*np.pi*1j*f*Cbias)
        Zfb = Rfb + 1j*2*np.pi*f*Lf
    # Test:
    fvIn = fftpack.fft(vIn[tcut] + i_dc*Rbias)
    fvOut = fftpack.fft(vOut[tcut])
    
    fvTh = fvIn*Rsh/Zbias
    
    fvRatio = fvIn/fvOut
    
    z = (fvRatio)*(M*Rsh*Zfb)/(Zbias)
    # sanity check
    tn = t - t[0]
    
    cut = np.logical_and(f > 1e3, f < 1e7)
    fcut = f > 0
    
    #gen_plot_points(t, vIn, 't', 'vIn', 'Input Voltage vs Time', '/Users/bwelliver/cuore/bolord/run10/z/vIn_vs_t.png', log='linear')
    # Make plots?
    gen_plot_line(f[fcut], np.abs(fvIn[fcut]), 'f (Hz)', 'vIn fft', 'vIn Voltage FFT', '/Users/bwelliver/cuore/bolord/run10/z/vin_fft_log.png', logx='log', logy='log')
    gen_plot_line(f[fcut], np.abs(fvOut[fcut]), 'f (Hz)', 'vOut fft', 'vOut Voltage FFT', '/Users/bwelliver/cuore/bolord/run10/z/vout_fft_log.png', logx='log', logy='log')
    gen_plot_line(f[fcut], np.abs(fvTh[fcut]), 'f (Hz)', 'vTh fft', 'vTh Voltage FFT', '/Users/bwelliver/cuore/bolord/run10/z/vth_fft_log.png', logx='log', logy='log')
    gen_plot_line(f[fcut], np.abs(fvRatio[fcut]), 'f (Hz)', 'vIn/vOut', 'vIn/vOut FFT', '/Users/bwelliver/cuore/bolord/run10/z/vratio_fft_log.png', logx='log', logy='log')
    
    gen_plot_line(f, np.abs(z), 'f (Hz)', 'z fft', 'z FFT', '/Users/bwelliver/cuore/bolord/run10/z/z_fft_log.png', logx='log', logy='log')
    
    gen_plot_points(np.real(fvRatio[cut]), np.imag(fvRatio[cut]), 'Re(vRatio)', 'Im(vRatio)', 'vRatio Real vs Imag', '/Users/bwelliver/cuore/bolord/run10/z/vRatio_re_im.png', log='linear')
    
    cut = np.logical_and(cut, np.abs(z) < 10)
    gen_plot_points(np.real(z[cut]), np.imag(z[cut]), 'Re(Z)', 'Im(Z)', 'Z Re vs Im', '/Users/bwelliver/cuore/bolord/run10/z/z_re_im_log.png', log='linear')
    #print(tData['time'])
    # We also need to obtain the UnixTimestamps from the ROOT file.
    #tree = 'BridgeLog'
    #branch = 'Time_sec'
    
    print('Noise thermometry data has been parsed into unix timestamps.')

