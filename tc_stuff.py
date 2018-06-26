import os
from os.path import isabs
from os.path import dirname
from os.path import basename

import sys
import getopt
import argparse
import re
import datetime
import time

import ROOT as rt
import numpy as np
import pandas as pan

from matplotlib import pyplot as plt
import matplotlib as mp

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

from numpy import exp as exp
from numpy import log as ln
from numpy import square as p2
from numpy import sqrt as sqrt
from numpy import sum as nsum
from numpy import tanh as tanh

from scipy.special import erf as erf
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import skew
from scipy.odr import ODR, Model, Data, RealData

from readROOT import readROOT

eps = np.finfo(float).eps
ln = np.log

class RingBuffer(object):

    def __init__(self, size_max, default_value=0.0, dtype=float):
        """initialization."""

        self.size_max = size_max

        self._data = np.empty(size_max, dtype=dtype)
        self._data.fill(default_value)

        self.size = 0

    def append(self, value):
        """append an element."""
        self._data = np.roll(self._data, 1)
        self._data[0] = value

        self.size += 1

        if self.size == self.size_max:
            self.__class__ = RingBufferFull

    def get_sum(self):
        """sum of the current values."""
        return(np.sum(self._data))
        
    def get_mean(self):
        '''mean of the current values.'''
        return(np.mean(self._data))
        
    def get_med(self):
        '''median of the current values.'''
        return(np.median(self._data))

    def argmax(self):
        """return index of first occurence of max value."""
        return(np.argmax(self._data))

    def get_all(self):
        """return a list of elements from the newest to the oldest (left to
        right)"""
        return(self._data)

    def get_partial(self):
        return(self.get_all()[0:self.size])

    def __getitem__(self, key):
        """get element."""
        return(self._data[key])

    def __repr__(self):
        """return string representation."""
        s = self._data.__repr__()
        s = s + '\t' + str(self.size)
        s = s + '\t' + self.get_all()[::-1].__repr__()
        s = s + '\t' + self.get_partial()[::-1].__repr__()
        return(s)


class RingBufferFull(RingBuffer):

    def append(self, value):
        """append an element when buffer is full."""
        self._data = np.roll(self._data, 1)
        self._data[0] = value


def mkdpaths(dirpath):
    os.makedirs(dirpath, exist_ok=True)


def get_excitationI(array):
    '''Function to map a numpy array of current map index values to the actual current values'''
    output = np.copy(array)
    key2cur = {1: 1e-12, 2: 3.16e-12, 3: 10e-12, 4: 31.6e-12, 5: 100e-12, 6: 316e-12,
              7: 1e-9, 8: 3.16e-9, 9: 10e-9, 10: 31.6e-9, 11: 100e-9, 12: 316e-9,
              13: 1e-6, 14: 3.16e-6, 15: 10e-6, 16: 31.6e-6, 17: 100e-6, 18: 316e-6,
              19: 1e-3, 20: 3.16e-3, 21: 10e-3, 22: 31.6e-3}
    for key, value in key2cur.items():
        output[array==key] = value
    return output

def get_excitationI2(array):
    '''Function to map a numpy array of current map index values to the actual current values. This function is slightly slower than get_excitationI'''
    from_values = np.arange(1,23)
    to_values = np.asarray([1e-12, 3.16e-12, 10.0e-12, 31.6e-12, 100e-12, 316e-12, 
                            1e-9, 3.16e-9, 10.0e-19, 31.6e-9, 100e-9, 316e-9,
                           1e-6, 3.16e-6, 10e-6, 31.6e-6, 100e-6, 316e-6,
                           1e-3, 3.16e-3, 10.0e-3, 31.6e-3])
    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values, array, sorter=sort_idx)
    return to_values[sort_idx][idx]


def get_currentString(current):
    '''Function to convert a current value to closest logical units'''
    if current < 317e-12:
        current = str(current * 1e12)
        current = str(round(float(current),2)) + 'pA'
    elif current > 317e-12 and current < 317e-9:
        current = str(current * 1e9)
        current = str(round(float(current),2)) + 'nA'
    elif current > 317e-9 and current < 317e-6:
        current = str(current * 1e6)
        current = str(round(float(current),2)) + 'uA'
    elif current > 317e-6 and current < 316e-3:
        current = str(current * 1e3)
        current = str(round(float(current),2)) + 'mA'
    else:
        current = str(current)
        current = str(round(float(current),2)) + 'A'
    return current

def get_voltageString(voltage):
    '''Function to convert a voltage value to closest logical units'''
    if voltage < 633e-6:
        voltage = str(voltage * 1e6)
        voltage = str(round(float(voltage),2)) + 'uV'
    elif voltage > 633e-6 and voltage < 633e-3:
        voltage = str(voltage * 1e3)
        voltage = str(round(float(voltage),2)) + 'mV'
    else:
        voltage = str(voltage)
        voltage = str(round(float(voltage),2)) + 'V'
    return voltage

# Plot root files
def gen_plot(x, y, xlab, ylab, title, fName, log='log'):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid()
    fig.savefig(fName, format='png', dpi=100)
    plt.close('all')


def gen_fitplot(x, y, xerr, yerr, yerrType, yfit, xfitp, yfitp, result, perr, xlab, ylab, title, fName, func2fit, log='log'):
    """Create generic plots that may be semilogx (default)"""
    if yerrType == 'relative':
        # Convert to absolute
        yerr = yerr*y
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', xerr=xerr, yerr=yerr)
    ax.plot(xfitp, yfitp, 'r-', marker='None', linewidth=2)
    ax.set_xscale(log)
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], 1.1*max(y)])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid()
    # Compute chisq per ndf
    ndf = x.size - result.size
    if not isinstance(yerr, list):
        chi = ((y-yfit)/yerr)**2
        icmax = np.argmax(chi)
        #print('The values associated with the max chi term are y = {}, yfit = {}, yerr = {}, chi2 = {}'.format(y[icmax], yfit[icmax], yerr[icmax], chi[icmax]))
    else:
        chi = ((y-yfit))**2
    chi = np.sum(chi)
    # Make text for box
    if func2fit != 'tanh_lin':
        tChi = '$\chi^2/\mathrm{ndf} = %.5f / %.d$'%(chi, ndf)
        tRn = '$\mathrm{R_n}=%.5f \pm %.5f \mathrm{\Omega}$'%(result[0], perr[0])
        tRsc = '$\mathrm{R_{sc}}=%.5f \pm %.5f \mathrm{\Omega}$'%(result[1], perr[1])
        tTc = '$\mathrm{T_c}=%.5f \pm %.5f \mathrm{mK}$'%(result[2]*1e3, perr[2]*1e3)
        tTw = '$\mathrm{\Delta T_c}=%.5f \pm %.5f \mathrm{mK}$'%(result[3]*1e3, perr[3]*1e3)
        textstr = tChi + '\n' + tRn + '\n' + tRsc + '\n' + tTc + '\n' + tTw
    else:
        # Rn is a*x^b + c
        # a has units of Ohms, b is unitless, c has units of Ohms
        tChi = '$\chi^2/\mathrm{ndf} = %.5f / %.d$'%(chi, ndf)
        tRsc = '$\mathrm{R_{sc}}=%.5f \pm %.5f \mathrm{\Omega}$'%(result[0], perr[0])
        tTc = '$\mathrm{T_c}=%.5f \pm %.5f \mathrm{mK}$'%(result[1]*1e3, perr[1]*1e3)
        tTw = '$\mathrm{\Delta T_c}=%.5f \pm %.5f \mathrm{mK}$'%(result[2]*1e3, perr[2]*1e3)
        tA = '$\mathrm{a} = %.4f \pm %.4f \mathrm{\Omega / K^{b}}$'%(result[3], perr[3])
        tB = '$\mathrm{b} = %.4f \pm %.4f$'%(result[4], perr[4])
        #tC = '$\mathrm{c} = %.4f \pm %.4f \mathrm{\Omega}$'%(result[5], perr[5])
        textstr = tChi + '\n' + tRsc + '\n' + tTc + '\n' + tTw + '\n' + tA + '\n' + tB
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    ax.text(0.5, 0.4, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    # Function for fitting
    if func2fit == 'tanh_lin':
        tX = r'$x = \frac{T - T_c}{\Delta T_c}$'
        tf = r'$R_{n}(T) = \mathrm{a}T^{b}$'
        #tR = r'$R(T) = \frac{R_n - R_{sc}}{2}\left(\frac{\left(1 + f(x) \right)e^{x} - e^{-x}}{e^{x} + e^{-x}} + 1\right) - R_{sc}$'
        tR = r'$R(T) = \frac{R_{n}(T)}{2}\left(tanh(x) + 1\right) + R_{sc}$'
        tFit = tR + '\n' + tX + '\n' + tf
        ax.text(0.05, 0.99, tFit, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left')
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')


def findStep(nTunix, nR):
    '''This function should attempt to locate Tc jumps.'''
    
    # We have some R vs t data, and it jumps from low to high. We should grab 
    # data that is low and some past where it is high (basically until it falls again)
    
    # Get dR
    dR = np.diff(nR)
    
    # Now detect jumps we can use a window
    rbuff = RingBuffer(10)
    jStart = np.empty(0)
    jEnd = np.empty(0)
    jumpFlag = False
    for ev in range(dR.shape[0]):
        rbuff.append(dR[ev])
        
        if ev >= 10:
            med2 = np.abs(rbuff.get_med())
            
            if med2 > 100*med1 and jumpFlag == False:
                print('Jump detected')
                jStart = np.append(jStart, ev)
                jumpFlag = True
                
            if med2 < 0.01*med1 and jumpFlag == True:
                print('Jump down detected')
                jEnd = np.append(jEnd, ev)
                jumpFlag = False
            med1 = med2
        else:
            med1 = np.abs(rbuff.get_med())
    return jStart, jEnd
                

def findTc():
    '''Routine to find the Tc and Rn and Rsc of our our measurement'''
    
    # The logistic curve L(x) goes from 0 to L
    # We also have L(x) = L(1 + tanh(x/2))/2
    # Examination:
    # At x --> Inf L(x) --> L and L*(1 + tanh(x/2))/2 goes to L*(2)/2 = L
    # At x --> -Inf, L(x) --> 0 and L*(1 + tanh(x/2))/2 goes to L*(0)/2 = 0
    # L(0) = L/2 and L*(1+tanh(0))/2 = L*(1)/2 = L/2
    #
    # This is a favorable behavior for what we want. Here we want to shift the lower boundary
    # to be Rsc and the upper boundary to be Rn.
    # Let L = 1
    # y = (1/2)*(1+tanh(x/2))
    # First shift up by Rsc
    # y = (1/2)*(1+tanh(x/2)) + Rsc
    # Next Rescale so that the upper limit goes to Rn, not Rsc + 1
    #
    # y = Rn/(Rsc+ * [(1/2)*(1+tanh(x/2))) + Rsc] does not work


def erf_tc(T, Rn, Rsc, Tc, Tw):
    '''Get Resistance values from fit T
    Rn is the normal resistance
    Rsc is the superconducting resistance
    Tc is the critical temperature when R = Rn/2
    Tw is the width of the transistion region
    T is the temperature data'''
    
    R = (Rn - Rsc)/2.0 * (erf((T - Tc)/Tw) + 1.0) + Rsc
    
    return R


def tanh_tc(T, Rn, Rsc, Tc, Tw):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance
    Rsc is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data'''
    
    R = (Rn - Rsc)/2.0 * (tanh((T - Tc)/Tw) + 1.0) + Rsc
    return R


def tanh_tc2(beta, T):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance
    Rsc is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data'''
    Rn, Rsc, Tc, Tw = beta
    if Rn > 1 or Tc > T.max() or Tw > 3e-3:
        return np.inf*T
    R = (Rn - Rsc)/2.0 * (tanh((T - Tc)/Tw) + 1.0) + Rsc
    return R


def tanh_linear_tc(T, Rn, Rsc, Tc, Tw, a, b):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance
    Rsc is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data'''
    x = (T - Tc)/Tw
    coeff = x/x
    cut = x >= 0
    coeff[cut] = a*np.power(x[cut], b) + 1
    fun = (coeff*exp(x) - exp(-x))/(exp(x) + exp(-x))
    R = (Rn - Rsc)/2.0 * (fun + 1) + Rsc
    return R


def tanh_linear_tc2(T, Rsc, Tc, Tw, a, b, c):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance which is now a function of T
    Rsc is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data'''
    x = (T - Tc)/Tw
    cut = x >= 0
    fun = tanh(x)
    nR = c*x/x
    Rn = 0
    nR[cut] = a*np.power(x[cut], b) + c
    R = nR*(fun + 1)/2 + Rsc
    return R


def tanh_linear_tc3(T, Rsc, Tc, Tw, a, b):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance which is now a function of T
    Rsc is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data'''
    x = (T - Tc)/Tw
    fun = tanh(x)
    Rn = 0
    nR = a*np.power(T, b)
    R = nR*(fun + 1)/2 + Rsc
    return R


def nll_erf(params, R, T):
    """A negative log-Likelihood of erf fit"""
    Rn,Rsc,Tc,Tw = params
    if Tc <= 0 or Rn <= 0 or Rsc <= 0 or Tw <= 0:
        return np.inf
    else:
        model = erf_tc(T,Rn,Rsc,Tc,Tw)
        lnl = nsum((R - model)**2)
        return lnl


def nll_tanh(params, R, T):
    '''A fit function for the tanh'''
    Rn,Rsc,Tc,Tw = params
    if Tc <= 0 or Rn <= 0 or Rsc <= 0 or Tw <= 0:
        return np.inf
    else:
        model = tanh_tc(T,Rn,Rsc,Tc,Tw)
        lnl = nsum((R - model)**2)
        return lnl

def nll(params, R, T, func):
    '''A fit for whatever function'''
    Rn, Rsc, Tc, Tw = params
    if Tc <= 0 or Rn <= 0 or Rsc <= 0 or Tw <= 0:
        return np.inf
    else:
        model = func(T,Rn,Rsc,Tc,Tw)
        lnl = nsum((R - model)**2)
        return lnl


def nll_error(params, R, Rerr, T, func):
    '''A fit for whatever function with y-errors'''
    Rn, Rsc, Tc, Tw = params
    if Tc <= 0 or Rn <= 0 or Rsc <= 0 or Tw <= 0:
        return np.inf
    else:
        model = func(T,Rn,Rsc,Tc,Tw)
        lnl = nsum(((R - model)/Rerr)**2)
        return lnl


def gaus(x, a, mu, sigma):
    """The gaussian pdf
    mu is mean
    sigma is standard deviation."""
    return exp(-1 * p2((x - mu)) / (2*p2(sigma))) * ( a/(sqrt(2*np.pi) * sigma) )/a


def gaus2(x, a1, mu1, sigma1, a2, mu2, sigma2):
    """The gaussian pdf
    mu is mean
    sigma is standard deviation."""
    model = ((a1/(sqrt(2*np.pi) * sigma1)) * exp(-1 * p2((x - mu1)) / (2*p2(sigma1))) + (a2/(sqrt(2*np.pi) * sigma2)) * exp(-1 * p2((x - mu2)) / (2*p2(sigma2))))/(a1+a2)
    return model


def nll_gaus(params, data):
    """A negative log-Likelihood function"""
    a, mu, sigma = params
    if sigma <= 0 or a < 0 or mu > data.max() or mu < data.min():
        return np.inf
    else:
        lnl = - nsum( ln(gaus(data, a, mu, sigma) + eps ) )
        return lnl
    

def nll_gaus2(params, data):
    """A negative log-Likelihood function"""
    a1, mu1, sigma1, a2, mu2,sigma2 = params
    if sigma1 <= 0 or sigma2 <= 0 or a1 < 0 or a2 < 0 or mu2 < mu1 or mu2 < data.min() or mu1 > data.max():
        return np.inf
    else:
        lnl = -nsum( ln(gaus2(data, a1, mu1, sigma1, a2, mu2, sigma2) + eps ) )
        return lnl


def hist_plot(data, xModel, yModel, title, figPrefix, params, peaks):
    if peaks == 2:
        aRn, muRn, sigRn, aRs, muRs, sigRs = params
    if peaks == 1:
        aRn, muRn, sigRn = params
    fig5 = plt.figure(figsize=(8, 6))
    ax = fig5.add_subplot(111)
    n = data.size
    print('The data is size {}, the xModel and yModel are sizes {}'.format(data.size, (xModel.size, yModel.size)))
    sg = np.sqrt((6*(n-2))/((n+1)*(n+3)))
    bins = np.int(np.ceil(1 + np.log2(data.size) + np.log2(1 + np.abs(skew(data))/sg)))
    hist, bin_edges = np.histogram(data, bins='fd')
    bins = bin_edges.size + 2 
    bins = data.size
    print('Bins are {0}'.format(bins))
    ax.hist(data, bins=bins, normed=True)
    ax.plot(xModel, yModel, 'r-', marker='None', linewidth=2)
    ax.grid()
    #ax.set_xscale('log')
    ymin, ymax = ax.get_ylim()
    if peaks == 2:
        textstr = '$\mu_{\mathrm{R_n}}=%.5f \mathrm{\Omega}$\n $\sigma_{\mathrm{R_n}}=%.5f \mathrm{\Omega}$\n $\mu_{\mathrm{R_{sc}}}=%.5f \mathrm{\Omega}$\n $\sigma_{\mathrm{R_{sc}}}=%.5f \mathrm{\Omega}$'%(muRn, sigRn, muRs, sigRs)
    if peaks == 1:
        textstr = '$\mu_{\mathrm{R}}=%.5f \mathrm{\Omega}$\n $\sigma_{\mathrm{R}}=%.5f \mathrm{\Omega}$'%(muRn, sigRn)
    print(textstr)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    ax.text(0.7, 0.9, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    #ax.plot((ksThresh, ksThresh), (ymin, ymax), color='red', linewidth=2.0, label='KS threshold')
    ax.set_xlabel('Resistance Values')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    #ax.legend(loc=2)
    fig5.savefig(figPrefix + '.png', format='png', dpi=100)
    plt.close('all')


def dump2text(R,T,fileName):
    '''Quick function to dump R and T values to a text file'''
    print('The shape of R and T are: {0} and {1}'.format(R.shape, T.shape))
    np.savetxt(fileName, np.stack((R,T), axis=1), fmt='%12.10f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filePath', default='/Users/bwelliver/Downloads/tc_data/Aligned_Scanner_20160719_190456_p00000.root', help='Specify where to find data')
    parser.add_argument('-c', '--channel', default=1, help='Specify which channel to run over, or if set to all, run over all channels')
    parser.add_argument('-i', '--current', default='all', help='Specify which excitation current to fit. Enter value in units of A. Default is all. To select all currents enter all')
    parser.add_argument('-m', '--fitMethod', default='curve', help='Select fit method: SLSQP, Nelder-Mead, or curve for scipy curve_fit (default)')
    parser.add_argument('-R', '--function', default='tanh', help='Select the functional form to fit: erf or tanh (default)')
    parser.add_argument('-o', '--outputPath', help='Specify the output path to put plots. If not specified will create a  directory in the input path based off the input filename. If not absolute path, will be relative to the input file path')
    parser.add_argument('-d', '--dumpFile', action='store_true', help='Specify whether a dump file to store R and T in text should be made. Default location will be set to outputPath')
    parser.add_argument('-s', '--smooth', action='store_true', help='Specify whether to use smoothing to handle overlap cases')
    parser.add_argument('-v', '--voltage', default='all', help='Specify which excitation voltage to fit. Enter value in units of V. Default is all. To select all voltages enter all')
    parser.add_argument('-T', '--temp', default='nt', help='Select the thermometer to use: nt for noise thermometer or ep for ep cal')
    args = parser.parse_args()

    path = args.filePath
    ch = int(args.channel) if args.channel != 'all' else args.channel
    eI = float(args.current) if args.current != 'all' else args.current
    eV = float(args.voltage) if args.voltage != 'all' else args.voltage
    fit_method = args.fitMethod
    func2fit = args.function
    outPath = args.outputPath if args.outputPath else dirname(path) + '/' + basename(path).strip('.root')
    if not isabs(outPath):
        outPath = dirname(path) + '/' + outPath
    bDump = args.dumpFile
    
    mkdpaths(outPath)
    mFilter = args.smooth
    # Create a dictionary of functions
    fDict = {'erf': erf_tc, 'tanh': tanh_tc, 'tanh_lin': tanh_linear_tc3}
    nllDict = {'erf': nll_erf, 'tanh': nll_tanh}

    print('We will run with the following options:')
    print('The input file is: {0}\nThe output path is: {1}\nThe channel number is: {2}\nThe fit method is: {3}\nThe functional form is: {4}'.format(path, outPath, ch, fit_method, func2fit))

    lof = path
    tree = 'BridgeLog'
    tree = 'btree'
    branches = ['Time_sec', 'Chan0' + str(ch) + '_R_Ohm', 'NTemp', 'NTemp_err']
    branches = ['MeasTime', 'Channel', 'ChannelExcitation', 'Resistance_Ohm', 'NTemp', 'NTemp_err', 'ChanPower_W', 'EPCal']
    branches = ['MeasTime', 'Channel', 'ChannelExcitation', 'Resistance_Ohm', 'NTemp', 'NTemp_err', 'ChanPower_W', 'EPCal']
    #branches = ['MeasTime', 'Channel', 'ChannelExcitation', 'Resistance_Ohm']
    method = 'single'
    rData = readROOT(lof, tree, branches, method)
    
    # Make life easier:
    rData = rData['data']
    # Append some fake data?
    debug = False
    if debug:
        sCut = np.logical_and(rData[branches[3]] > 0.02, rData[branches[1]] == ch)
        for idx in range(len(branches)):
            if branches[idx] == 'NTemp':
                rData[branches[idx]] = np.append(rData[branches[idx]],rData[branches[idx]][sCut] + 0.5*np.min(rData[branches[idx]]))
            elif branches[idx] == 'Resistance_Ohm':
                #print(np.median(rData[branches[idx]][sCut]))
                rData[branches[idx]] = np.append(rData[branches[idx]], np.random.normal(np.median(rData[branches[idx]][sCut]), np.std(rData[branches[idx]][sCut]), sCut.sum()) )
            else:
                rData[branches[idx]] = np.append(rData[branches[idx]],rData[branches[idx]][sCut])
    
    chan = rData[branches[1]]
    current = get_excitationI(rData[branches[2]])
    cut = chan == ch if eI == 'all' else np.logical_and(chan == ch, current == eI)
    nTunix = rData[branches[0]][cut]
    nR = rData[branches[3]][cut]
    nT = rData[branches[4]][cut]
    nEP = rData[branches[7]][cut]
    nTerr = rData[branches[5]][cut]
    nP = rData[branches[6]][cut]
    
    nI = np.sqrt(nP/nR)
    #nTunix = rData[branches[0]]
    #nR = rData[branches[1]]
    #nT = rData[branches[2]]
    #nTerr = rData[branches[3]]
    print('The start timestamp is: {}'.format(nTunix[0]))
    print('The minimum relative time is: {}'.format(np.min(nTunix - nTunix[0])))
    dt = (nTunix - nTunix[0])/1000
    #dt = nTunix
    #cut = np.logical_and(~np.logical_and(dt >= 15.5, dt < 20.5), nR >= 0)
    #cut = np.logical_and(cut, nT >= 20/1000)
    #cut = dt >= 6
    #cut = ~np.logical_and(dt > 4, dt < 7)
    #cut = np.logical_and(nR > 0.5, np.logical_and(dt > 24, dt < 32))
    #cut = np.logical_or(cut, np.logical_and(nR > 0.5, np.logical_and(dt > 2, dt < 7)) )
    #cut = np.logical_or(cut, np.logical_and(nR > 0.5, np.logical_and(dt > 17, dt < 23)) )
    #cut = np.logical_or(cut, np.logical_and(nR > 0.5, np.logical_and(dt > 41, dt < 52)) )
    #cut = ~cut
    #cut = np.logical_and(cut, ~np.logical_and(dt > 28, dt < 32))
    #cut = np.logical_and(cut, ~np.logical_and(dt > 40, dt < 51))
    #cut = dt <= 39
    cut = np.logical_and(nR >= -1, nTunix - nTunix[0] > 0)
    #cUse = cut
    print('Valid R and time cut keeps {} out of {} events'.format(cut.sum(), cut.size))
    # Set cut Temp > absolute zero
    if args.temp == 'nt':
        cStart = np.logical_and(nT > 0, cut)
    elif args.temp == 'ep':
        cStart = np.logical_and(nEP > 0, cut)
        #cBad = np.logical_and(nR > 0.5, nEP < 0.0341)
        #cStart = np.logical_and(cStart, ~cBad)
    print('cStart (T > 0 cut) keeps {} out of {} events'.format(cStart.sum(), cStart.size))
    # Set cut on EP/Cal Range
    cEP = np.logical_and(nEP > 0, nEP < 0.3)
    print('cEP (EP Cal between 0 and 300 mK) keeps {} out of {} events'.format(cEP.sum(), cEP.size))
    cStart = np.logical_and(cEP, cStart)
    print('cStart & cEP keeps {} out of {} events'.format(cStart.sum(), cStart.size))
    # Resistance cut...this is redundant with cut at the top
    cR = nR > -1
    print('cR (R > -1) keeps {} out of {} events'.format(cR.sum(), cR.size))
    # Final cut
    cUse = np.logical_and(cStart,cR)
    print('Data to use cut, cUse keeps {} out of {} events'.format(cUse.sum(), cUse.size))
    
    Istring = get_currentString(eI) if eI != 'all' else eI
    Vstring = get_voltageString(eV) if eV != 'all' else eV
    # Decide to use noise thermometer or EP Cal
    if args.temp == 'nt':
        T = nT
        fPrefix = 'nT'
        TPrefix = 'NT'
    elif args.temp == 'ep':
        T = nEP
        fPrefix = 'epcal'
        TPrefix = 'EPCal'
    #nT = nEP
    #nTerr = nTerr
    gen_plot(nTunix[cUse]-nTunix[cUse][0], nR[cUse], 'Time', 'Resistance [Ohm]', 'Resistance vs Time for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'R_vs_t_ch_' + str(ch) + '_' + Istring, 'Linear')

    gen_plot(nTunix[cUse]-nTunix[cUse][0], T[cUse], 'Time', TPrefix + ' Temp [K]', 'Temperature vs Time for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + fPrefix + '_vs_t' + '_ch_' + str(int(ch)) + '_' + Istring, 'Linear')
    
    gen_plot(T[cUse], nP[cUse], TPrefix + ' Temp [K]', 'Power [W]', 'Power vs Temperature for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'P_vs_' + fPrefix + '_ch_' + str(int(ch)) + '_' + Istring, 'Linear')
    
    gen_plot(nP[cUse], nR[cUse], 'Power [W]', 'Resistance [Ohm]', 'Resistance vs Power for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'R_vs_P' + '_ch_' + str(int(ch)) + '_' + Istring, 'Linear')
    
    cI = np.logical_and(nI > np.mean(nI) - 0.5*np.std(nI), nI < np.mean(nI) + 0.5*np.std(nI))
    cI = np.logical_and(cUse, cI)
    gen_plot(nI[cI]*1e6, nR[cI], 'Current [A]', 'Resistance [Ohm]', 'Resistance vs Current for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'R_vs_I' + '_ch_' + str(int(ch)) + '_' + Istring, 'Linear')
    
    gen_plot(T[cUse], nR[cUse], TPrefix + ' Temp [K]', 'Resistance [Ohm]', 'Resistance vs Temperature for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'R_vs_' + fPrefix + '_ch_' + str(int(ch)) + '_' + Istring, 'Linear')
    
    # Try to get fit of Rn, Rsc and Tc from data
    
    # Order the data
    Rm = nR[cUse]
    Tm = T[cUse]
    dTm = nTerr[cUse]
    idx = Tm.argsort()
    Rm = Rm[idx]
    Tm = Tm[idx]
    dTm = dTm[idx]
    
    # Get some useful bounds
    Rmax = Rm.max()
    Rmin = np.diff(np.sort(Rm)).mean()
    Tc_max = Tm.max()
    Tc_min = Tm.min()
    Tw_min = np.diff(Tm).mean()
    # Here we could try to play some sort of moving median game...take a window of 20 points and median it use rms for the error
    if mFilter is True:
        newRm = np.zeros(0)
        newRmrms = np.zeros(0)
        newTm = np.zeros(0)
        newTmrms = np.zeros(0)
        ss = 15
        lb = 0
        ub = ss
        while lb < Rm.size - 1 and ub - lb > 1:
            tR = Rm[lb:ub]
            tT = Tm[lb:ub]
            newRm = np.append(newRm, np.median(tR))
            newRmrms = np.append(newRmrms, 1.2533*np.std(tR)/np.sqrt(tT.size))
            newTm = np.append(newTm, np.median(tT))
            newTmrms = np.append(newTmrms, 1.2533*np.std(tT)/np.sqrt(tT.size))
            lb = ub
            ub = ub + ss if ub + ss <= Rm.size - 1 else Rm.size - 1
            print(lb, ub)
        print(newRm)
        print(newRmrms)
        Rm = newRm
        Tm = newTm
        dTm = newTmrms
    # Use SSE minimization
    if args.temp == 'nt':
        dTm = dTm
    elif args.temp == 'ep':
        dTm = None
    
    # Get the histogram of Rm points that are normal.
    Rnorm = Rm
    # Get fit parameters
    params = minimize(nll_gaus, x0=np.array([0.5, 0.99*Rnorm.max(), 0.1]), args=(Rnorm,))
    print('NLL fits are {0}'.format(params))
    results = params.x
    print(results)
    xModel = np.linspace(Rnorm.min(),Rnorm.max(),1e4)
    hist_plot(Rnorm, xModel, gaus(xModel, params.x[0], params.x[1], params.x[2]), 'Resistance Distribution for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'R_hist_single_' + '_ch_' + str(int(ch)) + '_' + Istring, results, 1)
    
    sigmaL_i = np.std(Rnorm[Rnorm < np.percentile(Rnorm, 10)])
    muL_i = np.mean(Rnorm[Rnorm < np.percentile(Rnorm, 10)])
    sigmaR_i = np.std(Rnorm[Rnorm > np.percentile(Rnorm, 90)])
    muR_i = np.median(Rnorm[Rnorm > np.percentile(Rnorm, 90)])
    x0=np.array([0.5, muL_i, sigmaL_i, 0.5, muR_i, sigmaR_i])
    print('x0 = {}'.format(x0))
    print('Initial nll = {}'.format(nll_gaus2(x0,Rnorm)))
    fitCut = np.logical_or(Rnorm < np.percentile(Rnorm, 25), Rnorm > np.percentile(Rnorm, 75))
    params = minimize(nll_gaus2, x0=x0, args=(Rnorm[fitCut],))
    print('NLL fits are {0}'.format(params))
    results = params.x
    if results[4] > results[1]:
        results = [results[3], results[4], results[5], results[0], results[1], results[2]]
    print(results)
    xModel = np.linspace(Rnorm.min(),Rnorm.max(),1e4)
    hist_plot(Rnorm, xModel, gaus2(xModel, params.x[0], params.x[1], params.x[2], params.x[3], params.x[4], params.x[5]), 'Resistance Distributions for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'R_hist' + '_ch_' + str(int(ch)) + '_' + Istring, results, 2)
    
    #Rerr = np.sqrt( (0.001*Rm + 0.00005*2)**2 + (1e-3)**2 + (results[2]**2 + results[5]**2)/2)
    RP = 0.1/100
    RA = 1e-3
    #Rerr = np.sqrt( (RP*Rm + 0.003*2)**2 + (RA)**2 + (results[2]**2 + results[5]**2)/2)
    Rerr = np.sqrt((results[2]**2 + results[5]**2)/2)*np.ones(Rm.shape)
    #Rerr = (0.01*(np.min(Rm) + np.max(Rm))/2)*np.ones(Rm.shape)
    if mFilter:
        Rerr = newRmrms
    RerrType = 'absolute'
    #Rerr = None
    #dTm = None
    print('The resistance error is Rerr: {0}'.format(Rerr))
    if fit_method == 'SLSQP':
        x0 = [1*Rmax, 1*Rmin, (Tc_min), Tc_min]
        print('The initial guess is {0}'.format(x0))
        result = minimize(nll,x0=np.array(x0),args=(Rm,Tm,fDict[func2fit]),method='SLSQP',bounds=[ (0, 3*Rmax), (0, np.max([1000*Rmin, Rmax])), (Tc_min, Tc_max), (Tw_min, Tc_max) ])
        perr = result.jac
        result = result.x
    elif fit_method == 'Nelder-Mead':
        x0 = [1*Rmax, 1*Rmin, (Tc_min), 10*Tw_min]
        print('The initial guess is {0}'.format(x0))
        #result = minimize(nll, x0=np.array(x0), args=(Rm,Tm,fDict[func2fit]), method='Nelder-Mead', options={'fatol':1e-11})
        result = minimize(nll_error, x0=np.array(x0), args=(Rm,Rerr,Tm,fDict[func2fit]), method='Nelder-Mead', options={'fatol':1e-11})
        perr = result.x/100
        result = result.x
    elif fit_method == 'curve':
        if func2fit == 'tanh_lin':
            a,b,c = [0, 0, 0]
            x0 = [1*Rmin, (Tc_min), Tw_min, a, b, c]
        else:
            x0 = [1*Rmin, (Tc_min), Tw_min]
        print('The initial guess is {0}'.format(x0))
        # Use curve fit package
        #Reminder: [Rn, Rsc, Tc, Tw]
        if func2fit == 'tanh_lin':
            a, b, c = [0, 0, 0]
            lbounds = [-np.max([1000*Rmin, Rmax]), Tc_min, min([Tc_min,4*Tw_min]), a, b]
            a, b, c = [1e4, 10, 5]
            ubounds = [np.max([1000*Rmin, Rmax]), Tc_max, max([Tc_min, 4*Tw_min]), a, b]
        else:
            lbounds = [0.5*Rmax, -np.max([1000*Rmin, Rmax]), Tc_min, 4*Tw_min]
            ubounds = [3*Rmax, np.max([1000*Rmin, Rmax]), Tc_max, Tc_min]
        print('The lower bounds are {}'.format(lbounds))
        print('The upper bounds are {}'.format(ubounds))
        result, pcov = curve_fit(fDict[func2fit], Tm, Rm, bounds=(lbounds,ubounds), sigma=Rerr, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
    elif fit_method == 'odr':
        x0 = [1*Rmax, 1*Rmin, (Tc_max+Tc_min)/2, Tw_min]
        print('The initial guess is {0}'.format(x0))
        # Use orthogonal distance regression instead of linear distance
        data = RealData(Tm, Rm, sx=dTm, sy=Rerr)
        model = Model(tanh_tc2)
        odr = ODR(data, model, x0)
        odr.set_job(fit_type=0)
        result = odr.run()
        perr = np.sqrt(np.diag(result.cov_beta))
        result = result.beta
    print(result)
    if fit_method == 'curve':
        print(perr)
    
    # Plot result
    Tfit = T[cUse][idx]
    if func2fit == 'tanh_lin':
        Rfit = fDict[func2fit](Tfit, result[0],result[1],result[2],result[3], result[4])
        Rfitm = fDict[func2fit](Tm,result[0],result[1],result[2],result[3], result[4])
    else:
        Rfit = fDict[func2fit](Tfit,result[0],result[1],result[2],result[3])
        Rfitm = fDict[func2fit](Tm,result[0],result[1],result[2],result[3])
    gen_fitplot(Tm, Rm, dTm, Rerr, RerrType, Rfitm, Tfit, Rfit, result, perr, TPrefix + ' Temp [K]', 'Resistance [Ohm]', 'Resistance vs Temperature for Ch ' + str(ch) + ' with ' + r'$\mathsf{I_e = }$' + Istring, outPath + '/' + 'R_vs_' + fPrefix + '_fit' + '_ch_' + str(int(ch)) + '_' + Istring, func2fit, 'Linear')
    if bDump:
        fn = outPath + '/' + 'woH_ch_' + str(ch) + '_' + Istring + '_' + fPrefix + '.txt'
        dump2text(nR, T, fn)