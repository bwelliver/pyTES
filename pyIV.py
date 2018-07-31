import os
import argparse
from os.path import isabs
from os.path import dirname
from os.path import basename

import numpy as np
from numpy import exp as exp
from numpy import log as ln
from numpy import square as pow2
from numpy import sqrt as sqrt
from numpy import sum as nsum

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData

import matplotlib as mp
from matplotlib import pyplot as plt

import ROOT as rt

from readROOT import readROOT
from writeROOT import writeROOT

eps = np.finfo(float).eps
ln = np.log

#mp.use('agg')

class FitParameters:
    
    def __init__(self):
        self.left = FitResult()
        self.right = FitResult()
        self.parasitic = FitResult()
    def __repr__(self):
        """return string representation."""
        s = 'Left Branch:\n' + str(self.left)
        s = s + '\n' + 'Right Branch:\n' + str(self.right)
        s = s + '\n' + 'Parasitic:\n' + str(self.parasitic)
        return(s)


class FitResult:
    
    def __init__(self):
        self.result = None
        self.error = None
    def set_values(self, result=None, error=None):
        self.result = result
        self.error = error
    def __repr__(self):
        """return string representation."""
        s = '\t' + 'Result:\t' + str(self.result)
        s = s + '\n' + '\t' + 'Error:\t' + str(self.error)
        return(s)


class TESResistance:
    
    def __init__(self, left=None, right=None, parasitic=None):
        self.left = left
        self.right = right
        self.parasitic = parasitic
        
    def get_all(self):
        """return a list of elements from the newest to the oldest (left to
        right)"""
        return(self)
    
    def __repr__(self):
        """return string representation."""
        s = 'Left Branch:\t' + str(self.left) + ' Ohms'
        s = s + '\n' + 'Right Branch:\t' + str(self.right) + ' Ohms'
        s = s + '\n' + 'Parasitic:\t' + str(self.parasitic) + ' Ohms'
        return(s)


class IVData:
    
    def __init__(self, i=None, iRMS=None, v=None, vRMS=None):
        self.i = i
        self.iRMS = iRMS
        self.v = v
        self.vRMS = vRMS
    
    def compute_resistance(self, left=0, right=-1):
        '''Compute the resistance in a particular event range. Returns resistance and RMS of resistance'''
        rRMS = np.sqrt((self.vRMS[left:right]/self.v[left:right])**2 + (self.iRMS[left:right]/self.i[left:right])**2)
        r = self.v[left:right]/self.i[left:right]
        return r, rRMS



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

    def get_std(self):
        '''std of the current values'''
        return np.std(self._data)

    def argmax(self):
        """return index of first occurence of max value."""
        return(np.argmax(self._data))

    def get_all(self):
        """return a list of elements from the newest to the oldest (left to
        right)"""
        return(self._data)

    def get_partial(self):
        return(self.get_all()[0:self.size])

    def get_size(self):
        return(np.size(self._data))

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
    return True


def get_treeNames(input_file):
    '''Quick and dirty function to get name of trees'''
    tFile = rt.TFile.Open(input_file)
    #tDir = tFile.Get(directory)
    keys = tFile.GetListOfKeys()
    keyList = [key.GetName() for key in keys]
    del tFile
    return keyList


def lin_sq(x, m, b):
    '''Get the output for a linear response to an input'''
    y = m*x + b
    return y


def quad_sq(x, a, b, c):
    '''Get the output for a quadratic response to an input'''
    y = a*x*x + b*x + c
    return y


def get_squid_parameters(channel):
    '''Return SQUID Parameters based on a given channel'''
    squid_dictionary = {
        2: {
            'Serial': 'S0121',
            'Li': 6e-9,
            'Min': 1/26.062,
            'Mf': 1/33.27,
            'Rfb': 1e4,
            'Rsh': 21e-3,
            'Rbias': 1e4,
            'Cbias': 150e-12
        },
        3: {
            'Serial': 'S0094',
            'Li': 6e-9,
            'Min': 1/23.99,
            'Mf': 1/32.9,
            'Rfb': 1e4,
            'Rsh': 22.8e-3,
            'Rbias': 1e4,
            'Cbias': 150e-12
        }
    }
    # Compute auxillary SQUID parameters based on ratios
    for key in squid_dictionary.keys():
        squid_dictionary[key]['M'] = -squid_dictionary[key]['Min']/squid_dictionary[key]['Mf']
        squid_dictionary[key]['Lf'] = squid_dictionary[key]['M']*squid_dictionary[key]['M']*squid_dictionary[key]['Li']
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


def power_temp(T, t_tes, k, n):
    P = k*(np.power(t_tes, n) - np.power(T, n))
    return P


def power_temp5(T, t_tes, k):
    P = k*(np.power(t_tes, 5) - np.power(T, 5))
    return P


def fillBuffer(buffer, offset, data):
    '''Fill a buffer'''
    for idx in range(buffer.size_max):
        buffer.append(data[idx + offset])
    return buffer


def find_end(pBuff, cBuff, fBuff, Tstep, vTemp, ev):
    '''find an ending step boundary'''
    # Compute dMu_ij pairs
    stepEv = None
    while ev < vTemp.size - 1:
        dMu_pc = np.abs(pBuff.get_mean() - cBuff.get_mean())
        dMu_cf = np.abs(cBuff.get_mean() - fBuff.get_mean())
        dMu_pf = np.abs(pBuff.get_mean() - fBuff.get_mean())
        #print(pBuff.get_mean(), cBuff.get_mean(), fBuff.get_mean())
        #print(dMu_pc, dMu_cf, dMu_pf)
        # Test if past and future are similar...if these are then we assume current is similar and so advance
        bPC = dMu_pc < 0.5*Tstep
        bPF = dMu_pf < 0.5*Tstep
        bCF = dMu_cf < 0.5*Tstep
        if bPC and not bCF:
            # This is the return case...current similar to past but not future
            # Make the stepEv be the first event of the future
            stepEv = ev - fBuff.get_size()
            print('The step event is {}'.format(stepEv))
            return [ev, stepEv, pBuff, cBuff, fBuff]
        else:
            # past and future are within 1/2 Tstep so increment all things by 1 event
            # other conditions too but for now increment by 1 event
            oF = fBuff.get_all()[-1]
            oC = cBuff.get_all()[-1]
            pBuff.append(oC)
            cBuff.append(oF)
            fBuff.append(vTemp[ev])
            ev += 1
    return [ev - 1, ev - 1, pBuff, cBuff, fBuff]


def find_start(pBuff, cBuff, fBuff, Tstep, vTemp, ev):
    '''find a starting step boundary'''
    # Compute dMu_ij pairs
    stepEv = None
    while ev < vTemp.size - 1:
        dMu_pc = np.abs(pBuff.get_mean() - cBuff.get_mean())
        dMu_cf = np.abs(cBuff.get_mean() - fBuff.get_mean())
        dMu_pf = np.abs(pBuff.get_mean() - fBuff.get_mean())
        #print(pBuff.get_mean(), cBuff.get_mean(), fBuff.get_mean())
        #print(dMu_pc, dMu_cf, dMu_pf)
        # Test if past and future are similar...if these are then we assume current is similar and so advance
        bPC = dMu_pc < 0.5*Tstep
        bPF = dMu_pf < 0.5*Tstep
        bCF = dMu_cf < 0.5*Tstep
        if bCF and not bPC:
            # This is the return case...current similar to future but not past
            # Make the stepEv be the first event of the current
            stepEv = ev - fBuff.get_size() - cBuff.get_size()
            #print('The step event is {}'.format(stepEv))
            return [ev, stepEv, pBuff, cBuff, fBuff]
        else:
            # past and future are within 1/2 Tstep so increment all things by 1 event
            # other conditions too but for now increment by 1 event
            #print('The 3 dMu values for start search are: {}'.format([dMu_pc, dMu_cf, dMu_pf]))
            oF = fBuff.get_all()[-1]
            oC = cBuff.get_all()[-1]
            pBuff.append(oC)
            cBuff.append(oF)
            fBuff.append(vTemp[ev])
            ev += 1
    return [ev - 1, ev - 1, pBuff, cBuff, fBuff]

def getStabTemp(vTime, vTemp, lenBuff=10, Tstep=5e-5):
    '''Function that attempts to find temperature steps and select periods of relatively stable T
    This method will create 3 sliding windows and compute the mean in each window.
    lenBuff - the length of the ring buffer (seconds)
    Tstep - the step change in temperature (K) to be sensitive to
    pBuff - the leftmost window and what defines "the past"
    cBuff - the center window and what defines the window of interest
    fBuff - the rightmost window and what defines "the future"
    Note that smaller lenBuff means we generally lose sensitivity to larger values of Tstep
    Values that seem OK are (10, 5e-5), (60, 2e-4), (90, 5e-4)
    We compute pMu, cMu and fMu and then decide which region cMu is similar to
    metric: dMu_ij = |mu_i - mu_j|
    compute for all pairs: p-c, c-f, p-f
    if dMu_ij < 0.5*T_step consider mu_i ~ mu_j
    otherwise means are not the same so a boundary has occurred.
    We start with t = 0 as one boundary and create boundaries in pairs
    First boundary we find will be the end of the first region
    Once an end boundary is found we set pBuff = fBuff
    Note: if all dMu combinations are dissimilar we're in a transition region
    '''
    pBuff = RingBuffer(lenBuff, dtype=float)
    cBuff = RingBuffer(lenBuff, dtype=float)
    fBuff = RingBuffer(lenBuff, dtype=float)
    # Fill the buffers initially
    ev = 0
    tList = []
    tStart = vTime[0]
    while ev < vTemp.size - 1 and ev + lenBuff < vTemp.size - 1:
        # We start with assuming the first window starts at t = vTime[0]
        if len(tList) == 0:
            tStart = vTime[0]
            # Start pBuff at ev = 0
            pBuff = fillBuffer(pBuff, ev, vTemp)
            ev += lenBuff + 1
            cBuff = fillBuffer(cBuff, ev, vTemp)
            ev += lenBuff + 1
            fBuff = fillBuffer(fBuff, ev, vTemp)
            ev += lenBuff + 1
            # Now proceed to find an end
        else:
            # we have now found an end so now need to find a new start
            # We need new windows first
            # When we find an end point past ~ current !~ future
            # So adjust past <-- current, current <-- future, future <-- vTemp
            oC = np.flip(cBuff.get_all(), 0)
            oF = np.flip(fBuff.get_all(), 0)
            for idx in oC:
                pBuff.append(idx)
            for idx in oF:
                cBuff.append(idx)
            fBuff = fillBuffer(fBuff, ev, vTemp)
            ev += lenBuff + 1
            ev, stepEv, pBuff, cBuff, fBuff = find_start(pBuff, cBuff, fBuff, Tstep, vTemp, ev)
            tStart = vTime[stepEv]
        # Now we have a tStart so we need to find a tEnd.
        # When we find a start point, past !~ current ~ future
        # we can keep sliding forward until we reach past ~ current !~ future
        ev, stepEv, pBuff, cBuff, fBuff = find_end(pBuff, cBuff, fBuff, Tstep, vTemp, ev)
        tEnd = vTime[stepEv]
        cut = np.logical_and(vTime >= tStart, vTime <= tEnd)
        mTemp = np.mean(vTemp[cut])
        tri = (tStart, tEnd, mTemp)
        tList.append(tri)
    return tList


def walk_normal(x, y, side, buffer_size=20):
    '''Function to walk the normal branches and find the line fit
    To do this we will start at the min or max input current and compute a walking derivative
    If the derivative starts to change then this indicates we entered the biased region and should stop
    NOTE: We assume data is sorted by voltage values
    '''
    
    # First let us compute the gradient (dy/dx)
    dydx = np.gradient(y, x, edge_order=2)
    if side == 'right':
        # Flip the array
        dydx = dydx[::-1]
    # In the normal region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time. 
    # If the new average differs from the previous by some amount mark that as the boundary to the bias region
    dbuff = RingBuffer(buffer_size, dtype=float)
    for ev in range(buffer_size):
        dbuff.append(dydx[ev])
    # Now our buffer is initialized so loop over all events until we find a change
    ev = buffer_size
    dMean = 0
    dev = 0
    while dMean < 5e-1 and ev < dydx.size - 1 and dev < 20:
        currentMean = dbuff.get_mean()
        dbuff.append(dydx[ev])
        newMean = dbuff.get_mean()
        dMean = np.abs((currentMean - newMean)/currentMean)
        ev += 1
        dev += 1
    if side == 'right':
        # Flip event index back the right way
        ev = dydx.size - 1 - ev
    #print('The {} deviation occurs at ev = {} with current = {} and voltage = {} with dMean = {}'.format(side, ev, current[ev], voltage[ev], dMean))
    return ev


def walk_sc(x, y, buffer_size=4):
    '''Function to walk the superconducting region of the IV curve and get the left and right edges
    Generally when ib = 0 we should be superconducting so we will start there and go up until the bias
    then return to 0 and go down until the bias
    '''
    # First let us compute the gradient (i.e. dy/dx)
    dydx = np.gradient(y, x, edge_order=2)
    # In the sc region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time. 
    # If the new average differs from the previous by some amount mark that as the end.
    
    # First we should find whereabouts of (0,0)
    # This should roughly correspond to x = 0 since if we input nothing we should get out nothing. In reality there are parasitics of course
    index_min_x = np.argmin(np.abs(x))
    # Occasionally we may have a shifted curve that is not near 0 for some reason (SQUID jump)
    # So find the min and max iTES and then find the central point
    index_max_y = np.argmax(y)
    index_min_y = np.argmin(y)
    mean_x = (x[index_max_y] + x[index_min_y])/2
    # Find the index of x nearest this mean value
    index_min_x = np.argmin(np.abs(x-mean_x))
    # NOTE: The above will fail for small SC regions where iTES normal > iTES sc!!!!
    
    # First go from index_min_x and increase
    # Create ring buffer of to store signal
    buffer_size = 4
    #TODO: FIX THIS TO HANDLE SQUID JUMPS
    dbuff = RingBuffer(buffer_size, dtype=float)
    # Start by walking buffer_size events to the right from the minimum abs. voltage
    for ev in range(buffer_size):
        dbuff.append(dydx[index_min_x + ev])
    # Now our buffer is initialized so loop over all events until we find a change
    ev = index_min_x + buffer_size
    dMean = 0
    while dMean < 1e-2 and ev < dydx.size - 1:
        currentMean = dbuff.get_mean()
        dbuff.append(dydx[ev])
        newMean = dbuff.get_mean()
        dMean = np.abs((currentMean - newMean)/currentMean)
        ev += 1
    #print('The right deviation occurs at ev = {} with current = {} and voltage = {} with dMean = {}'.format(ev, current[ev], voltage[ev], dMean))
    evRight = ev
    # Now repeat but go to the left from the minimum abs. voltage
    buffer_size = 4
    dbuff = RingBuffer(buffer_size, dtype=float)
    for ev in range(buffer_size):
        dbuff.append(dydx[index_min_x - ev])
    # Now our buffer is initialized so loop over all events until we find a change
    ev = index_min_x - buffer_size
    dM = 0
    while dMean < 1e-2 and ev < didv.size - 1:
        currentMean = dbuff.get_mean()
        dbuff.append(dydx[ev])
        newMean = dbuff.get_mean()
        dMean = np.abs((currentMean - newMean)/currentMean)
        ev -= 1
    #print('The left deviation occurs at ev = {} with current = {} and voltage = {} with dMean = {}'.format(ev, current[ev], voltage[ev], dMean))
    evLeft = ev
    return (evLeft, evRight)


# Plot root files
def make_gen_plot(x, y, xlab, ylab, titleStr, fName, logx='linear', logy='linear'):
    '''General error plotting function'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xscale(logx)
    ax.set_yscale(logy)
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    ax.set_title(titleStr, fontsize=18)
    ax.set_ylim((0.95*y.min(), 1.05*y.max()))
    if x.min() > 0:
        ax.set_xlim((0.95*x.min(), 1.05*x.max()))
    else:
        ax.set_xlim((1.05*x.min(), 1.05*x.max()))
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return True


def make_gen_fitplot(x, y, xfit, yfit, xlab, ylab, titleStr, fName, logx='linear', logy='linear', y0=False):
    '''General error plotting function'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    for xf, yf in zip(xfit, yfit):
        ax.plot(xf, yf, '-', marker='None', linewidth=2)
    ax.set_xscale(logx)
    ax.set_yscale(logy)
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    ax.set_title(titleStr, fontsize=18)
    if y.min() > 0:
        ax.set_ylim(0.95*y.min(), 1.05*y.max())
    else:
        ax.set_ylim((1.05*y.min(), 1.05*y.max()))
    if y0 is True:
        ax.set_ylim((0, 1.05*y.max()))

    if x.min() > 0:
        ax.set_xlim((0.95*x.min(), 1.05*x.max()))
    else:
        ax.set_xlim((1.05*x.min(), 1.05*x.max()))
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return True


def make_power_voltage_fit(x, y, xf, yf, fitResults, xLabel, yLabel, titleStr, fName, xErr=None, yErr=None, logx='linear', logy='linear', y0=False):
    '''Generate power vs voltage with fit'''
    [[R,iL,p], [Rerr, iLerr, perr]] = fitResults
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    if xErr is not None or yErr is not None:
        ax.errorbar(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', xerr=xErr, yerr=yErr)
    else:
        ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    
    ax.plot(xf, yf, '-', marker='None', linewidth=2)
    ax.set_xscale(logx)
    ax.set_yscale(logy)
    ax.set_xlabel(xLabel, fontsize=18)
    ax.set_ylabel(yLabel, fontsize=18)
    ax.set_title(titleStr, fontsize=18)
    if y.min() > 0:
        ax.set_ylim(0.95*y.min(), 1.05*y.max())
    else:
        ax.set_ylim((1.05*y.min(), 1.05*y.max()))
    if y0 is True:
        ax.set_ylim((0, 1.05*y.max()))

    if x.min() > 0:
        ax.set_xlim((0.95*x.min(), 1.05*x.max()))
    else:
        ax.set_xlim((1.05*x.min(), 1.05*x.max()))
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    tR = r'$\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R*1e3, Rerr*1e3)
    tI = r'$\mathrm{i_{p}} = %.5f \pm %.5f \mathrm{nA}$'%(iL*1e9, iLerr*1e9)
    tP = r'$\mathrm{P_{p}} = %.5f \pm %.5f \mathrm{fW}$'%(p*1e15, perr*1e15)
    textStr = tR + '\n' + tI + '\n' + tP
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return True
    
def test_plot(x, y, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
    #ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    #ax.set_title(title)
    ax.grid()
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    #plt.draw()
    #plt.show()


def make_gen_errplot(x, xerr, y, yerr, xlab, ylab, titleStr, fName, log='linear'):
    '''General error plotting function'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', xerr=xerr, yerr=yerr)
    ax.set_xscale(log)
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    ax.set_title(titleStr, fontsize=18)
    ax.set_ylim((0.95*y.min(), 1.05*y.max()))
    if x.min() > 0:
        ax.set_xlim((0.95*x.min(), 1.05*x.max()))
    else:
        ax.set_xlim((1.05*x.min(), 1.05*x.max()))
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return True


def test_steps(x, y, v, t0, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
    # Next add horizontal lines for each step it thinks it found
    for item in v:
        ax.plot([item[0]-t0,item[1]-t0], [item[2], item[2]], marker='.', linestyle='-', color='r')
    #ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    #ax.set_title(title)
    ax.grid()
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    #plt.draw()
    #plt.show()
    return None


def generic_fitplot_with_errors(ax, x, y, labels, params, xScale=1, yScale=1, logx='linear', logy='linear'):
    '''A helper function that puts data on a specified axis'''
    out = ax.errorbar(x*xScale, y*yScale, **params)
    ax.set_xscale(logx)
    ax.set_yscale(logy)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])
    ax.set_title(labels['title'])
    ax.grid()
    return ax


def add_model_fits(ax, x, y, model, sc_bounds=None, xScale=1, yScale=1, model_function=lin_sq):
    '''Add model fits to plots'''
    if model.left.result is not None:
        yFit = model_function(x, *model.left.result)
        ax.plot(x*xScale, yFit*yScale, 'r-', marker='None', linewidth=2)
    if model.right.result is not None:
        yFit = model_function(x, *model.right.result)
        ax.plot(x*xScale, yFit*yScale, 'g-', marker='None', linewidth=2)
    if model.parasitic.result is not None:
        sc_bounds = (0, -1) if sc_bounds is None else sc_bounds
        if sc_bounds[0] > sc_bounds[1]:
            sc_bounds = (sc_bounds[1], sc_bounds[0])
        yFit = model_function(x[sc_bounds[0]:sc_bounds[1]], *model.parasitic.result)
        ax.plot(x[sc_bounds[0]:sc_bounds[1]]*xScale, yFit*yScale, 'b-', marker='None', linewidth=2)
    return ax


def add_fit_textbox(ax, R, Rerr, model):
    '''Add decoration textbox to a plot'''
    
    lR = r'$\mathrm{Left R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.left*1e3, Rerr.left*1e3)
    lOff = r'$\mathrm{Left V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)
    
    sR = r'$\mathrm{SC R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.parasitic*1e3, Rerr.parasitic*1e3)
    sOff = r'$\mathrm{SC V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.parasitic.result[1]*1e3, model.parasitic.error[1]*1e3)
    
    rR = r'$\mathrm{Right R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.right*1e3, Rerr.right*1e3)
    rOff = r'$\mathrm{Right V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)
    
    textStr = lR + '\n' + lOff + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    out = ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return ax


def add_power_textbox(ax, model):
    '''Add dectoration textbox for a power vs resistance fit'''
    lR = r'$\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(1/model.left.result[0]*1e3, model.left.error[0]/pow2(model.left.result[0])*1e3)
    lI = r'$\mathrm{I_{para}} = %.5f \pm %.5f \mathrm{uA}$'%(model.left.result[1]*1e6, model.left.error[1]*1e6)
    lP = r'$\mathrm{P_{para}} = %.5f \pm %.5f \mathrm{fW}$'%(model.left.result[2]*1e15, model.left.error[2]*1e15)
    
    textStr = lR + '\n' + lI + '\n' + lP
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return ax


def save_plot(fig, ax, fName):
    '''Save a specified plot'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return None
    

def iv_fitplot(x, y, xerr, yerr, fitParams, xLabel, yLabel, title, fName, sc=None, xScale=1, yScale=1, logx='linear', logy='linear'):
    '''Wrapper for plotting an iv curve with fit parameters'''
    R, Rerr, model = fitParams
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    ax.errorbar(x*xScale, y*yScale, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0, linestyle='None', xerr=xerr*xScale, yerr=yerr*yScale)
    if model.left.result is not None:
        yFit = lin_sq(x, model.left.result[0], model.left.result[1])
        ax.plot(x*xScale, yFit*yScale, 'r-', marker='None', linewidth=2)
    if model.right.result is not None:
        yFit = lin_sq(x, model.right.result[0], model.right.result[1])
        ax.plot(x*xScale, yFit*yScale, 'g-', marker='None', linewidth=2)
    if model.parasitic.result is not None:
        sc = (0, -1) if sc is None else sc
        if sc[0] > sc[1]:
            sc = (sc[1], sc[0])
        yFit = lin_sq(x[sc[0]:sc[1]], model.parasitic.result[0], model.parasitic.result[1])
        ax.plot(x[sc[0]:sc[1]]*xScale, yFit*yScale, 'b-', marker='None', linewidth=2)
    ax.set_xscale(logx)
    ax.set_yscale(logy)
    ax.set_xlabel(xLabel, fontsize=18)
    ax.set_ylabel(yLabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    #ax.set_ylim((0.95*y.min(), 1.05*y.max()))
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    # Now generate text strings
    # model values are [results, perr] --> [[m, b], [merr, berr]]
    lR = r'$\mathrm{Left R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.left*1e3, Rerr.left*1e3)
    lOff = r'$\mathrm{Left V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)
    
    sR = r'$\mathrm{SC R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.parasitic*1e3, Rerr.parasitic*1e3)
    sOff = r'$\mathrm{SC V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.parasitic.result[1]*1e3, model.parasitic.error[1]*1e3)
    
    rR = r'$\mathrm{Right R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.right*1e3, Rerr.right*1e3)
    rOff = r'$\mathrm{Right V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)
    
    textStr = lR + '\n' + lOff + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return None


def gen_fitplot(x, y, xerr, yerr, lFit, rFit, scFit, xlab, ylab, title, fName, log='linear', model='y'):
    '''Generate fitplot for some data'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', xerr=xerr, yerr=yerr)
    if lFit != None:
        if model == 'y':
            ax.plot(x, lFit['model'], 'r-', marker='None', linewidth=2)
        elif model == 'x':
            ax.plot(lFit['model'], y, 'r-', marker='None', linewidth=2)
    if rFit != None:
        if model == 'y':
            ax.plot(x, rFit['model'], 'g-', marker='None', linewidth=2)
        elif model == 'x':
            ax.plot(rFit['model'], y, 'g-', marker='None', linewidth=2)
    # do not plot all of the sc model
    if scFit != None:
        if model == 'y':
            cut = np.logical_and(scFit['model'] <= np.max(y), scFit['model'] >= np.min(y))
            ax.plot(x[cut], scFit['model'][cut], 'b-', marker='None', linewidth=2)
        elif model == 'x':
            cut = np.logical_and(scFit['model'] <= np.max(x), scFit['model'] >= np.min(x))
            ax.plot(scFit['model'][cut], y[cut], 'b-', marker='None', linewidth=2)
    ax.set_xscale(log)
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.set_ylim((0.95*y.min(), 1.05*y.max()))
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    begin = '$'
    lR = '$\mathrm{R_{nL}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(lFit['result'][0]*1e3, lFit['perr'][0]*1e3)
    lOff = '$\mathrm{i_{offL}} = %.5f \pm %.5f \mathrm{\mu A}$'%(lFit['result'][1]*1e6, lFit['perr'][1]*1e6)
    scR = '$\mathrm{R_{SC}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(scFit['result'][0]*1e3, scFit['perr'][0]*1e3)
    scOff = '$\mathrm{i_{offSC}} = %.5f \pm %.5f \mathrm{\mu A}$'%(scFit['result'][1]*1e6, scFit['perr'][1]*1e6)
    rR = '$\mathrm{R_{nR}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(rFit['result'][0]*1e3, rFit['perr'][0]*1e3)
    rOff = '$\mathrm{i_{offR}} = %.5f \pm %.5f \mathrm{\mu A}$'%(rFit['result'][1]*1e6, rFit['perr'][1]*1e6)
    ending = '$'
    textStr = lR + '\n' + lOff + '\n' + scR + '\n' + scOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    #plt.draw()
    #plt.show()
    return True


def read_from_ivroot(filename):
    '''Read data from special IV root file and put it into a dictionary
    TDir - iv
        TTrees - temperatures
            TBranches - iv properties
    '''
    print('Trying to open {}'.format(filename))
    iv_dictionary = {}
    treeNames = get_treeNames(filename)
    branches = ['iBias', 'iBias_rms', 'vOut', 'vOut_rms']
    #branches = ['iv/' + branch for branch in branches]
    method = 'single'
    for treeName in treeNames:
        print('Trying to get tree {} and branches {}'.format(treeName, branches))
        rdata = readROOT(filename, treeName, branches, method)
        iv_dictionary[treeName] = rdata['data']
    return iv_dictionary


def save_to_root(output_directory, iv_dictionary):
    '''Function to save iv data to a root file
    Here we will let the temperatures be the name of TTrees
    Branches of each TTree will be the various iv components
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
    data = {'TTree': {} }
    for temperature, iv_data in iv_dictionary.items():
        data['TTree'][temperature] = {'TBranch': {} }
        for key, value in iv_data.items():
            data['TTree'][temperature]['TBranch'][key] = value
    #print(data)
    # We should also make an object that tells us what the other tree values are
    #data['TDirectory']['iv']['TTree']['names']['TBranch'] = 
    mkdpaths(output_directory + '/root')
    outFile = output_directory + '/root/iv_data.root'
    writeROOT(outFile, data)
    return None


def make_tes_plots(output_path, data_channel, temperature, sc_bounds, data, fit_parameters):
    '''Helper function to generate standard TES plots
    iTES vs vTES
    rTES vs iTES
    rTES vs vTES
    pTES vs rTES
    pTES vs vTES
    '''
    rTES = get_rTES(data['iTES'], data['vTES'])
    rTES_rms = get_rTES_rms(data['iTES'], data['iTES_rms'], data['vTES'], data['vTES_rms'])
    pTES = get_pTES(data['iTES'], data['vTES'])
    pTES_rms = get_pTES_rms(data['iTES'], data['iTES_rms'], data['vTES'], data['vTES_rms'])
    # Current vs Voltage
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e6
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': data['iTES_rms']*yScale}
    labels = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Current [uA]', 'title': 'Channel {} TES Current vs TES Voltage for T = {} mK'.format(data_channel, temperature)}
    
    ax = generic_fitplot_with_errors(ax=ax, x=data['vTES'], y=data['iTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    ax = add_model_fits(ax=ax, x=data['vTES'], y=data['iTES'], model=fit_parameters, sc_bounds=sc_bounds, xScale=xScale, yScale=yScale, model_function=lin_sq)
    ax = add_fit_textbox(ax=ax, R=data['rTES'], Rerr=data['rTES_err'], model=fit_parameters)
    
    fName = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    
    # Resistance vs Current
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['iTES_rms']*xScale, 'yerr': rTES_rms*yScale}
    labels = {'xlabel': 'TES Current [uA]', 'ylabel': 'TES Resistance [mOhm]', 'title': 'Channel {} TES Resistance vs TES Current for T = {} mK'.format(data_channel, temperature)}
    ax = generic_fitplot_with_errors(ax=ax, x=data['iTES'], y=rTES, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'rTES_vs_iTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    # Resistance vs Voltage
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': rTES_rms*yScale}
    labels = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Resistance [mOhm]', 'title': 'Channel {} TES Resistance vs TES Voltage for T = {} mK'.format(data_channel, temperature)}
    ax = generic_fitplot_with_errors(ax=ax, x=data['vTES'], y=rTES, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    # Power vs rTES
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e12
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': rTES_rms*xScale, 'yerr': pTES_rms*yScale}
    labels = {'xlabel': 'TES Resistance [mOhm]', 'ylabel': 'TES Power [pW]', 'title': 'Channel {} TES Power vs TES Resistance for T = {} mK'.format(data_channel, temperature)}
    ax = generic_fitplot_with_errors(ax=ax, x=rTES, y=pTES, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'pTES_vs_rTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    # Power vs vTES
    # Note this ideally is a parabola
    cut = rTES > 100e-3
    if nsum(cut) < 3:
        cut = np.ones(pTES.size, dtype=bool)
    v = data['vTES'][cut]
    p = pTES[cut]
    prms = pTES_rms[cut]
    print('For T = {} mK'.format(temperature))
    print('The cut has {} events, vTES has {} events and pTES has {} events'.format(nsum(cut), v.size, p.size))
    result, pcov = curve_fit(quad_sq, v, p, sigma=prms, absolute_sigma=True)
    print('R = {} mOhm, Ipara = {} uA, Ppara = {} fW'.format(1/result[0] * 1e3, result[1]*1e6, result[2]*1e15))
    perr = np.sqrt(np.diag(pcov))
    pFit = quad_sq(data['vTES'], result[0], result[1], result[2])
    fitResult = FitParameters()
    fitResult.left.set_values(result, perr)
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e12
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': pTES_rms*yScale}
    labels = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Power [pW]', 'title': 'Channel {} TES Power vs TES Resistance for T = {} mK'.format(data_channel, temperature)}
    
    ax = generic_fitplot_with_errors(ax=ax, x=data['vTES'], y=pTES, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    ax = add_model_fits(ax=ax, x=data['vTES'], y=pTES, model=fitResult, sc_bounds=None, xScale=xScale, yScale=yScale, model_function=quad_sq)
    ax = add_power_textbox(ax=ax, model=fitResult)
    
    fName = output_path + '/' + 'pTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    #iv_fitplot(data['vTES'], data['iTES'], data['vTES_rms'], data['iTES_rms'], [data['rTES'], data['rTES_err'], fit_parameters], xLabel, yLabel, titleStr, fName, sc=sc_bounds, xScale=1e6, yScale=1e6, logx='linear', logy='linear')
    return None
        

#@obsolete
#def make_TES_plots(outPath, data_channel, mean_temperature, tes_values, vF_left, vF_sc, vF_right):
#    '''Make TES plots for various quantities'''
#    # First unpack tes_values
#    iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms = tes_values
#    
#    # Make iTES vs vTES plot
#    xLabel = 'TES Voltage [uV]'
#    yLabel = 'TES Current [uA]'
#    T = np.round(mean_temperature*1e3,2)
#    titleStr = 'Channel {} TES Current vs TES Voltage for T = {} mK'.format(data_channel, T)
#    fName = outPath + '/' + 'iTES_vs_vTES_ch_' + str(data_channel) + '_' + str(T) + 'mK'
#    gen_fitplot(vTES*1e6, iTES*1e6, vTES_rms*1e6, iTES_rms*1e6, vF_left, vF_right, vF_sc, xLabel, yLabel, titleStr, fName, log='linear', model='x')
#    
#    # Make rTES vs vTES plot
#    xLabel = 'TES Voltage [uV]'
#    yLabel = 'TES Resistance [mOhm]'
#    T = np.round(mean_temperature*1e3,2)
#    titleStr = 'Channel {} TES Resistance vs TES Voltage for T = {} mK'.format(data_channel, T)
#    fName = outPath + '/' + 'rTES_vs_vTES_ch_' + str(data_channel) + '_' + str(T) + 'mK'
#    make_gen_errplot(vTES*1e6, vTES_rms*1e6, rTES*1e3, rTES_rms*1e3, xLabel, yLabel, titleStr, fName, log='linear')
#    
#    # Make rTES vs iTES plot
#    xLabel = 'TES Current [uA]'
#    yLabel = 'TES Resistance [mOhm]'
#    T = np.round(mean_temperature*1e3,2)
#    titleStr = 'Channel {} TES Resistance vs TES Current for T = {} mK'.format(data_channel, T)
#    fName = outPath + '/' + 'rTES_vs_iTES_ch_' + str(data_channel) + '_' + str(T) + 'mK'
#    make_gen_errplot(iTES*1e6, iTES_rms*1e6, rTES*1e3, rTES_rms*1e3, xLabel, yLabel, titleStr, fName, log='linear')
#    
#    # Make power vs resistance plots
#    pTES = iTES*vTES
#    pTES_rms = pTES*np.sqrt((vTES_rms/vTES)**2 + (iTES_rms/iTES)**2)
#    
#    xLabel = 'TES Resistance [Ohm]'
#    yLabel = 'TES Joule Power [pW]'
#    T = np.round(mean_temperature*1e3,2)
#    titleStr = 'Channel {} TES Joule Power vs Resistance for T = {} mK'.format(data_channel, T)
#    fName = outPath + '/' + 'pTES_vs_rTES_ch_' + str(data_channel) + '_' + str(T) + 'mK'
#    make_gen_plot(rTES, pTES*1e12, xLabel, yLabel, titleStr, fName, logx='linear', logy='linear')
#    #make_gen_errplot(rTES, rTES_rms, pTES*1e12, pTES_rms*1e12, xLabel, yLabel, titleStr, fName, log='linear')
#    
#    # Make Power vs vTES plot (note: vTES = (iBias-iTES)*Rsh)
#    # This is a parabola. Ideally it is pTES = vTES^2/rTES
#    # We fit it to pTES = a*vTES^2 + b*vTES + c
#    # a -> 1/rTES, b -> parasitic current, c -> parasitic power offset
#    cut = np.logical_or(vTES < -0.5e-6, vTES > 0.5e-6)
#    result, pcov = curve_fit(quad_sq, vTES[cut], pTES[cut], sigma=pTES_rms[cut], absolute_sigma=True)
#    perr = np.sqrt(np.diag(pcov))
#    pFit = quad_sq(vTES, result[0], result[1], result[2])
#    
#    xLabel = 'TES Voltage [uV]'
#    yLabel = 'TES Joule Power [pW]'
#    T = np.round(mean_temperature*1e3,2)
#    titleStr = 'Channel {} TES Joule Power vs Voltage for T = {} mK'.format(data_channel, T)
#    fName = outPath + '/' + 'pTES_vs_vTES_ch_' + str(data_channel) + '_' + str(T) + 'mK'
#    # Note: Element 0 is technically rho = 1/R so let us switch it now.
#    # dR/R = d(rho)/rho
#    #perr[0] = perr[0]/result[0]/result[0]
#    #result[0] = 1/result[0]
#    make_power_voltage_fit(vTES*1e6, pTES*1e12, vTES*1e6, pFit*1e12, [result, perr], xLabel, yLabel, titleStr, fName, xErr=vTES_rms*1e6, yErr=pTES_rms*1e12, logx='linear', logy='linear')
#    
#    return True


def make_power_plot(outPath, power_list, temp_list):
    '''Make TES plots for various quantities'''
    # First unpack tes_values
    #iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms = tes_values
    cut = np.logical_and(temp_list >= 0.032, temp_list <= 0.036)
    badT = [34.95e-3, 35.42e-3, 35.91e-3]
    
    # Make iTES vs vTES plot
    xLabel = 'Temperature [mK]'
    yLabel = 'TES Power [fW]'
    titleStr = 'TES Power vs Temperature'
    fName = outPath + '/' + 'pTES_vs_T'
    make_gen_plot(temp_list*1e3, power_list*1e15, xLabel, yLabel, titleStr, fName, logx='linear', logy='linear')
    # Test fit
    testT = np.asarray([10, 20, 30, 32, 33, 35, 40])*1e-3
    testP = power_temp(testT, 0.0381, 1.2e-6, 4.7)
    x0 = [0.03, 1e-6, 5]
    lbounds = [0.01, 1e-9, 3]
    ubounds = [0.50, 1e-4, 6]
    testResult, pcov = curve_fit(power_temp, temp_list[cut], power_list[cut], p0=x0, bounds=(lbounds,ubounds))
    t_tes, k, n = testResult
    print('The test values actually are: Ttes = {} mK, k = {} W/K^{}, n = {}'.format(38.1, 1.2e-6, 4.7, 4.7))
    print('The test values fitted: Ttes = {} mK, k = {} W/K^{}, n = {}'.format(t_tes*1e3, k, n, n))
    
    # Attempt a real fit
    # k will shift it up or down, n is steepness. smaller n looks steeper
    x0 = [0.036, 7.9e-7, 4.3]
    lbounds = [0.01, 1e-9, 3]
    ubounds = [0.1, 1e-4, 6]
    result, pcov = curve_fit(power_temp, temp_list[cut], power_list[cut], p0=x0, bounds=(lbounds,ubounds))
    t_tes, k, n = result
    p_fit = power_temp(temp_list, t_tes, k, n)
    print('The guess values are: Ttes = {} mK, k = {} W/K^{}, n = {}'.format(x0[0]*1e3, x0[1], x0[2], x0[2]))
    print('The values are: Ttes = {} mK, k = {} W/K^{}, n = {}'.format(t_tes*1e3, k, n, n))
    
    # Attempt a linear fit: P = m*T + b
    # Here we assume "T" is T^n and b is "k*Ttes^n" and m is dP/dT
    #x0 = [, 2e-6]
    #lbounds = [0, 1e-8]
    #ubounds = [0.1, 1e-4]
    resultL, pcovL = curve_fit(lin_sq, temp_list[cut], power_list[cut])
    m, b = resultL
    p_fitL = lin_sq(temp_list, m, b)
    print('The values are: dP/dT = {} pW/K, b = {} pW'.format(m*1e12, b*1e12))
    
    xLabel = 'Temperature [mK]'
    yLabel = 'TES Power [fW]'
    titleStr = 'TES Power vs Temperature'
    fName = outPath + '/' + 'pTES_vs_T_fit'
    model_x = [temp_list*1e3, temp_list*1e3]
    model_y = [p_fit*1e15, p_fitL*1e15]
    make_gen_fitplot(temp_list*1e3, power_list*1e15, model_x, model_y, xLabel, yLabel, titleStr, fName, logx='linear', logy='linear', y0=True)
    return True


def get_iTES(vOut, Rfb, M):
    '''Computes the TES current and TES current RMS in Amps'''
    iTES = vOut/Rfb/M
    return iTES


def get_rTES(iTES, vTES):
    '''Computes the TES resistance in Ohms'''
    rTES = vTES/iTES
    return rTES


def get_rTES_rms(iTES, iTES_rms, vTES, vTES_rms):
    '''Comptues the RMS on the TES resistance in Ohms'''
    # Fundamentally this is R = a*iBias/vOut - b
    # a = Rsh*Rfb*M
    # b = Rsh
    # dR/diBias = a/vOut
    # dR/dVout = -a*iBias/vOut^2
    # rTES_rms = sqrt( (dR/diBias * iBiasRms)^2 + (dR/dVout * vOutRms)^2 )
    dR = sqrt(pow2(vTES_rms/iTES) + pow2(-1*vTES*iTES_rms/pow2(iTES)))
    return dR


def get_pTES(iTES, vTES):
    '''Compute the TES power dissipation (Joule)'''
    pTES = iTES*vTES
    return pTES


def get_pTES_rms(iTES, iTES_rms, vTES, vTES_rms):
    '''Computes the RMS on the TES (Joule) power dissipation'''
    dP = sqrt(pow2(iTES*vTES_rms) + pow2(vTES*iTES_rms))
    return dP


def get_vTES(iBias, vOut, Rfb, M, Rsh, Rp):
    '''computes the TES voltage in Volts
    vTES = vSh - vPara
    vTES = Rsh*(iSh) - Rp*iTES
    vTES = Rsh*(iBias - iTES) - Rp*iTES = Rsh*iBias - iTES*(Rp+Rsh)
    '''
    vTES = iBias*Rsh - (vOut/(M*Rfb))*Rsh - (vOut/(M*Rfb))*Rp
    # Simple model
    #vTES = Rsh*(iBias - vOut/M/Rfb)
    return vTES


def get_vTES_rms(iBias_rms, vOut, vOut_rms, Rfb, M, Rsh, Rp, Rp_err):
    '''compute the RMS on the TES voltage in Volts'''
    # Fundamentally this is V = Rsh*iBias - Rp/(MRfb)*vOut - Rsh/(MRfb)*vOut
    # errV**2 = (dV/diBias * erriBias)**2 + (dV/dvOut * errvOut)**2 + (dV/dRp * errRp)**2
    # a = Rsh
    # b = Rsh/Rf/M
    # dV/diBias = a
    # dV/dVout = -b
    # So this does as dV = sqrt((dV/dIbias * iBiasRMS)^2 + (dV/dVout * vOutRMS)^2)
    dV = sqrt(pow2(Rsh*iBias_rms) + pow2(((Rp+Rsh)/(M*Rfb))*vOut_rms) + pow2((vOut/(M*Rfb))*Rp_err))
    return dV


def sort_tes_values(tes_values, sortKey):
    '''For a list of TES circuit values and a given sorting key, sort each value by the key'''
    for index, item in enumerate(tes_values):
        tes_values[index] = item[sortKey]
    return tes_values


def get_TES_fits(tes_values):
    '''Function to get the TES fit parameters for the normal branches and superconducting branch'''
    #TODO: determine if we want to get slopes of V = m*I+ b or I = m*V + b and if we want to plot I vs V or V vs I
    # First umpack. These are sorted by vTES value
    iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms = tes_values
    
    # Get the superconducting branch first. This will tell us what the parasitic series resistance is
    # And will let us knwo the correction shift needed so iTES = 0 when vTES = 0
    # Get SC branch
    (evLeft, evRight) = walk_sc(vTES, iTES)
    # From here let us make a fit of the form y = m*x+b where y = vTES and x = iTES
    # Here m = Resistance and b = voltage offset
    # So when iTES = 0, vTES = b
    # NOTE: This follows Ohm's law but our plotting plane is inverted
    result, pcov = curve_fit(lin_sq, iTES[evLeft:evRight], vTES[evLeft:evRight], sigma=vTES_rms[evLeft:evRight], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = lin_sq(iTES, result[0], result[1])
    vF_sc = {'result': result, 'perr': perr, 'model': vFit*1e6}
    # Get the left side normal branch first
    lev = walk_normal(vTES, iTES, 'left')
    # Model is vTES = m*iTES + b
    result, pcov = curve_fit(lin_sq, iTES[0:lev], vTES[0:lev], sigma=vTES_rms[0:lev], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = lin_sq(iTES, result[0], result[1])
    vF_left = {'result': result, 'perr': perr, 'model': vFit*1e6}

    rev = walk_normal(vTES, iTES, 'right')
    result, pcov = curve_fit(lin_sq, iTES[rev:], vTES[rev:], sigma=vTES_rms[rev:], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = lin_sq(iTES, result[0], result[1])
    vF_right = {'result': result, 'perr': perr, 'model': vFit*1e6}
    
    # Adjust data based on intersection of SC and Normal data
    # V = Rn*I + Bn
    # V = Rs*I + Bs
    # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
    current_intersection = (vF_sc['result'][1] - vF_left['result'][1])/(vF_left['result'][0] - vF_sc['result'][0])
    voltage_intersection = vF_sc['result'][0]*current_intersection + vF_sc['result'][1]
    print('The current and voltage intersections are {} uA and {} uV'.format(current_intersection*1e6, voltage_intersection*1e6))
    iTES = iTES - current_intersection
    vTES = vTES - voltage_intersection
    
    # Redo walks
    (evLeft, evRight) = walk_sc(vTES, iTES)
    # From here let us make a fit of the form y = m*x+b where y = vTES and x = iTES
    # Here m = Resistance and b = voltage offset
    # So when iTES = 0, vTES = b
    # NOTE: This follows Ohm's law but our plotting plane is inverted
    result, pcov = curve_fit(lin_sq, iTES[evLeft:evRight], vTES[evLeft:evRight], sigma=vTES_rms[evLeft:evRight], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = lin_sq(iTES, result[0], result[1])
    vF_sc = {'result': result, 'perr': perr, 'model': vFit*1e6}
    # Get the left side normal branch first
    lev = walk_normal(vTES, iTES, 'left')
    # Model is vTES = m*iTES + b
    result, pcov = curve_fit(lin_sq, iTES[0:lev], vTES[0:lev], sigma=vTES_rms[0:lev], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = lin_sq(iTES, result[0], result[1])
    vF_left = {'result': result, 'perr': perr, 'model': vFit*1e6}

    rev = walk_normal(vTES, iTES, 'right')
    result, pcov = curve_fit(lin_sq, iTES[rev:], vTES[rev:], sigma=vTES_rms[rev:], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = lin_sq(iTES, result[0], result[1])
    vF_right = {'result': result, 'perr': perr, 'model': vFit*1e6}
    # Finally also recompute rTES
    rTES = vTES/iTES
    return vF_left, vF_sc, vF_right, [iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms]


def dump2text(R,T,fileName):
    '''Quick function to dump R and T values to a text file'''
    print('The shape of R and T are: {0} and {1}'.format(R.shape, T.shape))
    np.savetxt(fileName, np.stack((R,T), axis=1), fmt='%12.10f')
    return None


def get_iv_data_from_file(input_path):
    '''Load IV data from specified directory'''
    
    tree = 'data_tree'
    branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform', 'EPCal_K']
    method = 'single'
    rData = readROOT(path, tree, branches, method)
    # Make life easier:
    rData = rData['data']
    return rData


def format_iv_data(iv_data, output_path):
    '''Format the IV data into easy to use forms'''
    
    # Now data structure:
    # Everything is recorded on an event level but some events are globally the same (i.e., same timestamp)
    # In general for example channel could be [5,6,7,5,6,7...]
    # Waveform however is a vector that is the actual sample. Here waveforms at 0,1,2 are for
    # the first global event for channels 5,6,7 respectively.
    # WARNING!!! Because entries are written for each waveform for each EVENT things like time or temperature
    # will contain N(unique channels) duplicates
    # The things we need to return: Time, Temperature, Vin, Vout
    
    # Construct the complete timestamp from the s and mus values
    time_values = iv_data['Timestamp_s'] + iv_data['Timestamp_mus']/1e6
    temperatures = iv_data['EPCal_K']
    cut = temperatures > -1
    # Get unique time values for valid temperatures
    time_values, idx = np.unique(time_values[cut], return_index=True)
    # Reshape temperatures to be only the valid values that correspond to unique times
    temperatures = temperatures[cut]
    temperatures = temperatures[idx]
    test_plot(time_values, temperatures, 'Unix Time', 'Temperature [K]', output_path + '/' + 'quick_look_Tvt.png')
    
    # Next process waveforms into input and output arrays
    waveForms = {ch: {} for ch in np.unique(iv_data['Channel'])}
    nChan = np.unique(iv_data['Channel']).size
    for ev, ch in enumerate(iv_data['Channel'][cut]):
        waveForms[ch][ev//nChan] = iv_data['Waveform'][ev]
    # Ultimately the cut we form from times will tell us what times, and hence events to cut
    # waveForms are dicts of channels with dicts of event numbers that point to the event's waveform
    # Collapse a waveform down to a single value per event means we can form np arrays then
    mean_waveforms = {}
    rms_waveforms = {}
    #print('waveforms keys: {}'.format(list(waveForms[biasChannel].keys())))
    for ch in waveForms.keys():
        mean_waveforms[ch], rms_waveforms[ch] = process_waveform(waveForms[ch], 'mean')
    return time_values, temperatures, mean_waveforms, rms_waveforms


def find_temperature_steps(time_values, temperatures, output_path):
    ''' Given an array of time and temperatures identify temperature steps and return the start, stop and mean Temp
    Will return these as a list of tuples
    '''
    # At this point we should only process data further based on unique IV time blocks so here we must find
    # the list of time tuples (tStart, tEnd, tTemp) that define a particular IV-curve temperature set
    
    # First construct dT/dt and set time values to start at 0
    dT = np.gradient(temperatures, time_values)
    dt = time_values-time_values[0]
    # Construct a diagnostic plot
    cut = np.logical_and(dt > 2000, dt < 4000)
    test_plot(dt[cut], temperatures[cut], 'Time', 'T', output_path + '/' + 'test_Tbyt.png')
    test_plot(dt[cut], dT[cut], 'Time', 'dT/dt', output_path + '/' + 'test_dTbydt.png')
    # Now set some parameters for stablized temperatures
    # Define temperature steps to be larger than Tstep Kelvins
    # Define the rolling windows to contain lenBuff entries
    Tstep = 5e-5
    lenBuff = 10
    timeList = getStabTemp(time_values, temperatures, lenBuff, Tstep)
    test_steps(dt, temperatures, timeList, time_values[0], 'Time', 'T', output_path + '/' + 'test_Tsteps.png')
    return timeList


def get_pyIV_data(input_path, output_path, squid_run, bias_channel):
    '''Function to gather data in correct format for running IV data
    Returns time_values, temperatures, mean_waveforms, rms_waveforms, and timeList
    '''
    
    # First load the data files and return data
    iv_data = get_iv_data_from_file(input_path)
    # Next process data into a more useful format
    time_values, temperatures, mean_waveforms, rms_waveforms = format_iv_data(iv_data, output_path)
    # Next identify temperature steps
    timeList = find_temperature_steps(time_values, temperatures, output_path)
    # Now we have our timeList so we can in theory loop through it and generate IV curves for selected data!
    print('Diagnostics:')
    print('Channels: {}'.format(np.unique(iv_data['Channel'])))
    print('length of time: {}'.format(time_values.size))
    print('length of mean waveforms vector: {}'.format(mean_waveforms[bias_channel].size))
    print('There are {} temperature steps with values of: {}'.format(len(timeList), timeList))
    return time_values, temperatures, mean_waveforms, rms_waveforms, timeList


def fit_sc_branch(x, y, sigmaY):
    '''Walk and fit the superconducting branch
    In the vOut vs iBias plane x = iBias, y = vOut --> dy/dx ~ resistance
    In the iTES vs vTES plane x = vTES, y = iTES --> dy/dx ~ 1/resistance
    '''
    #walk_sc(x, y,
    (evLeft, evRight) = walk_sc(x, y)
    result, pcov = curve_fit(lin_sq, x[evLeft:evRight], y[evLeft:evRight], sigma=sigmaY[evLeft:evRight], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    # These probably are not valid for all cases
    index_y_min = np.argmin(y)
    index_y_max = np.argmax(y)
    return result, perr, (index_y_max, index_y_min)


def fit_normal_branches(x, y, sigmaY):
    '''Walk and fit the normal branches in the vOut vs iBias plane.'''
    # Get the left side normal branch first
    left_ev = walk_normal(x, y, 'left')
    left_result, pcov = curve_fit(lin_sq, x[0:left_ev], y[0:left_ev], sigma=sigmaY[0:left_ev], absolute_sigma=True)
    left_perr = sqrt(np.diag(pcov))
    # Now get the other branch
    right_ev = walk_normal(x, y, 'right')
    right_result, pcov = curve_fit(lin_sq, x[right_ev:], y[right_ev:], sigma=sigmaY[right_ev:], absolute_sigma=True)
    right_perr = np.sqrt(np.diag(pcov))
    return left_result, left_perr, right_result, right_perr


def correct_offsets(fitParams):
    ''' Based on the fit parameters for the normal and superconduting branch correct the offset'''
    # Adjust data based on intersection of SC and Normal data
    # V = Rn*I + Bn
    # V = Rs*I + Bs
    # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
    current_intersection = (fitParams.parasitic.result[1] - fitParams.right.result[1])/(fitParams.right.result[0] - fitParams.parasitic.result[0])
    voltage_intersection = fitParams.parasitic.result[0]*current_intersection + fitParams.parasitic.result[1]
    return current_intersection, voltage_intersection


def fit_iv_regions(x, y, sigmaY, fittype='iv'):
    '''Fit the iv data regions and extract fit parameters'''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    
    R = TESResistance()
    Rerr = TESResistance()
    fitParams = FitParameters()
    # We need to walk and fit the superconducting region first since there RTES = 0
    result, perr, sc_bounds = fit_sc_branch(x, y, sigmaY)
    # Now we can fit the rest
    left_result, left_perr, right_result, right_perr = fit_normal_branches(x, y, sigmaY)

    # The interpretation of the fit parameters depends on what plane we are in
    if fittype == 'iv':
        # We fit something of the form vOut = a*iBias + b
        R.parasitic = Rsh * ((M*Rfb/result[0]) - 1)
        Rerr.parasitic = np.abs( ((-1*M*Rfb*Rsh)/(pow2(result[0])))*perr[0] )
        R.left = (M*Rfb*Rsh/left_result[0]) - Rsh - R.parasitic
        Rerr.left = sqrt(pow2(left_perr[0]*(-1*M*Rfb*Rsh)/pow2(left_result[0])) + pow2(-1*Rerr.parasitic))
        R.right = (M*Rfb*Rsh/right_result[0]) - Rsh - R.parasitic
        Rerr.right = sqrt(pow2(right_perr[0]*(-1*M*Rfb*Rsh)/pow2(right_result[0])) + pow2(-1*Rerr.parasitic))
    elif fittype == 'tes':
        # Here we fit something of the form iTES = a*vTES + b
        # Fundamentally iTES = vTES/rTES ...
        R.parasitic = 1/result[0]
        Rerr.parasitic = np.abs(-1*perr[0]/pow2(result[0]))
        R.left = 1/left_result[0]
        Rerr.left = np.abs(-1*left_perr[0]/pow2(left_result[0]))
        R.right = 1/right_result[0]
        Rerr.right = np.abs(-1*right_perr[0]/pow2(right_result[0]))
    fitParams.parasitic.set_values(result, perr)
    fitParams.left.set_values(left_result, left_perr)
    fitParams.right.set_values(right_result, right_perr)
    #TODO: Make sure everything is right here with the equations and error prop.
    return R, Rerr, fitParams, sc_bounds


def process_iv_curves(outPath, data_channel, iv_dictionary):
    '''Take I-V data and find Rp and Rn values. We need Rp in particular ot get vTES'''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    keys = ['iBias', 'iBias_rms', 'vOut', 'vOut_rms']
    for temperature, iv_data in iv_dictionary.items():
        R, Rerr, fitParams, sc_bounds = fit_iv_regions(x=iv_data['iBias'], y=iv_data['vOut'], sigmaY=iv_data['vOut_rms'], fittype='iv')
        # Next we should center the IV data...it should pass through (0,0)
        # Adjust data based on intersection of SC and Normal data
        # V = Rn*I + Bn
        # V = Rs*I + Bs
        # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
        current_intersection, voltage_intersection = correct_offsets(fitParams)
        iv_data['iBias'] = iv_data['iBias'] - current_intersection
        iv_data['vOut'] = iv_data['vOut'] - voltage_intersection
        # Re-walk on shifted data
        R, Rerr, fitParams, sc_bounds = fit_iv_regions(x=iv_data['iBias'], y=iv_data['vOut'], sigmaY=iv_data['vOut_rms'], fittype='iv')
        iv_data['R'] = R
        iv_data['Rerr'] = Rerr
        # Make I-V plot
        xLabel = 'Bias Current [uA]'
        yLabel = 'Output Voltage [mV]'
        titleStr = 'Channel {} Output Voltage vs Bias Current for T = {} mK'.format(data_channel, temperature)
        fName = outPath + '/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
        iv_fitplot(iv_data['iBias'], iv_data['vOut'], iv_data['iBias_rms'], iv_data['vOut_rms'], 
                   [R, Rerr, fitParams], xLabel, yLabel, titleStr, fName, sc=sc_bounds, xScale=1e6, yScale=1e3, logx='linear', logy='linear')
    return iv_dictionary


def process_tes_curves(outPath, data_channel, iv_dictionary):
    '''Take TES data and find Rp and Rn values.'''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    for temperature, iv_data in iv_dictionary.items():
        R, Rerr, fitParams, sc_bounds = fit_iv_regions(x=iv_data['vTES'], y=iv_data['iTES'], sigmaY=iv_data['iTES_rms'], fittype='tes')
        # Next we should center the IV data...it should pass through (0,0)
        # Adjust data based on intersection of SC and Normal data
        # V = Rn*I + Bn
        # V = Rs*I + Bs
        # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
        #current_intersection, voltage_intersection = correct_offsets(fitParams)
        #iv_data['iBias'] = iv_data['iBias'] - current_intersection
        #iv_data['vOut'] = iv_data['vOut'] - voltage_intersection
        # Re-walk on shifted data
        #R, Rerr, fitParams, sc_bounds = fit_iv_regions(iv_data, keys)
        iv_data['rTES'] = R
        iv_data['rTES_err'] = Rerr
        # Make TES Plots
        make_tes_plots(output_path=outPath, data_channel=data_channel, temperature=temperature, sc_bounds=sc_bounds, data=iv_data, fit_parameters=fitParams)
        xLabel = 'TES Voltage [uV]'
        yLabel = 'TES Current [uA]'
        titleStr = 'Channel {} TES Current vs TES Voltage for T = {} mK'.format(data_channel, temperature)
        fName = outPath + '/' + 'iTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
        #iv_fitplot(iv_data['vTES'], iv_data['iTES'], iv_data['vTES_rms'], iv_data['iTES_rms'], [R, Rerr, fitParams], xLabel, yLabel, titleStr, fName, sc=sc_bounds, xScale=1e6, yScale=1e6, logx='linear', logy='linear')
    return iv_dictionary


def get_TES_values(outPath, data_channel, iv_dictionary):
    '''From I-V data values compute the TES values for iTES and vTES, ultimately yielding rTES'''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    tes_dictionary = {}
    for temperature, iv_data in iv_dictionary.items():
        tes_dictionary[temperature] = {}
        Rfb = squid_parameters['Rfb']
        M = squid_parameters['M']
        Rsh = squid_parameters['Rsh']
        
        iTES = get_iTES(iv_data['vOut'], Rfb, M)
        iTES_rms = get_iTES(iv_data['vOut_rms'], Rfb, M)
        vTES = get_vTES(iv_data['iBias'], iv_data['vOut'], Rfb, M, Rsh, iv_data['R'].parasitic)
        vTES_rms = get_vTES_rms(iv_data['iBias_rms'], iv_data['vOut'], iv_data['vOut_rms'], Rfb, M, Rsh, iv_data['R'].parasitic, iv_data['Rerr'].parasitic)
        # Problem: Data is sorted by increasing iBias right now. Let's sort it by increasing vTES
        sortKey = np.argsort(vTES)
        iv_data['iTES'] = iTES[sortKey]
        iv_data['iTES_rms'] = iTES_rms[sortKey]
        iv_data['vTES'] = vTES[sortKey]
        iv_data['vTES_rms'] = vTES_rms[sortKey]
    return iv_dictionary


def get_iv_data(input_path, output_path, squid_run, bias_channel, data_channel):
    '''Function that returns an iv dictionary from waveform root file'''
    time_values, temperatures, mean_waveforms, rms_waveforms, timeList = get_pyIV_data(input_path, output_path, squid_run, bias_channel)
    # Next chop up waveform data into an iv dictionary
    iv_dictionary = generate_iv_curves(output_path, time_values, temperatures, mean_waveforms, rms_waveforms, timeList, bias_channel, data_channel)
    return iv_dictionary


def generate_iv_curves(outPath, time_values, temperatures, mean_waveforms, rms_waveforms, timeList, bias_channel, data_channel):
    '''Function that will process the multiple waveform vs temperature windows into separate I-V curve regions'''
    squid_parameters = get_squid_parameters(2)
    # Unfold useful squid parameters
    Rbias = squid_parameters['Rbias']
    T = [32.57, 33.04, 34.46, 34.94, 35.43, 35.91]
    power_list = np.empty(0)
    temp_list = np.empty(0)
    iv_dictionary = {}
    for values in timeList:
        start_time, stop_time, mean_temperature = values
        cut = np.logical_and(time_values >= start_time, time_values <= stop_time)
        times = time_values[cut]
        iBias = mean_waveforms[bias_channel][cut]/Rbias
        iBias_rms = rms_waveforms[bias_channel][cut]/Rbias
        # sort in order of increasing bias current
        sortKey = np.argsort(iBias)
        vOut = mean_waveforms[data_channel][cut]
        vOut_rms = rms_waveforms[data_channel][cut]
        # We can technically get iTES at this point too since it is proportional to vOut but since it is let's not.
        T = str(np.round(mean_temperature*1e3, 3))
        iv_dictionary[T] = {'iBias': iBias[sortKey],
                            'iBias_rms': iBias_rms[sortKey],
                            'vOut': vOut[sortKey],
                            'vOut_rms': vOut_rms[sortKey]
                           }
    return iv_dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', default='/Users/bwelliver/cuore/bolord/squid/', help='Specify full file path of input data')
    parser.add_argument('-o', '--outputPath', help='Path to put the plot directory. Defaults to input directory')
    parser.add_argument('-r', '--run', type=int, help='Specify the SQUID run number to use')
    parser.add_argument('-b', '--biasChannel', type=int, default=5, help='Specify the digitizer channel that corresponds to the bias channel. Defaults to 5')
    parser.add_argument('-d', '--dataChannel', type=int, default=7, help='Specify the digitizer channel that corresponds to the output channel. Defaults to 7')
    parser.add_argument('-s', '--makeRoot', action='store_true', help='Specify whether to write data to a root file')
    parser.add_argument('-p', '--readROOT', action='store_true', help='Read iv data from processed root file. Stored in outputPath /root/iv_data.root')
    args = parser.parse_args()

    path = args.inputFile
    run = args.run
    outPath = args.outputPath if args.outputPath else dirname(path) + '/' + basename(path).replace('.root', '')
    if not isabs(outPath):
        outPath = dirname(path) + '/' + outPath    
    mkdpaths(outPath)
    print('We will run with the following options:')
    print('The squid run is {}'.format(run))
    print('The output path is: {}'.format(outPath))
    
    if args.readROOT is False:
        iv_dictionary = get_iv_data(input_path=args.inputFile, output_path=outPath, squid_run=args.run, bias_channel=args.biasChannel, data_channel=args.dataChannel)
        # Next save the iv_curves
        save_to_root(outPath, iv_dictionary)
    if args.readROOT is True:
        # If we saved the root file and want to load it do so here
        iv_dictionary = read_from_ivroot(outPath + '/root/iv_data.root')
    # Next we can process the IV curves to get Rn and Rp values. Once we have Rp we can obtain vTES and go onward
    iv_dictionary = process_iv_curves(outPath, args.dataChannel, iv_dictionary)
    get_TES_values(outPath, args.dataChannel, iv_dictionary)
    iv_dictionary = process_tes_curves(outPath, args.dataChannel, iv_dictionary)
    print('done')