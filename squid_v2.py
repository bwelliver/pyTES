import os
from os.path import isabs
from os.path import dirname
from os.path import basename

import glob
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

from scipy.stats import mode
from scipy.stats import ks_2samp
import scipy.signal as signal
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


def findStartTemp(vTemp, mThresh, pBuff, cBuff, fBuff, ev):
    '''Function to find the start of a stable period
    Basically here as we slide the three windows along, if the current window is more similar to the past
    then it implies the future has become different enough to mean we are entering a new region
    
    '''
    
    boundaryFlag = False
    while boundaryFlag == False and ev < vTemp.size - 1:
        # First compare whether past and future are similar enough
        dM = np.abs( (pBuff.get_mean() - fBuff.get_mean())/pBuff.get_mean() )
        if dM > mThresh:
            # past and future are the same so slide everybody forward...recall buffers fill left to right
            # so oldest entry is the 'last' entry
            # Here we make the oldest entry of the current buffer the newest entry of the past buffer
            # and the oldest entry of the future buffer becomes the newest current value
            oldF = fBuff.get_all()[-1]
            oldC = cBuff.get_all()[-1]
            pBuff.append(oldC)
            cBuff.append(oldF)
            fBuff.append(vTemp[ev])
            ev += 1
        else:
            # past and future are not the same so test if the present time is similar enough to either window
            dMcp = np.abs( (cBuff.get_mean() - pBuff.get_mean())/cBuff.get_mean() )
            dMcf = np.abs( (cBuff.get_mean() - fBuff.get_mean())/cBuff.get_mean() )
            # Construct the 1 and only 1 case where we return something
            if dMcf > mThresh and dMcf > dMcp:
                # Present and future are similar enough and the present is more similar to the future than the past.
                # Here the window has slid enough so we can decide that we have a time boundary now at the current event
                # So let us return everything as it is.
                boundaryFlag = True
            else:
                # Here we are not in our return case. This could be a result of the following situations:
                # - pcf > pThresh and pcf < ppc - present and future similar but present more similar to past
                # - pcf < pThresh and pcf > ppc - present and future not similar enough but moreso than past and present
                # - pcf < pThresh and pcf < ppc - present and future not similar enough and present more similar to past
                # Note in the 3rd case we could have subcases depending on if ppc > pThresh or ppc < pThresh
                # This does not change anything though because we have not met the stopping condition
                oldF = fBuff.get_all()[-1]
                oldC = cBuff.get_all()[-1]
                pBuff.append(oldC)
                cBuff.append(oldF)
                fBuff.append(vTemp[ev])
                ev += 1
    return pBuff, cBuff, fBuff, ev


def findEndTemp(vTemp, pThresh, pBuff, cBuff, fBuff, ev):
    '''Function to find the start of a stable period
    Basically here as we slide the three windows along, if the current window is more similar to the past
    then it implies the future has become different enough to mean we are entering a new region
    
    '''
    
    boundaryFlag = False
    while boundaryFlag == False and ev < vTemp.size - 1:
        # First compare whether past and future are similar enough
        D,p = ks_2samp(pBuff.get_all(), fBuff.get_all())
        if p > pThresh:
            # past and future are the same so slide everybody forward...recall buffers fill left to right
            # so oldest entry is the 'last' entry
            # Here we make the oldest entry of the current buffer the newest entry of the past buffer
            # and the oldest entry of the future buffer becomes the newest current value
            oldF = fBuff.get_all()[-1]
            oldC = cBuff.get_all()[-1]
            pBuff.append(oldC)
            cBuff.append(oldF)
            fBuff.append(vTemp[ev])
            ev += 1
        else:
            # past and future are not the same so test if the present time is similar enough to either window
            Dpc, ppc = ks_2samp(pBuff.get_all(), cBuff.get_all())
            Dcf, pcf = ks_2samp(cBuff.get_all(), fBuff.get_all())
            # Construct the 1 and only 1 case where we return something
            if ppc > pThresh and ppc > pcf:
                # Present and past are similar enough and the present is more similar to the past than the future.
                # Here the window has slid enough so we can decide that we have a time boundary now at the current event
                # So let us return everything as it is.
                boundaryFlag = True
            else:
                # Here we are not in our return case. This could be a result of the following situations:
                # - ppc > pThresh and ppc < pcf - present and past similar but present more similar to future
                # - ppc < pThresh and ppc > pcf - present and past not similar enough but moreso than future and present
                # - ppc < pThresh and ppc < pcf - present and past not similar enough and present more similar to future
                # Note in the 3rd case we could have subcases depending on if pcf > pThresh or pcf < pThresh
                # This does not change anything though because we have not met the stopping condition
                oldF = fBuff.get_all()[-1]
                oldC = cBuff.get_all()[-1]
                pBuff.append(oldC)
                cBuff.append(oldF)
                fBuff.append(vTemp[ev])
                ev += 1
    return pBuff, cBuff, fBuff, ev


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



def findStartTempKS(vTemp, pThresh, pBuff, cBuff, fBuff, ev):
    '''Function to find the start of a stable period
    Basically here as we slide the three windows along, if the current window is more similar to the past
    then it implies the future has become different enough to mean we are entering a new region
    
    '''
    
    boundaryFlag = False
    while boundaryFlag == False and ev < vTemp.size - 1:
        # First compare whether past and future are similar enough
        D,p = ks_2samp(pBuff.get_all(), fBuff.get_all())
        if p > pThresh:
            # past and future are the same so slide everybody forward...recall buffers fill left to right
            # so oldest entry is the 'last' entry
            # Here we make the oldest entry of the current buffer the newest entry of the past buffer
            # and the oldest entry of the future buffer becomes the newest current value
            oldF = fBuff.get_all()[-1]
            oldC = cBuff.get_all()[-1]
            pBuff.append(oldC)
            cBuff.append(oldF)
            fBuff.append(vTemp[ev])
            ev += 1
        else:
            # past and future are not the same so test if the present time is similar enough to either window
            Dpc, ppc = ks_2samp(pBuff.get_all(), cBuff.get_all())
            Dcf, pcf = ks_2samp(cBuff.get_all(), fBuff.get_all())
            # Construct the 1 and only 1 case where we return something
            if pcf > pThresh and pcf > ppc:
                # Present and future are similar enough and the present is more similar to the future than the past.
                # Here the window has slid enough so we can decide that we have a time boundary now at the current event
                # So let us return everything as it is.
                boundaryFlag = True
            else:
                # Here we are not in our return case. This could be a result of the following situations:
                # - pcf > pThresh and pcf < ppc - present and future similar but present more similar to past
                # - pcf < pThresh and pcf > ppc - present and future not similar enough but moreso than past and present
                # - pcf < pThresh and pcf < ppc - present and future not similar enough and present more similar to past
                # Note in the 3rd case we could have subcases depending on if ppc > pThresh or ppc < pThresh
                # This does not change anything though because we have not met the stopping condition
                oldF = fBuff.get_all()[-1]
                oldC = cBuff.get_all()[-1]
                pBuff.append(oldC)
                cBuff.append(oldF)
                fBuff.append(vTemp[ev])
                ev += 1
    return pBuff, cBuff, fBuff, ev


def findEndTempKS(vTemp, pThresh, pBuff, cBuff, fBuff, ev):
    '''Function to find the start of a stable period
    Basically here as we slide the three windows along, if the current window is more similar to the past
    then it implies the future has become different enough to mean we are entering a new region
    
    '''
    
    boundaryFlag = False
    while boundaryFlag == False and ev < vTemp.size - 1:
        # First compare whether past and future are similar enough
        D,p = ks_2samp(pBuff.get_all(), fBuff.get_all())
        if p > pThresh:
            # past and future are the same so slide everybody forward...recall buffers fill left to right
            # so oldest entry is the 'last' entry
            # Here we make the oldest entry of the current buffer the newest entry of the past buffer
            # and the oldest entry of the future buffer becomes the newest current value
            oldF = fBuff.get_all()[-1]
            oldC = cBuff.get_all()[-1]
            pBuff.append(oldC)
            cBuff.append(oldF)
            fBuff.append(vTemp[ev])
            ev += 1
        else:
            # past and future are not the same so test if the present time is similar enough to either window
            Dpc, ppc = ks_2samp(pBuff.get_all(), cBuff.get_all())
            Dcf, pcf = ks_2samp(cBuff.get_all(), fBuff.get_all())
            # Construct the 1 and only 1 case where we return something
            if ppc > pThresh and ppc > pcf:
                # Present and past are similar enough and the present is more similar to the past than the future.
                # Here the window has slid enough so we can decide that we have a time boundary now at the current event
                # So let us return everything as it is.
                boundaryFlag = True
            else:
                # Here we are not in our return case. This could be a result of the following situations:
                # - ppc > pThresh and ppc < pcf - present and past similar but present more similar to future
                # - ppc < pThresh and ppc > pcf - present and past not similar enough but moreso than future and present
                # - ppc < pThresh and ppc < pcf - present and past not similar enough and present more similar to future
                # Note in the 3rd case we could have subcases depending on if pcf > pThresh or pcf < pThresh
                # This does not change anything though because we have not met the stopping condition
                oldF = fBuff.get_all()[-1]
                oldC = cBuff.get_all()[-1]
                pBuff.append(oldC)
                cBuff.append(oldF)
                fBuff.append(vTemp[ev])
                ev += 1
    return pBuff, cBuff, fBuff, ev



def getStabTempKS(vTime, vTemp):
    '''Function that attemps to get periods of sufficiently stable temperature to use in an IV curve
    This method relies on using 3 sliding windows and a KS test to compare whether the target window
    is more similar to the past (left) or future (right) window and whether the past and future are similar
    '''
    lenBuff = 20
    pBuff = RingBuffer(lenBuff, dtype=float)
    cBuff = RingBuffer(lenBuff, dtype=float)
    fBuff = RingBuffer(lenBuff, dtype=float)
    # Fill the buffers initially
    for idx in range(lenBuff):
        pBuff.append(vTemp[idx])
    ev = idx + 1
    for idx in range(lenBuff):
        cBuff.append(vTemp[ev + idx])
    ev += idx + 1
    for idx in range(lenBuff):
        fBuff.append(vTemp[ev + idx])
    ev += idx + 1
    # Now let us initialize things for the step scan
    dM = 1
    pThresh = 1e-3
    tList = []
    while ev < vTemp.size - 1:
        pBuff, cBuff, fBuff, ev = findStartTempKS(vTemp, pThresh, pBuff, cBuff, fBuff, ev)
        ev = ev + 1
        if ev >= vTemp.size:
            break
        else:
            tStart = vTime[ev - 1]
            oldF = fBuff.get_all()[-1]
            oldC = cBuff.get_all()[-1]
            pBuff.append(oldC)
            cBuff.append(oldF)
            fBuff.append(vTemp[ev])
        pBuff, cBuff, fBuff, ev = findEndTempKS(vTemp, pThresh, pBuff, cBuff, fBuff, ev)
        ev += 1
        if ev >= vTemp.size:
            break
        else:
            tEnd = vTime[ev - 1]
            oldF = fBuff.get_all()[-1]
            oldC = cBuff.get_all()[-1]
            pBuff.append(oldC)
            cBuff.append(oldF)
            fBuff.append(vTemp[ev])
            cTime = np.logical_and(vTime >= tStart, vTime <= tEnd)
            mTemp = np.mean(vTemp[cTime])
            tList.append((tStart, tEnd, mTemp))
        print(ev)
    return tList


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


def walk_normal(v, j, side):
    '''Function to walk the normal branches and find the line fit
    To do this we will start at the min or max input current and compute a walking derivative
    If the derivative starts to change then this indicates we entered the biased region and should stop
    '''
    
    # First let us compute the gradient (dy/dx) of the TES current (y) with respect to the TES voltage (x)
    df = np.gradient(j, v)
    if side == 'right':
        df = df[::-1]
    # In the normal region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time. If the new average differs from the previous
    # by some amount mark that as the end.
    # Create ring buffer of size 20 to store bias and signal
    lenBuff = 40
    dbuff = RingBuffer(lenBuff, dtype=float)
    for ev in range(lenBuff):
        dbuff.append(df[ev])
    # Now our buffer is initialized so loop over all events until we find a change
    ev = lenBuff
    dM = 0
    while dM < 5e-1 and ev < df.size - 1:
        currentM = dbuff.get_mean()
        dbuff.append(df[ev])
        newM = dbuff.get_mean()
        dM = np.abs((currentM - newM)/currentM)
        ev += 1
    if side == 'right':
        # Flip event index back the right way
        ev = df.size - 1 - ev
    print('The {} deviation occurs at ev = {} with current = {} and voltage = {} with dM = {}'.format(side, ev, j[ev], v[ev], dM))
    return ev


def walk_sc(v, j):
    '''Function to walk the superconducting region of the IV curve
    Generally when ib = 0 we should be superconducting so we will start there and go up until the bias
    then return to 0 and go down until the bias
    '''
    # First let us compute the gradient of the signal with respect to the bias
    df = np.gradient(j, v)
    # In the sc region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time. If the new average differs from the previous
    # by some amount mark that as the end.
    # First we should find whereabouts the region of ibias = 0 is
    mindex = np.argmin(np.abs(v))
    # First go from mindex and increase bias current
    # Create ring buffer of to store signal
    lenBuff = 4
    #TODO: FIX THIS TO HANDLE SQUID JUMPS
    dbuff = RingBuffer(lenBuff, dtype=float)
    for ev in range(lenBuff):
        dbuff.append(df[mindex + ev])
    # Now our buffer is initialized so loop over all events until we find a change
    ev = mindex + lenBuff
    dM = 0
    while dM < 1e-2 and ev < df.size - 1:
        currentM = dbuff.get_mean()
        dbuff.append(df[ev])
        newM = dbuff.get_mean()
        dM = np.abs((currentM - newM)/currentM)
        ev += 1
    print('The deviation occurs at ev = {} with current = {} and voltage = {} with dM = {}'.format(ev, j[ev], v[ev], dM))
    evR = ev
    # Now repeat but go to the left
    lenBuff = 4
    dbuff = RingBuffer(lenBuff, dtype=float)
    for ev in range(lenBuff):
        dbuff.append(df[mindex - ev])
    # Now our buffer is initialized so loop over all events until we find a change
    ev = mindex - lenBuff
    dM = 0
    while dM < 1e-2 and ev < df.size - 1:
        currentM = dbuff.get_mean()
        dbuff.append(df[ev])
        newM = dbuff.get_mean()
        dM = np.abs((currentM - newM)/currentM)
        ev -= 1
    print('The deviation occurs at ev = {} with current = {} and voltage = {} with dM = {}'.format(ev, j[ev], v[ev], dM))
    evL = ev
    return (evL, evR)


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
    fig.savefig(fName, format='png', dpi=100)
    plt.close('all')
    
    
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

def gen_fitplot(x, y, xerr, yerr, lFit, rFit, scFit, xlab, ylab, title, fName, log='linear'):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', xerr=xerr, yerr=yerr)
    if lFit != None:
        ax.plot(x, lFit['model'], 'r-', marker='None', linewidth=2)
    if rFit != None:
        ax.plot(x, rFit['model'], 'g-', marker='None', linewidth=2)
    # do not plot all of the sc model
    if scFit != None:
        cut = np.logical_and(scFit['model'] <= np.max(y), scFit['model'] >= np.min(y))
        ax.plot(x[cut], scFit['model'][cut], 'b-', marker='None', linewidth=2)
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



def gen_fitplot_diagnostic(x, y, xerr, yerr, lFit, xlab, ylab, title, fName, log='linear'):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None', xerr=xerr, yerr=yerr)
    ax.plot(x, lFit['model'], 'r-', marker='None', linewidth=2)
    ax.set_xscale(log)
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    ax.set_title(title, fontsize=18)
    #ax.set_ylim((0.95*y.min(), 1.05*y.max()))
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    begin = '$'
    lR = '$\mathrm{R_{nL}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(lFit['result'][0]*1e3, lFit['perr'][0]*1e3)
    lOff = '$\mathrm{i_{offL}} = %.5f \pm %.5f \mathrm{\mu A}$'%(lFit['result'][1]*1e6, lFit['perr'][1]*1e6)
    ending = '$'
    textStr = lR + '\n' + lOff
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


def lin_sq(x,m,b):
    '''Get the output for a linear response to an input'''
    y = m*x + b
    return y


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


def nll_lin(params, x, y):
    '''A fit function for a linear relation'''
    m,b = params
    model = lin_sq(x,m,b)
    lnl = nsum((y - model)**2)
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
    if sigma1 <= 0 or sigma2 <= 0 or a1 < 0 or a2 < 0 or mu1 > data.max() or mu1 < data.min() or mu2 > data.max() or mu2 < data.min():
        lnl = np.inf
    else:
        lnl = -nsum( ln(gaus2(data, a1, mu1, sigma1, a2, mu2, sigma2) + eps ) )
    return lnl


def lin_sq(x,m,b):
    '''Get the output for a linear response to an input'''
    y = m*x + b
    return y


def make_diagnostic_plots(pTime, iBias, iBiasRMS, vOut, vOutRMS, sortKey, t0, mT, ch, outPath ):
    '''Make diagnostic plots for various quantities'''
    xLabel = 'Time From Start [s]'
    yLabel = 'Bias Current [uA]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} Bias Current vs Time for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'iBias_vs_time_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    make_gen_errplot(pTime[sortKey] - t0, None, iBias[sortKey]*1e6, iBiasRMS[sortKey]*1e6, xLabel, yLabel, titleStr, fName, log='linear')
    # Next is vOut
    xLabel = 'Time From Start [s]'
    yLabel = 'Output Voltage [mV]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} Output Voltage vs Time for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'vOut_vs_time_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    make_gen_errplot(pTime[sortKey] - t0, None, vOut[sortKey]*1e3, vOutRMS[sortKey]*1e3, xLabel, yLabel, titleStr, fName, log='linear')
    # Next is vOut
    xLabel = 'Bias Current [uA]'
    yLabel = 'Output Voltage [mV]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} Output Voltage vs Bias Current for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'vOut_vs_iBias_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    make_gen_errplot(iBias[sortKey]*1e6, iBiasRMS[sortKey]*1e6, vOut[sortKey]*1e3, vOutRMS[sortKey]*1e3, xLabel, yLabel, titleStr, fName, log='linear')
    return True


def make_TES_plots(iBias, iBiasRMS, iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms, iF_left, iF_sc, iF_right, mT, ch, outPath ):
    '''Make TES plots for various quantities'''
    xLabel = 'TES Voltage [uV]'
    yLabel = 'TES Current [uA]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} TES Current vs TES Voltage for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'iTES_vs_vTES_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    gen_fitplot(vTES*1e6, iTES*1e6, vTES_rms*1e6, iTES_rms*1e6, iF_left, iF_right, iF_sc, xLabel, yLabel, titleStr, fName, log='linear')
    xLabel = 'TES Voltage [uV]'
    yLabel = 'TES Resistance [mOhm]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} TES Resistance vs TES Voltage for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'rTES_vs_vTES_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    make_gen_errplot(vTES*1e6, vTES_rms*1e6, rTES*1e3, rTES_rms*1e3, xLabel, yLabel, titleStr, fName, log='linear')
    
    xLabel = 'TES Current [uA]'
    yLabel = 'TES Resistance [mOhm]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} TES Resistance vs TES Current for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'rTES_vs_iTES_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    make_gen_errplot(iTES*1e6, iTES_rms*1e6, rTES*1e3, rTES_rms*1e3, xLabel, yLabel, titleStr, fName, log='linear')
    
    xLabel = 'Shunt Current (iBias - iTES) [uA]'
    yLabel = 'Shunt Voltage [uV]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} Shunt Voltage vs Shunt Current for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'vSh_vs_iSh_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    #make_gen_errplot((iBias-iTES)*1e6, np.sqrt((iBiasRMS*iBiasRMS + iTES_rms*iTES_rms))*1e6, vTES*1e6, vTES_rms*1e6, xLabel, yLabel, titleStr, fName, log='linear')
    
    return True


def make_TES_plots_diagnostic(iTES, iTES_rms, vTES, vTES_rms, iF_left, outPath ):
    '''Make TES plots for various quantities'''
    ch = -1
    mT = -1
    xLabel = 'TES Voltage [uV]'
    yLabel = 'TES Current [uA]'
    T = np.round(mT*1e3,2)
    titleStr = 'Channel {} TES Current vs TES Voltage for T = {} mK'.format(ch, T)
    fName = outPath + '/' + 'iTES_vs_vTES_ch_' + str(int(ch)) + '_' + str(T) + 'mK'
    gen_fitplot_diagnostic(vTES*1e6, iTES*1e6, vTES_rms*1e6, iTES_rms*1e6, iF_left, xLabel, yLabel, titleStr, fName, log='linear')
    
    return True


def get_iTES(v, Rfb, M):
    '''Computes the TES current and TES current RMS in Amps'''
    iTES = v/Rfb/M
    return iTES


def get_rTES(iBias, vOut, Rfb, M, Rsh):
    '''Computes the TES resistance in Ohms'''
    rTES = Rsh*(Rfb*M*iBias/vOut - 1.0)
    return rTES


def get_rTES_rms(iBias, iBiasRMS, vOut, vOutRMS, Rfb, M, Rsh):
    '''comptues the RMS on the TES resistance in Ohms'''
    # Fundamentally this is R = a*iBias/vOut - b
    # a = Rsh*Rfb*M
    # b = Rsh
    # dR/diBias = a/vOut
    # dR/dVout = -a*iBias/vOut^2
    # rTES_rms = sqrt( (dR/diBias * iBiasRms)^2 + (dR/dVout * vOutRms)^2 )
    dR = np.sqrt(np.power(Rsh*Rfb*M*iBiasRMS/vOut, 2) + np.power(-Rsh*Rfb*M*iBias*vOutRMS/np.power(vOut, 2), 2))
    return dR


def get_vTES(iBias, vOut, Rfb, M, Rsh):
    '''computes the TES voltage in Volts'''
    vTES = Rsh*(iBias - vOut/Rfb/M)
    return vTES


def get_vTES_rms(iBias, iBiasRMS, vOut, vOutRMS, Rfb, M, Rsh, iTES_rms):
    '''compute the RMS on the TES voltage in Volts'''
    # Fundamentally this is V = a*iBias - b*vOut
    # a = Rsh
    # b = Rsh/Rf/M
    # dV/diBias = a
    # dV/dVout = -b
    # So this does as dV = sqrt((dV/dIbias * iBiasRMS)^2 + (dV/dVout * vOutRMS)^2)
    dV = np.sqrt(np.power(Rsh * iBiasRMS, 2) + np.power(-Rsh/Rfb/M * vOutRMS, 2))
    return dV


def get_TES_values(iBias, iBiasRMS, vOut, vOutRMS, sortKey, sqParams):
    '''Function to get TES current, voltage and resistance values'''
    Rfb = sqParams['Rfb']
    M = sqParams['Mratio']
    Rsh = sqParams['Rsh']
    iTES = get_iTES(vOut[sortKey], Rfb, M)
    iTES_rms = get_iTES(vOutRMS[sortKey], Rfb, M)
    rTES = get_rTES(iBias[sortKey], vOut[sortKey], Rfb, M, Rsh)
    rTES_rms = get_rTES_rms(iBias[sortKey], iBiasRMS[sortKey], vOut[sortKey], vOutRMS[sortKey], Rfb, M, Rsh)
    vTES = get_vTES(iBias[sortKey], vOut[sortKey], Rfb, M, Rsh)
    vTES_rms = get_vTES_rms(iBias[sortKey], iBiasRMS[sortKey], vOut[sortKey], vOutRMS[sortKey], Rfb, M, Rsh, iTES_rms)
    return iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms


def get_TES_fits(iTES, iTES_rms, vTES, vTES_rms):
    '''Function to get the TES fit parameters for the normal branches and superconducting branch'''
    #TODO: determine if we want to get slopes of V = m*I+ b or I = m*V + b and if we want to plot I vs V or V vs I
    lev = walk_normal(vTES, iTES, 'left')
    #lev = 500
    # Model is vTES = m*iTES + b
    # We are plotting iTES vs vTES though so invert the linear fit parameters to plot but report the model results
    # We need to get iTES = m'*vTES + b'
    # In general then
    # iTES = (1/m)*vTES - b/m
    # so m' = +1/m and dm' = -dm/m^2
    # so b' = -b/m and db' = -db/m
    result, pcov = curve_fit(lin_sq, iTES[0:lev], vTES[0:lev], sigma=vTES_rms[0:lev], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    iFit = lin_sq(vTES, 1/result[0], -result[1]/result[0])
    iF_left = {'result': result, 'perr': perr, 'model': iFit*1e6}
    #make_TES_plots_diagnostic(iTES[0:lev], iTES_rms[0:lev], vTES[0:lev], vTES_rms[0:lev], iF_left, outPath)

    rev = walk_normal(vTES, iTES, 'right')
    result, pcov = curve_fit(lin_sq, iTES[rev:], vTES[rev:], sigma=vTES_rms[rev:], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    iFit = lin_sq(vTES, 1/result[0], -result[1]/result[0])
    iF_right = {'result': result, 'perr': perr, 'model': iFit*1e6}

    # Get SC branch
    (evL, evR) = walk_sc(vTES, iTES)
    result, pcov = curve_fit(lin_sq, iTES[evL:evR], vTES[evL:evR], sigma=vTES_rms[evL:evR], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    iFit = lin_sq(vTES, 1/result[0], -result[1]/result[0])
    iF_sc = {'result': result, 'perr': perr, 'model': iFit*1e6}
    return iF_left, iF_sc, iF_right


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
    print('Bins are {0}'.format(bins))
    ax.hist(data, bins=bins, normed=True)
    ax.plot(xModel, yModel, 'r-', marker='None', linewidth=2)
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
    parser.add_argument('-i', '--inputFile', default='/Users/bwelliver/cuore/bolord/squid/', help='Specify full file path of input data')
    parser.add_argument('-o', '--outputPath', help='Specify the output path to put plots. If not specified will create a  directory in the input path based off the input filename. If not absolute path, will be relative to the input file path')
    parser.add_argument('-r', '--run', help='Specify the SQUID run number to use')
    parser.add_argument('-d', '--dumpFile', action='store_true', help='Specify whether a dump file to store R and T in text should be made. Default location will be set to outputPath')
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
    # Now load the data
    tree = 'data_tree'
    branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform', 'EPCal_K']
    method = 'single'
    rData = readROOT(path, tree, branches, method)
    # Specify input bias resistance to convert Vbias to Ibias
    Rbias = 10000
    # Make life easier:
    rData = rData['data']
    # Now data structure:
    # Everything is recorded on an event level but some events are globally the same (i.e., same timestamp)
    # In general for example channel could be [5,6,7,5,6,7...]
    # Waveform however is a vector that is the actual sample. Here waveforms at 0,1,2 are for
    # the first global event for channels 5,6,7 respectively.
    # WARNING!!! Because entries are written for each waveform for each EVENT things like time or temperature
    # will contain N(unique channels) duplicates
    chanVec = rData['Channel']
    nTunix = rData['Timestamp_s']
    nTunix_mus = rData['Timestamp_mus']
    nTime = nTunix + nTunix_mus/1e6
    tempVec = rData['EPCal_K']
    cut = tempVec > -1
    nTime, idx = np.unique(nTime[cut], return_index=True)
    tempVec = tempVec[cut]
    tempVec = tempVec[idx]
    test_plot(nTime, tempVec, 'Unix Time', 'Temperature [K]', 'quick_look_Tvt.png')
    # Check here...something is going awry with the channels perhaps (probably when we cut out events,
    # it does not do so symmetrically so we have an offset)
    print('total number of events is {}'.format(chanVec.size))
    
    waveForms = {ch: {} for ch in np.unique(chanVec)}
    nChan = np.unique(chanVec).size
    for ev, ch in enumerate(chanVec[cut]):
        waveForms[ch][ev//nChan] = rData['Waveform'][ev]
    biasChannel = 5
    # Ultimately the cut we form from times will tell us what times, and hence events to cut
    # waveForms are dicts of channels with dicts of event numbers that point to the event's waveform
    # Collapse a waveform down to a single value per event means we can form np arrays then
    mWave = {}
    mWaveRMS = {}
    #print('waveforms keys: {}'.format(list(waveForms[biasChannel].keys())))
    for ch in waveForms.keys():
        mWave[ch], mWaveRMS[ch] = process_waveform(waveForms[ch], 'mean')
    # At this point we should only process data further based on unique IV time blocks so here we must find
    # the list of time tuples (tStart, tEnd, tTemp) that define a particular IV-curve temperature set
    dT = np.gradient(tempVec, nTime)
    dt = nTime-nTime[0]
    cut = np.logical_and(dt > 2000, dt < 4000)
    test_plot(dt[cut], tempVec[cut], 'Time', 'T', 'test_Tbyt.png')
    test_plot(dt[cut], dT[cut], 'Time', 'dT/dt', 'test_dTbydt.png')
    Tstep = 5e-5
    lenBuff = 10
    timeList = getStabTemp(nTime, tempVec, lenBuff, Tstep)
    test_steps(dt, tempVec, timeList, nTime[0], 'Time', 'T', 'test_Tsteps.png')
    # Now we have our timeList so we can in theory loop through it and generate IV curves for selected data!
    print('Diagnostics:')
    print('Channels: {}'.format(np.unique(chanVec)))
    print('length of time: {}'.format(nTime.size))
    print('length of waveforms vector: {}'.format(len(waveForms[biasChannel].keys())))
    print('The temperature steps are: {}'.format(timeList))
    #print('waveforms keys: {}'.format(list(waveForms[biasChannel].keys())))
    c2s = {5: 2, 6: 3, 7: 2}
    sqParams = {2: {'Rfb': 10e3, 'Mratio': -1.2786, 'Rsh': 21e-3}, 3: {'Rfb': 10e3, 'Mratio': -1.4256, 'Rsh': 22.8e-3}}
    t0 = nTime[0]
    for values in timeList:
        tStart, tEnd, mT = values
        cut = np.logical_and(nTime >= tStart, nTime <= tEnd)
        pTime = nTime[cut]
        iBias = mWave[biasChannel][cut]/Rbias
        iBiasRMS = mWaveRMS[biasChannel][cut]/Rbias
        # sort in order of increasing bias current
        sortKey = np.argsort(iBias)
        # We will need to eventually loop over this by channel but for now hardcode
        ch = 6
        vOut = mWave[ch][cut]
        vOutRMS = mWaveRMS[ch][cut]
        # Make some diagnostic plots
        if pTime.size > 700:
            make_diagnostic_plots(pTime, iBias, iBiasRMS, vOut, vOutRMS, sortKey, t0, mT, ch, outPath)
            # Next up compute TES quantities (all in SI units of course)
            iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms = get_TES_values(iBias, iBiasRMS, vOut, vOutRMS, sortKey, sqParams[c2s[ch]])
            # We need to get fit parameters too to get Rn values
            iF_left, iF_sc, iF_right = get_TES_fits(iTES, iTES_rms, vTES, vTES_rms)
            make_TES_plots(iBias, iBiasRMS, iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms, iF_left, iF_sc, iF_right, mT, ch, outPath)