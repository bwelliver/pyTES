import os
import argparse
from os.path import isabs
from os.path import dirname
from os.path import basename

import numpy as np
from numpy import exp as exp
from numpy import log as ln
from numpy import square as pow2
from numpy import power as power
from numpy import sqrt as sqrt
from numpy import sum as nsum
from numpy import tanh as tanh

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
from scipy.signal import detrend

import matplotlib as mp
from matplotlib import pyplot as plt

from RingBuffer import RingBuffer

import ROOT as rt

from readROOT import readROOT
from writeROOT import writeROOT

eps = np.finfo(float).eps
ln = np.log

#mp.use('agg')

class ArrayIsUnsortedException(Exception):
    pass

class InvalidChannelNumberException(Exception):
    pass

class InvalidObjectTypeException(Exception):
    pass

class RequiredValueNotSetException(Exception):
    pass


class FitParameters:
    
    def __init__(self):
        self.left = FitResult()
        self.right = FitResult()
        self.sc = FitResult()
    def __repr__(self):
        """return string representation."""
        s = 'Left Branch:\n' + str(self.left)
        s = s + '\n' + 'Right Branch:\n' + str(self.right)
        s = s + '\n' + 'Superconducting:\n' + str(self.sc)
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
    
    def __init__(self):
        self.left = Resistance()
        self.right = Resistance()
        self.parasitic = Resistance()
        self.sc = Resistance()
    def __repr__(self):
        '''Return string representation'''
        s = 'Left Branch:\n' + str(self.left)
        s += '\n' + 'Right Branch:\n' + str(self.right)
        s += '\n' + 'Parasitic:\n' + str(self.parasitic)
        s += '\n' + 'Superconducting:\n' + str(self.sc)
        return(s)

class Resistance:
    
    def __init__(self):
        self.value = None
        self.rms = None
    def set_values(self, value=None, rms=None):
        self.value = value
        self.rms = rms
    def __repr__(self):
        '''Return string representation'''
        s = '\t' + 'Value:\t' + str(self.value)
        s += '\n' + '\t' + 'RMS:\t' + str(self.rms)
        return(s)

    
class SQUIDParameters:
    '''Simple object class to store SQUID parameters'''
    
    def __init__(self, channel):
        self.squid = self.get_squid_parameters(channel)
        return None
    
    def get_squid_parameters(self, channel):
        '''Based on the channel number obtain SQUID parameters'''
        if channel == 2:
            self.serial = 'S0121'
            self.Li = 6e-9
            self.Mi = 1/26.062
            self.Mfb = 1/33.488
            self.Rfb = 10000.0
            self.Rsh = 21.0e-3
            self.Rb = 10000.0
            self.Cb = 100e-12
        elif channel == 3:
            self.serial = 'S0094'
            self.Li = 6e-9
            self.Mi = 1/23.99
            self.Mfb = 1/32.9
            self.Rfb = 10000.0
            self.Rsh = 22.8e-3
            self.Rb = 10000.0
            self.Cb = 100e-12
        else:
            raise InvalidChannelNumberException('Requested channel: {} is invalid. Please select 2 or 3'.format(channel))
        # Compute auxillary SQUID parameters based on ratios
        self.M = -self.Mi/self.Mfb
        self.Lfb = (self.M**2)*self.Li
        return None


class IV:
    '''A Class to contain IV data and some methods in a compact form. 
    All current, voltage, resistance and power is stored as an IVData class (value, RMS properties)
    Inputs:
        channel: The SQUID channel to use for proper transfer relationships
    Properties:
        bias current
        output voltage
        TES current
        TES voltage
        TES resistance
        TES (Joule) power dissipation
    Methods:
        This class contains methods to compute the TES values 
    
    '''

    def __init__(self, channel=2):
        self.iBias = IVData()
        self.vOut = IVData()
        self.iTES = IVData()
        self.vTES = IVData()
        self.rTES = IVData()
        self.pTES = IVData()
        self.squid = SQUIDParameters(channel)
        self.Rp = IVData()
        return None
    
    
    @staticmethod
    def compute_iTES(v, Rfb, M):
        '''Computes the TES current and TES current RMS in Amps'''
        i = v/(Rfb*M)
        return i
    
    
    @staticmethod
    def compute_vTES(iBias, vOut, Rfb, Rsh, M, Rp=0):
        '''Computes the TES voltage in V'''
        vTES = iBias*Rsh - (vOut/(M*Rfb))*Rsh - (vOut/(M*Rfb))*Rp
        return vTES
    
    
    @staticmethod
    def compute_vTES_rms(iBias_rms, vOut, vOut_rms, Rfb, Rsh, M, Rp=0, Rp_rms=0):
        '''Computes the TES voltage RMS in V'''
        # The error is composed of 3 terms: The iBias term, the vOut term, and the parasitic term
        dV_iBias = Rsh*iBias_rms
        dV_vOut = ((Rp + Rsh)/(M*Rfb))*vOut_rms
        dV_Rp = (vOut/(M*Rfb))*Rp_rms
        dV = sqrt(pow2(dV_iBias) + pow2(dV_vOut) + pow2(dV_Rp))
        return dV
    
    
    @staticmethod
    def compute_rTES(vTES, iTES):
        '''Compute the TES resistance in Ohms'''
        rTES = vTES/iTES
        return rTES
    
    
    @staticmethod
    def compute_rTES_rms(vTES, vTES_rms, iTES, iTES_rms):
        '''Compute the RMS of the TES resistance in Ohms'''
        # This is composed of the RMS of vTES in one part and the RMS of iTES in the other
        dR_v = vTES_rms/iTES
        dR_i = -1*(vTES/iTES)*(iTES_rms/iTES)
        dR = sqrt(pow2(dR_v) + pow2(dR_i))
        return dR
    
    
    @staticmethod
    def compute_pTES(vTES, iTES):
        '''Compute the TES power'''
        pTES = vTES*iTES
        return pTES
    
    
    @staticmethod
    def compute_pTES_rms(vTES, vTES_rms, iTES, iTES_rms):
        '''Compute the RMS of the TES power'''
        # This is composed of the portion due to the vTES and the iTES RMS
        dP_v = vTES_rms*iTES
        dP_i = vTES*iTES_rms
        dP = sqrt(pow2(dP_v) + pow2(dP_i))
        return dP
    
    
    def set_parasitic_resistance(self, R):
        '''Takes as input an IVData object for the parasitic resistance value'''
        if isinstance(R, IVData):
            self.Rp = R
        else:
            raise InvalidObjectTypeException("Input object R is of type {}. Must be of type IVData".format(type(R)))
        return None
    
    
    def get_iTES(self):
        '''Assign iTES data values and RMS values'''
        iTES = self.compute_iTES(self.vOut.data, self.squid.Rfb, self.squid.M)
        iTES_rms = self.compute_iTES(self.vOut.rms, self.Rfb, self.M)
        self.iTES.set_values(iTES, iTES_rms)
        return None
    
    
    def get_vTES(self):
        '''Assign vTES data values and vTES RMS values'''
        vTES = self.compute_vTES(self.iBias.data, self.vOut.data, self.squid.Rfb, self.squid.Rsh, self.squid.M, self.Rp.data)
        vTES_rms = self.compute_vTES_rms(self.iBias.rms, self.vOut.data, self.vOut.rms, self.squid.Rfb, self.squid.Rsh, self.squid.M, self.Rp.data, self.Rp.rms)
        self.vTES.set_values(vTES, vTES_rms)
        return None
    
    
    def get_rTES(self):
        '''Assign rTES values and rTES RMS values'''
        rTES = self.compute_rTES(self.vTES.data, self.iTES.data)
        rTES_rms = self.compute_rTES_rms(self.vTES.data, self.vTES.rms, self.iTES.data, self.iTES.rms)
        self.rTES.set_values(rTES, rTES_rms)
        return None
    
    
    def get_pTES(self):
        '''Assign pTES values and pTES RMS values'''
        pTES = self.compute_pTES(self.vTES.data, self.iTES.data)
        pTES_rms = self.compute_pTES_rms(self.vTES.data, self.vTES.rms, self.iTES.data, self.iTES.rms)
        self.pTES.set_data(pTES, pTES_rms)
        return None

    
    def get_all_TES(self):
        '''Obtains and assigns all TES values'''
        if self.Rp.data is None:
            raise RequiredValueNotSetException('Required series parasitic values are not set')
        if self.iBias.data is None:
            raise RequiredValueNotSetException('Required bias current values are not set')
        if self.vOut.data is None:
            raise RequiredValueNotSetException('Required output voltage values are not set')
        self.get_iTES()
        self.get_vTES()
        self.get_rTES()
        self.get_pTES()
        return None


class IVData:
    '''Class for IV Data objects. A simple container for the quantity and its RMS'''
    
    def __init__(self):
        self.data = None
        self.rms = None
        return None
    def set_values(self, data=None, rms=None):
        self.data = data
        self.rms = rms
        return None
    def __repr__(self):
        '''Return a string representation'''
        s = '\t' + 'Data:\t' + str(self.data) + '\n' + '\t' + 'RMS:\t' + str(self.rms)
        return(s)



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


def exp_tc(T, C, D, B, A):
    '''Alternative R vs T
    Here we have 
    C -> Rn
    D -> Rp
    -B/A -> Tc
    In the old fit we hade (T-Tc)/Tw -> T/Tw - Tc/Tw
    We have here A*T + B --> A = 1/Tw and B = -Tc/Tw
    '''
    R = (C*exp(A*T + B)) / (1 + exp(A*T + B)) + D
    return R


def tanh_tc(T, Rn, Rp, Tc, Tw):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance
    Rp is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data
    Usually the following is true:
        When T >> Tc: R = Rn + Rp
        When T << Tc: R = Rp
        When T = Tc: R = Rn/2 + Rp
    But Rp is already subtracted from our data so it should be 0ish
    
    '''
    #R = (Rn/2.0)*(tanh((T - Tc)/Tw) + 1) + Rp
    R = ((Rn - Rp)/2)*(tanh((T - Tc)/Tw) + 1) + Rp
    return R

def tanh_tc2(T, Rn, Tc, Tw):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance
    Rp is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data
    Usually the following is true:
        When T >> Tc: R = Rn + Rp
        When T << Tc: R = Rp
        When T = Tc: R = Rn/2 + Rp
    But Rp is already subtracted from our data so it should be 0ish
    
    '''
    
    R = (Rn/2.0)*(tanh((T - Tc)/Tw) + 1)
    return R



def nll_error(params, P, P_rms, T, func):
    '''A fit for whatever function with y-errors'''
    k, n, Ttes = params
    if k <= 0 or n <= 0 or Ttes <= 0:
        return np.inf
    else:
        model = func(T, k, n, Ttes)
        lnl = nsum(((P - model)/P_rms)**2)
        return lnl


def tes_power_polynomial(T, k, n, Ttes):
    '''General TES power equation
    P = k*(T^n - Tb^n)
    '''
    P = k*np.power(Ttes,n) - k*np.power(T,n)
    #P = k*(power(Ttes, n) - power(T, n))
    return P


def tes_power_polynomial5(T, k, Ttes):
    '''General TES power equation assuming n = 5
    P = k*(T^n - Tb^n)
    '''
    P = k*np.power(Ttes, 5) - k*np.power(T, 5)
    #P = k*(power(Ttes, n) - power(T, n))
    return P



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


def process_waveform(waveform, time_values, sample_length, number_of_windows=1, process_type='mean'):
    '''Function to process waveform into a statistically downsampled representative point
    waveform:
        A dictionary whose keys represent the event number. Values are numpy arrays with a length = NumberOfSamples
        The timestamp of waveform[event][sample] is time_values[event] + sample*sample_length
    time_values:
        An array containing the start time (with microsecond resolution) of an event
    sample_length:
        The length in seconds of a sample within an event. waveform[event].size*sample_length = event duration
    number_of_windows:
        How many windows should an event be split into for statistical sampling
    process_type:
        The particular type of processing to use.
        mean:
            For the given division of the event performs a mean and std over samples to obtain statistically
            representative values
        median:
            TODO
    This will return 2 dictionaries keyed by event number and an appropriately re-sampled time_values array
    '''
    # Allocate numpy arrays. Note that if we made some assumptions we could pre-allocate instead of append...
    mean_waveform = np.empty(0)
    rms_waveform = np.empty(0)
    new_time_values = np.empty(0)
    if process_type == 'mean':
        for event, samples in waveform.items():
            number_of_samples = samples.size
            base_index = samples.size // number_of_windows
            event_time = time_values[event]
            for n in range(number_of_windows):
                lower_index = n*base_index
                upper_index = (n+1)*base_index
                mean_waveform = np.append(mean_waveform, np.mean(samples[lower_index:upper_index]))
                rms_waveform = np.append(rms_waveform, np.std(samples[lower_index:upper_index]))
                new_time_values = np.append(new_time_values, event_time + ((upper_index + lower_index)/2)*sample_length)
        return mean_waveform, rms_waveform, new_time_values
    

def correct_squid_jumps(outPath, temperature, t, x, xrms, y, yrms, buffer_size=5):
    '''Function that will correct a SQUID jump in a given waveform'''
    # This is a little dicey. Sudden changes in output voltage can be the result of a SQUID jump
    # Or it could be simply a transition between a SC and N state.
    # Typically SQUID jumps will be larger, but not always.
    # One way to identify them is to first identify the "jump" whatever it is and then look at the slope on either side
    # If the slope on either side of a jump is approximately the same, it is a SQUID jump since the SC to N transition
    # results in a large change in slope.
    
    # The walk normal function should handle this fairly well with some modifications.
    # Note that once a jump is identified we should apply a correction to subsequent data
    
    # We will examine the data in the IV plane, after we have processed waveforms.
    # There will of course be a 'point' where the jump happens when we obtain the mean and RMS of the window
    # This point can be discarded.
    
    # Let us walk along the waveform to see if there are any squid jumps
    # Ensure we have the proper sorting of the data
    if np.all(t[:-1] <= t[1:]) == False:
        raise ArrayIsUnsortedException('Input argument t is unsorted')
    # First let us compute the gradients with respect to time
    dydt = np.gradient(y, t, edge_order=2)
    dxdt = np.gradient(x, t, edge_order=2)
    # Next construct dydx in time ordered sense
    dydx = dydt/dxdt
    # So we will walk along and compute the average of N elements at a time. 
    # If the new average differs from the previous by some amount mark that as the boundary of a SQUID jump
    # This should not be a subtle thing.
    # Make a plot of what we are testing
    test_plot(t, y, 'time', 'vOut', outPath + 'uncorrected_squid_jumps_' + str(temperature) + 'mK.png')
    dbuff = RingBuffer(buffer_size, dtype=float)
    for ev in range(buffer_size):
        dbuff.append(dydt[ev])
    # Now our buffer is initialized so loop over all events until we find a change
    ev = buffer_size
    dMean = 0
    print('The first y value is: {} and the location of the max y value is: {}'.format(y[0], np.argmax(y)))
    while dMean < 3 and ev < dydt.size - 1:
        currentMean = dbuff.get_mean()
        dbuff.append(dydt[ev])
        newMean = dbuff.get_mean()
        dMean = np.abs((currentMean - newMean)/currentMean)
        print('The current y value is: {}'.format(y[ev]))
        print('ev {}: currentMean = {}, newMean = {}, dMean = {}'.format(ev, currentMean, newMean, dMean))
        ev += 1
    # We have located a potential jump at this point (ev)
    # So compute the slope on either side of ev and compare
    print('The event and size of the data are {} and {}'.format(ev, t.size))
    if ev == t.size - 1:
        print('No jumps found after walking all of the data...')
    else:
        distance_ahead = 10 if ev + 10 < dydt.size - 1 else dydt.size - ev - 1
        distance_behind = 10 if ev - 10 > 0 else ev
        #slope_before = np.mean(dydt[ev-distance_behind:ev])
        #slope_after = np.mean(dydt[ev+1:ev+distance_ahead])
        result, pcov = curve_fit(lin_sq, t[ev-distance_behind:ev], y[ev-distance_behind:ev])
        fit_before = result[0]
        result, pcov = curve_fit(lin_sq, t[ev+1:ev+distance_ahead], y[ev+1:ev+distance_ahead])
        fit_after = result[0]
        print('The event and size are {} and {} and The slope before is: {}, slope after is: {} and slope ratio: {}'.format(ev, dydt.size, fit_before, fit_after, np.abs((fit_before - fit_after)/fit_after)))
        if np.abs((fit_before - fit_after)/fit_after) < 0.5:
            # This means the slopes are the same and so we have a squid jump
            # Easiest thing to do then is to determine what the actual value was prior to the jump
            # and then shift the offset values up to it
            result, pcov = curve_fit(lin_sq, t[ev-distance_behind:ev], y[ev-distance_behind:ev])
            extrapolated_value_of_post_jump_point = lin_sq(t[ev+5], *result)
            print('The extrapolated value for the post jump point should be: {}. It is actually right now {}'.format(extrapolated_value_of_post_jump_point, y[ev-10:ev+10]))
            # Now when corrected y' - evopjp = 0 so we need y' = y - dy where dy = y - evopjp
            dy = y[ev+5] - extrapolated_value_of_post_jump_point
            y[ev+5:] = y[ev+5:] - dy
            print('After correction the value of y is {}'.format(y[ev+5]))
            # Finally we must remove the point ev from the data. Let's be safe and remove the surrounding points
            points_to_remove = [ev + (i-4) for i in range(9)]
            x = np.delete(x, points_to_remove)
            y = np.delete(y, points_to_remove)
            xrms = np.delete(xrms, points_to_remove)
            yrms = np.delete(yrms, points_to_remove)
            t = np.delete(t, points_to_remove)
            test_plot(t, y, 'time', 'vOut', outPath + 'corrected_squid_jumps_' + str(temperature) + 'mK.png')
            # probably what we should do here is then call this function again until we find no jumps
            t, x, xrms, y, yrms = correct_squid_jumps(outPath, temperature, t, x, xrms, y, yrms)
        else:
            # Slopes are not the same...this is not a squid jump.
            print('No SQUID Jump detected')
    return t, x, xrms, y, yrms
    

#@obsolete
#def process_waveform(dWaveform, procType='mean'):
#    '''Take an input waveform dictionary and process it by collapse to 1 point
#    An incoming waveform is a dictionary with keys = event number and values = waveform of duration 1s
#    This function will collapse the waveform to 1 value based on procType and return it as a numpy array
#    We will also return a bonus array of the rms for each point too
#    '''
#    # We can have events missing so better to make the vectors equal to the max key
#    npWaveform = np.empty(len(dWaveform.keys()))
#    npWaveformRMS = np.empty(len(dWaveform.keys()))
#    if procType == 'mean':
#        for ev, waveform in dWaveform.items():
#            npWaveform[ev] = np.mean(waveform)
#            npWaveformRMS[ev] = np.std(waveform)
#    elif procType == 'median':
#        for ev, waveform in dWaveform.items():
#            npWaveform[ev] = np.median(waveform)
#            q75, q25 = np.percentile(waveform, [75, 25])
#            npWaveformRMS[ev] = q75 - q25
#    else:
#        raise Exception('Please enter mean or median for process')
#    return npWaveform, npWaveformRMS


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
    dt = 0
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
        # Check validity of this temperature step: It must last longer than some amount of time
        if tEnd - tStart > dt:
            cut = np.logical_and(vTime >= tStart + dt, vTime <= tEnd)
            mTemp = np.mean(vTemp[cut])
            tri = (tStart, tEnd, mTemp)
            tList.append(tri)
    return tList


def walk_normal(x, y, side, buffer_size=50):
    '''Function to walk the normal branches and find the line fit
    To do this we will start at the min or max input current and compute a walking derivative
    If the derivative starts to change then this indicates we entered the biased region and should stop
    NOTE: We assume data is sorted by voltage values
    '''
    # Ensure we have the proper sorting of the data
    if np.all(x[:-1] <= x[1:]) == False:
        raise ArrayIsUnsortedException('Input argument x is unsorted')
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


def walk_sc(x, y, buffer_size=4, plane='iv'):
    '''Function to walk the superconducting region of the IV curve and get the left and right edges
    Generally when ib = 0 we should be superconducting so we will start there and go up until the bias
    then return to 0 and go down until the bias
    In order to be correct your x and y data values must be sorted by x
    '''
    # Ensure we have the proper sorting of the data
    if np.all(x[:-1] <= x[1:]) == False:
        raise ArrayIsUnsortedException('Input argument x is unsorted')
        
    # First let us compute the gradient (i.e. dy/dx)
    dydx = np.gradient(y, x, edge_order=2)
    
    # In the sc region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time. 
    # If the new average differs from the previous by some amount mark that as the end.
    
    # First we should find whereabouts of (0,0)
    # This should roughly correspond to x = 0 since if we input nothing we should get out nothing. In reality there are parasitics of course
    if plane == 'tes':
        index_min_x = np.argmin(np.abs(x))
        # Occasionally we may have a shifted curve that is not near 0 for some reason (SQUID jump)
        # So find the min and max iTES and then find the central point
    elif plane == 'iv':
        # Try a new approach based on the behavior of dy/dx --> SC region will be fundamentally a lower dy/dx than normal
        #index_max_y = np.argmin(dydx[x < 0])
        #index_min_y = np.argmin(dydx[x > 0])
        #index_max_y = np.argmax(y)
        #index_min_y = np.argmin(y)
        #mean_x = (x[index_max_y] + x[index_min_y])/2
        # Find the index of x nearest this mean value
        #index_min_x = np.argmin(np.abs(x-mean_x))
        # wait...if we plot iBias as x, then by def iBias = 0 is 0 since we control it...
        index_min_x = np.argmin(np.abs(x))
        # NOTE: The above will fail for small SC regions where vOut normal > vOut sc!!!!
    # First go from index_min_x and increase
    # Create ring buffer of to store signal
    #buffer_size = 4
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
    while dMean < 1e-2 and ev < dydx.size - 1:
        currentMean = dbuff.get_mean()
        dbuff.append(dydx[ev])
        newMean = dbuff.get_mean()
        dMean = np.abs((currentMean - newMean)/currentMean)
        ev -= 1
    #print('The left deviation occurs at ev = {} with current = {} and voltage = {} with dMean = {}'.format(ev, current[ev], voltage[ev], dMean))
    evLeft = ev
    return (evLeft, evRight)


def test_plot(x, y, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=2, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
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


def test_steps(x, y, v, t0, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', markersize=1, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
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
    ax.yaxis.label.set_size(18)
    ax.xaxis.label.set_size(18)
    ax.set_title(labels['title'], fontsize=18)
    ax.grid(True)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    return ax


def add_model_fits(ax, x, y, model, xScale=1, yScale=1, model_function=lin_sq):
    '''Add model fits to plots'''
    xModel = np.linspace(x.min(), x.max(), 100)
    if model.left.result is not None:
        yFit = model_function(xModel, *model.left.result)
        ax.plot(xModel*xScale, yFit*yScale, 'r-', marker='None', linewidth=2)
    if model.right.result is not None:
        yFit = model_function(xModel, *model.right.result)
        ax.plot(xModel*xScale, yFit*yScale, 'g-', marker='None', linewidth=2)
    if model.sc.result is not None:
        yFit = model_function(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        ax.plot(x[cut]*xScale, yFit[cut]*yScale, 'b-', marker='None', linewidth=2)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    return ax


def add_fit_textbox(ax, R, model):
    '''Add decoration textbox to a plot'''
    
    lR = r'$\mathrm{Left R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.left.value*1e3, R.left.rms*1e3)
    lOff = r'$\mathrm{Left V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)
    
    sR = r'$\mathrm{SC R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.parasitic.value*1e3, R.parasitic.rms*1e3)
    sOff = r'$\mathrm{SC V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.sc.result[1]*1e3, model.sc.error[1]*1e3)
    
    rR = r'$\mathrm{Right R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.right.value*1e3, R.right.rms*1e3)
    rOff = r'$\mathrm{Right V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)
    
    textStr = lR + '\n' + lOff + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    out = ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    return ax


def add_power_voltage_textbox(ax, model):
    '''Add dectoration textbox for a power vs resistance fit'''
    lR = r'$\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(1/model.left.result[0]*1e3, model.left.error[0]/pow2(model.left.result[0])*1e3)
    lI = r'$\mathrm{I_{para}} = %.5f \pm %.5f \mathrm{uA}$'%(model.left.result[1]*1e6, model.left.error[1]*1e6)
    lP = r'$\mathrm{P_{para}} = %.5f \pm %.5f \mathrm{fW}$'%(model.left.result[2]*1e15, model.left.error[2]*1e15)
    
    textStr = lR + '\n' + lI + '\n' + lP
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return ax


def add_resistance_temperature_textbox(ax, model):
    '''Add dectoration textbox for a power vs resistance fit'''
    
    # First is the ascending (SC to N) parameters
    textStr = ''
    if model.left.result is not None:
        lR = r'SC to N: $\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.left.result[0]*1e3, model.left.error[0]*1e3)
        lRp = r'SC to N: $\mathrm{R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)
        lTc = r'SC to N: $\mathrm{T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.left.result[2]*1e3, model.left.error[2]*1e3)
        lTw = r'SC to N: $\mathrm{\Delta T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.left.result[3]*1e3, model.left.error[3]*1e3)
        textStr += lR + '\n' + lRp + '\n' + lTc + '\n' + lTw
    # Next the descending (N to SC) parameters
    if model.right.result is not None:
        rR = r'N to SC: $\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.right.result[0]*1e3, model.right.error[0]*1e3)
        rRp = r'N to SC: $\mathrm{R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)
        rTc = r'N to SC: $\mathrm{T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.right.result[2]*1e3, model.right.error[2]*1e3)
        rTw = r'N to SC: $\mathrm{\Delta T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.right.result[3]*1e3, model.right.error[3]*1e3)
        if textStr is not '':
            textStr += '\n' 
        textStr += rR + '\n' + rRp + '\n' + rTc + '\n' + rTw
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.10, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return ax



def add_power_temperature_textbox(ax, model):
    '''Add decoration textbox for a power vs temperature fit'''
    k = model.left.result[0]
    dk = model.left.error[0]
    n = model.left.result[1]
    dn = model.left.error[1]
    Ttes = model.left.result[2]
    dTtes = model.left.error[2]
    lk = r'$k = %.5f \pm %.5f \mathrm{nW/K^{%.5f}}$'%(k*1e9, dk*1e9, n)
    ln = r'$n = %.5f \pm %.5f$'%(n, dn)
    lTt = r'$T_{TES} = %.5f \pm %.5f \mathrm{mK}$'%(Ttes*1e3, dTtes*1e3)
    # Compute G at T = Ttes
    # G = dP/dT
    G = n*k*power(Ttes, n-1)
    dG_k = n*power(Ttes, n-1)*dk
    dG_T = n*(n-1)*k*power(1e-4, n-2) # RMS on T not Ttes
    dG_n = dn*(k*power(Ttes, n-1)*(n*np.log(Ttes) + 1))
    dG = sqrt(pow2(dG_k) + pow2(dG_T) + pow2(dG_n))
    lG = r'$G(T_{TES}) = %.5f \pm %.5f \mathrm{pW/K}$'%(G*1e12, dG*1e12)
    textStr = lk + '\n' + ln + '\n' + lTt + '\n' + lG
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
    

def iv_fitplot(x, y, xerr, yerr, model, Rp, xLabel, yLabel, title, fName, xScale=1, yScale=1, logx='linear', logy='linear'):
    '''Wrapper for plotting an iv curve with fit parameters'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    yFit1 = lin_sq(x, *model.right.result)
    ax.errorbar(x*xScale, y*yScale, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0, linestyle='None', xerr=xerr*xScale, yerr=yerr*yScale)
    if model.left.result is not None:
        yFit = lin_sq(x, *model.left.result)
        ax.plot(x*xScale, yFit*yScale, 'r-', marker='None', linewidth=2)
    if model.right.result is not None:
        yFit = lin_sq(x, *model.right.result)
        ax.plot(x*xScale, yFit*yScale, 'g-', marker='None', linewidth=2)
    if model.sc.result is not None:
        # Need to plot only a subset of data
        yFit = lin_sq(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        ax.plot(x[cut]*xScale, yFit[cut]*yScale, 'b-', marker='None', linewidth=2)
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
    R = fit_to_resistance(model, fit_type='iv', Rp=Rp.value, Rp_rms=Rp.rms)
    lR = r'$\mathrm{Left \ R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.left.value*1e3, R.left.rms*1e3)
    lOff = r'$\mathrm{Left \ V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)
    
    sR = r'$\mathrm{R_{sc} - R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.sc.value*1e3, R.sc.rms*1e3)
    sOff = r'$\mathrm{V_{sc,off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.sc.result[1]*1e3, model.sc.error[1]*1e3)
    
    rR = r'$\mathrm{Right \ R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.right.value*1e3, R.right.rms*1e3)
    rOff = r'$\mathrm{Right \ V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)
    
    pR = r'$\mathrm{R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(Rp.value*1e3, Rp.rms*1e3)
    
    textStr = lR + '\n' + lOff + '\n' + pR + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #ax.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    ax.text(0.65, 0.9, textStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return None


def make_root_plot(outPath, data_channel, temperature, iv_data, model, Rp, xScale=1, yScale=1):
    '''A helper function to generate a TMultiGraph for the IV curve
    Recipe for this type of plot:
    Create a TCanvas object and adjust its parameters
    Create a TMultiGraph
    Create a TGraph - adjust its parameters and plant it in the TMultiGraph (which now owns the TGraph)
    Finish styling for the TMultiGraph and then save the canvas as .png and .C
    
    There will be 4 TGraphs
        vOut vs iBias
        Left Normal fit
        Right Normal fit
        SC fit
    '''
    
    # Create TCanvas
    w = 1600
    h = 1200 
    c = rt.TCanvas("iv", "iv", w, h)
    c.SetWindowSize(w + (w - c.GetWw()), h + (h - c.GetWh()))
    c.cd()
    c.SetGrid()
    mg = rt.TMultiGraph()
    
    # Now let us generate some TGraphs!
    x = iv_data['iBias']
    xrms = iv_data['iBias_rms']
    y = iv_data['vOut']
    yrms = iv_data['vOut_rms']
    # First up: vOut vs iBias
    g0 = rt.TGraphErrors(x.size, x*xScale, y*yScale, xrms*xScale, yrms*yScale)
    g0.SetMarkerSize(0.5)
    g0.SetLineWidth(1)
    g0.SetName("vOut_iBias")
    g0.SetTitle("OutputVoltage vs Bias Current")
    
    mg.Add(g0)
    
    # Next up let's add the fit lines
    if model.left.result is not None:
        yFit = lin_sq(x, *model.left.result)
        gLeft = rt.TGraph(x.size, x*xScale, yFit*yScale)
        gLeft.SetMarkerSize(0)
        gLeft.SetLineWidth(2)
        gLeft.SetLineColor(rt.kRed)
        gLeft.SetTitle("Left Normal Branch Fit")
        gLeft.SetName("left_fit")
        mg.Add(gLeft)
    if model.right.result is not None:
        yFit = lin_sq(x, *model.right.result)
        gRight = rt.TGraph(x.size, x*xScale, yFit*yScale)
        gRight.SetMarkerSize(0)
        gRight.SetLineWidth(2)
        gRight.SetLineColor(rt.kGreen)
        gRight.SetTitle("Right Normal Branch Fit")
        gRight.SetName("right_fit")
        mg.Add(gRight)
    if model.sc.result is not None:
        yFit = lin_sq(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min()) 
        gSC = rt.TGraph(x[cut].size, x[cut]*xScale, yFit[cut]*yScale)
        gSC.SetMarkerSize(0)
        gSC.SetLineWidth(2)
        gSC.SetLineColor(rt.kBlue)
        gSC.SetTitle("Superconducting Branch Fit")
        gSC.SetName("sc_fit")
        mg.Add(gSC)
    # Now I guess we format the multigraph
    xLabel = 'Bias Current [uA]'
    yLabel = 'Output Voltage [mV]'
    titleStr = 'Channel {} Output Voltage vs Bias Current for T = {} mK'.format(data_channel, temperature)
    fName = outPath + '/root/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
    mg.Draw("APL")
    mg.GetXaxis().SetTitle(xLabel)
    mg.GetYaxis().SetTitle(yLabel)
    mg.SetTitle(titleStr)
    # Construct Legend
    leg = rt.TLegend(0.6, 0.7, 0.9, 0.9)
    leg.AddEntry(g0, "IV Data", "le")
    leg.AddEntry(gLeft, "Left Normal Branch Fit", "l")
    leg.AddEntry(gRight, "Right Normal Branch Fit", "l")
    leg.AddEntry(gSC, "Superconducting Branch Fit", "l")
    leg.SetTextSize(0.02)
    #leg.SetTextFont(2)
#    TLegend *leg = new TLegend(0.55, 0.7, 0.9, 0.9);
#    //TLegendEntry *le = leg->AddEntry(h2, "All PhaseIDs: noise distribution across all channels", "fl");
#    leg->AddEntry(h2, "All PhaseIDs: noise distribution across all channels", "fl");
#    //TLegendEntry *le_optimal = leg->AddEntry(hc_optimal, Form("PhaseID %d: noise distribution across all channels", optimal_phaseid), "fl");
#    leg->AddEntry(hc_optimal, Form("PhaseID %d: noise distribution across all channels", optimal_phaseid), "fl");
#    leg->SetTextSize(0.02);
#    leg->SetTextFont(2);
#    leg->Draw();
    leg.Draw()
    # Add some annotations?
    tN = rt.TLatex()
    tN.SetNDC()
    tN.SetTextSize(0.025)
    tN.SetTextAlign(12)
    tN.SetTextAngle(343)
    tN.DrawLatex(0.14, 0.66, "Normal Branch")
    
    tB = rt.TLatex()
    tB.SetNDC()
    tB.SetTextSize(0.025)
    tB.SetTextAlign(12)
    tB.SetTextAngle(0)
    tB.DrawLatex(0.55, 0.41, "Biased Region")
    
    tS = rt.TLatex()
    tS.SetNDC()
    tS.SetTextSize(0.025)
    tS.SetTextAlign(12)
    tS.SetTextAngle(282)
    tS.DrawLatex(0.49, 0.77, "SC Region")
    
#    t = new TLatex();
#    t->SetNDC();
#    t->SetTextFont(62);
#    t->SetTextColor(36);
#    t->SetTextSize(0.08);
#    t->SetTextAlign(12);
#    t->DrawLatex(0.6,0.85,"p - p");
#
#    t->SetTextSize(0.05);
#    t->DrawLatex(0.6,0.79,"Direct #gamma");
#    t->DrawLatex(0.6,0.75,"#theta = 90^{o}");
#
#    t->DrawLatex(0.70,0.55,"H(z)");
#    t->DrawLatex(0.68,0.50,"(barn)");
#
#    t->SetTextSize(0.045);
#    t->SetTextColor(46);
#    t->DrawLatex(0.20,0.30,"#sqrt{s}, GeV");
#    t->DrawLatex(0.22,0.26,"63");
#    t->DrawLatex(0.22,0.22,"200");
#    t->DrawLatex(0.22,0.18,"500");
#
#    t->SetTextSize(0.05);
#    t->SetTextColor(1);
#    t->DrawLatex(0.88,0.06,"z");
    #c.BuildLegend()
    c.Update()
    # Now save the damned thing
    c.SaveAs(fName + '.png')
    fName = outPath + '/root/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + str(int(float(temperature)*1e3)) + 'uK'
    c.SaveAs(fName + '.C')
    c.Close()
    del mg
    del c
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


def read_from_ivroot(filename, branches):
    '''Read data from special IV root file and put it into a dictionary
    TDir - iv
        TTrees - temperatures
            TBranches - iv properties
    '''
    print('Trying to open {}'.format(filename))
    iv_dictionary = {}
    treeNames = get_treeNames(filename)
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


def make_tes_multiplot(output_path, data_channel, iv_dictionary, fit_parameters):
    '''Make a plot of all temperatures at once
    rTES vs iBias
    
    '''
    # Convert fit parameters to R values
    #R = fit_to_resistance(fit_parameters, fit_type='tes')
    # Current vs Voltage
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    for temperature, data in iv_dictionary.items():
        params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
        labels = {'xlabel': 'Bias Current [uA]', 'ylabel': 'TES Resistance [m \Omega]', 'title': 'Channel {} TES Resistance vs Bias Current'.format(data_channel)}
        ax = generic_fitplot_with_errors(ax=ax, x=data['iBias'], y=data['rTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
        #ax = add_model_fits(ax=ax, x=data['vTES'], y=data['iTES'], model=fit_parameters, xScale=xScale, yScale=yScale, model_function=lin_sq)
        #ax = add_fit_textbox(ax=ax, R=R, model=fit_parameters)
    ax.set_ylim((0*yScale, 1*yScale))
    ax.set_xlim((-20, 20))
    fName = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    return None


def tes_plots(output_path, data_channel, temperature, data, fit_parameters):
    '''Helper function to generate standard TES plots
    iTES vs vTES
    rTES vs iTES
    rTES vs vTES
    rTES vs iBias
    pTES vs rTES
    pTES vs vTES
    '''
    
    # Convert fit parameters to R values
    R = fit_to_resistance(fit_parameters, fit_type='tes')
    # Current vs Voltage
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e6
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': data['iTES_rms']*yScale}
    labels = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Current [uA]', 'title': 'Channel {} TES Current vs TES Voltage for T = {} mK'.format(data_channel, temperature)}
    
    ax = generic_fitplot_with_errors(ax=ax, x=data['vTES'], y=data['iTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    ax = add_model_fits(ax=ax, x=data['vTES'], y=data['iTES'], model=fit_parameters, xScale=xScale, yScale=yScale, model_function=lin_sq)
    ax = add_fit_textbox(ax=ax, R=R, model=fit_parameters)
    
    fName = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    
    # Resistance vs Current
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['iTES_rms']*xScale, 'yerr': data['rTES_rms']*yScale}
    labels = {'xlabel': 'TES Current [uA]', 'ylabel': 'TES Resistance [mOhm]', 'title': 'Channel {} TES Resistance vs TES Current for T = {} mK'.format(data_channel, temperature)}
    ax = generic_fitplot_with_errors(ax=ax, x=data['iTES'], y=data['rTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'rTES_vs_iTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    # Resistance vs Voltage
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': data['rTES_rms']*yScale}
    labels = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Resistance [mOhm]', 'title': 'Channel {} TES Resistance vs TES Voltage for T = {} mK'.format(data_channel, temperature)}
    ax = generic_fitplot_with_errors(ax=ax, x=data['vTES'], y=data['rTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    # Resistance vs Bias Current
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['iBias_rms']*xScale, 'yerr': data['rTES_rms']*yScale}
    labels = {'xlabel': 'Bias Current [uA]', 'ylabel': 'TES Resistance [mOhm]', 'title': 'Channel {} TES Resistance vs Bias Current for T = {} mK'.format(data_channel, temperature)}
    ax = generic_fitplot_with_errors(ax=ax, x=data['iBias'], y=data['rTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    # Power vs rTES
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e12
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['rTES_rms']*xScale, 'yerr': data['pTES_rms']*yScale}
    labels = {'xlabel': 'TES Resistance [mOhm]', 'ylabel': 'TES Power [pW]', 'title': 'Channel {} TES Power vs TES Resistance for T = {} mK'.format(data_channel, temperature)}
    ax = generic_fitplot_with_errors(ax=ax, x=data['rTES'], y=data['pTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'pTES_vs_rTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    # Power vs vTES
    # Note this ideally is a parabola
    cut = np.logical_and(data['rTES'] > 500e-3, data['rTES'] < 2*500e-3)
    if nsum(cut) < 3:
        cut = np.ones(data['pTES'].size, dtype=bool)
    v = data['vTES'][cut]
    p = data['pTES'][cut]
    prms = data['pTES_rms'][cut]
    result, pcov = curve_fit(quad_sq, v, p, sigma=prms, absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))
    pFit = quad_sq(data['vTES'], result[0], result[1], result[2])
    fitResult = FitParameters()
    fitResult.left.set_values(result, perr)
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e12
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': data['pTES_rms']*yScale}
    labels = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Power [pW]', 'title': 'Channel {} TES Power vs TES Resistance for T = {} mK'.format(data_channel, temperature)}
    
    ax = generic_fitplot_with_errors(ax=ax, x=data['vTES'], y=data['pTES'], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    ax = add_model_fits(ax=ax, x=data['vTES'], y=data['pTES'], model=fitResult, xScale=xScale, yScale=yScale, model_function=quad_sq)
    ax = add_power_voltage_textbox(ax=ax, model=fitResult)
    
    fName = output_path + '/' + 'pTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, ax, fName)
    
    #iv_fitplot(data['vTES'], data['iTES'], data['vTES_rms'], data['iTES_rms'], [data['rTES'], data['rTES_err'], fit_parameters], xLabel, yLabel, titleStr, fName, sc=sc_bounds, xScale=1e6, yScale=1e6, logx='linear', logy='linear')
    return None


def make_tes_plots(output_path, data_channel, iv_dictionary, fit_dictionary):
    '''Loop through data to generate TES specific plots'''
    
    for temperature, data in iv_dictionary.items():
        tes_plots(output_path, data_channel, temperature, data, fit_dictionary[temperature])
    # Make a for all temperatures here
    make_tes_multiplot(output_path=outPath, data_channel=data_channel, iv_dictionary=iv_dictionary, fit_parameters=fit_dictionary)
    return None


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
    #branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform', 'EPCal_K']
    branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform', 'NT']
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
    # For a given waveform the exact timestamp of sample number N is as follows:
    # t[N] = Timestamp_s + Timestamp_mus*1e-6 + N*SamplingWidth_s
    # Note that there will be NEvents different values of Timestamp_s and Timestamp_mus, each of these containing 1/SamplingWidth_s samples
    # (Note that NEvents = NEntries/NChannels)
    time_values = iv_data['Timestamp_s'] + iv_data['Timestamp_mus']/1e6
    #temperatures = iv_data['EPCal_K']
    temperatures = iv_data['NT']
    cut = temperatures > -1
    # Get unique time values for valid temperatures
    time_values, idx = np.unique(time_values[cut], return_index=True)
    # Reshape temperatures to be only the valid values that correspond to unique times
    temperatures = temperatures[cut]
    temperatures = temperatures[idx]
    test_plot(time_values, temperatures, 'Unix Time', 'Temperature [K]', output_path + '/' + 'quick_look_Tvt.png')
    print('Processing IV waveforms...')
    # Next process waveforms into input and output arrays
    waveForms = {ch: {} for ch in np.unique(iv_data['Channel'])}
    nChan = np.unique(iv_data['Channel']).size
    for ev, ch in enumerate(iv_data['Channel'][cut]):
        waveForms[ch][ev//nChan] = iv_data['Waveform'][ev]
    # We now have a nested dictionary of waveforms.
    # waveForms[ch] consists of a dictionary of Nevents keys (actual events)
    # So waveForms[ch][ev] will be a numpy array with size = NumberOfSamples.
    # The timestamp of waveForms[ch][ev][sample] is time_values[ev] + sample*SamplingWidth_s
    
    # Ultimately let us collapse the finely sampled waveforms into more coarse waveforms.
    mean_waveforms = {}
    rms_waveforms = {}
    for channel in waveForms.keys():
        mean_waveforms[channel], rms_waveforms[channel], mean_time_values = process_waveform(waveForms[channel], time_values, iv_data['SamplingWidth_s'][0], number_of_windows=1)
    print('The number of things in the mean time values are: {} and the number of waveforms are: {}'.format(np.size(mean_time_values), len(mean_waveforms[5])))
#    
#    # Ultimately the cut we form from times will tell us what times, and hence events to cut
#    # waveForms are dicts of channels with dicts of event numbers that point to the event's waveform
#    # Collapse a waveform down to a single value per event means we can form np arrays then
#    mean_waveforms = {}
#    rms_waveforms = {}
#    #print('waveforms keys: {}'.format(list(waveForms[biasChannel].keys())))
#    for ch in waveForms.keys():
#        mean_waveforms[ch], rms_waveforms[ch] = process_waveform(waveForms[ch], 'mean')
#    print('The number of things in the time_values are: {} and the number of waveforms are: {}'.format(np.size(time_values), len(mean_waveforms[5])))
    return time_values, temperatures, mean_waveforms, rms_waveforms, mean_time_values


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
    temp_sensor = 'NT'
    if temp_sensor == 'EP_Cal':
        Tstep = 5e-5
        lenBuff = 10
    elif temp_sensor == 'NT':
        # Usually is noisy
        Tstep = 7e-4
        lenBuff = 400
    print('Attempting to find temperature steps now...')
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
    time_values, temperatures, mean_waveforms, rms_waveforms, mean_time_values = format_iv_data(iv_data, output_path)
    # Next identify temperature steps
    timeList = find_temperature_steps(time_values, temperatures, output_path)
    # Now we have our timeList so we can in theory loop through it and generate IV curves for selected data!
    print('Diagnostics:')
    print('Channels: {}'.format(np.unique(iv_data['Channel'])))
    print('length of time: {}'.format(time_values.size))
    print('length of mean waveforms vector: {}'.format(mean_waveforms[bias_channel].size))
    print('There are {} temperature steps with values of: {}'.format(len(timeList), timeList))
    return mean_time_values, temperatures, mean_waveforms, rms_waveforms, timeList


def fit_sc_branch(x, y, sigmaY, plane):
    '''Walk and fit the superconducting branch
    In the vOut vs iBias plane x = iBias, y = vOut --> dy/dx ~ resistance
    In the iTES vs vTES plane x = vTES, y = iTES --> dy/dx ~ 1/resistance
    '''
    # First generate a sortKey since dy/dx will require us to be sorted
    sortKey = np.argsort(x)
    (evLeft, evRight) = walk_sc(x[sortKey], y[sortKey], plane=plane)
    print('Diagnostics: The input into curve_fit is as follows:')
    print('x size: {}, y size: {}, x NaN: {}, y NaN: {}'.format(x[sortKey][evLeft:evRight].size, y[sortKey][evLeft:evRight].size, nsum(np.isnan(x[sortKey][evLeft:evRight])), nsum(np.isnan(y[sortKey][evLeft:evRight]))))
    result, pcov = curve_fit(lin_sq, x[sortKey][evLeft:evRight], y[sortKey][evLeft:evRight], sigma=sigmaY[sortKey][evLeft:evRight], absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))
    # In order to properly plot the superconducting branch fit try to find the boundaries of the SC region
    # One possibility is that the region has the smallest and largest y-value excursions. However this may not be the case
    # and certainly unless the data is sorted these indices are meaningless to use in a slice
    #index_y_min = np.argmin(y)
    #index_y_max = np.argmax(y)
    return result, perr #, (index_y_max, index_y_min)


def fit_normal_branches(x, y, sigmaY):
    '''Walk and fit the normal branches in the vOut vs iBias plane.'''
    # Generate a sortKey since dy/dx must be sorted
    sortKey = np.argsort(x)
    # Get the left side normal branch first
    left_ev = walk_normal(x[sortKey], y[sortKey], 'left')
    left_result, pcov = curve_fit(lin_sq, x[sortKey][0:left_ev], y[sortKey][0:left_ev], sigma=sigmaY[sortKey][0:left_ev], absolute_sigma=True, method='trf')
    left_perr = sqrt(np.diag(pcov))
    # Now get the other branch
    right_ev = walk_normal(x[sortKey], y[sortKey], 'right')
    right_result, pcov = curve_fit(lin_sq, x[sortKey][right_ev:], y[sortKey][right_ev:], sigma=sigmaY[sortKey][right_ev:], absolute_sigma=True, method='trf')
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


def fit_to_resistance(fit_parameters, fit_type='iv', Rp=None, Rp_rms=None):
    '''Given a FitParameters object convert to Resistance and Resistance error TESResistance objects
    
    If a parasitic resistance is provided subtract it from the normal and superconducting branches and assign it
    to the parasitic property.
    
    If no parasitic resistance is provided assume that the superconducting region values are purely parasitic
    and assign the resulting value to both properties.
    
    '''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    
    R = TESResistance()
    # The interpretation of the fit parameters depends on what plane we are in
    if fit_type == 'iv':
        # We fit something of the form vOut = a*iBias + b
        Rsc = Rsh * ((M*Rfb)/fit_parameters.sc.result[0] - 1)
        if Rp is None:
            Rp = Rsc
        else:
            Rsc = Rsc - Rp
        Rsc_rms = np.abs( (-1*M*Rfb*Rsh)/pow2(fit_parameters.sc.result[0]) * fit_parameters.sc.error[0] )
        if Rp_rms is None:
            Rp_rms = Rsc_rms
        else:
            Rsc_rms = sqrt(pow2(Rsc_rms) + pow2(Rp_rms))
        if fit_parameters.left.result is None:
            Rl, Rl_rms = None, None
        else:
            Rl = (M*Rfb*Rsh)/fit_parameters.left.result[0] - Rsh - Rp
            Rl_rms = sqrt(pow2(fit_parameters.left.error[0] * (-1*M*Rfb*Rsh)/pow2(fit_parameters.left.result[0])) + pow2(-1*Rp_rms))
        if fit_parameters.right.result is None:
            Rr, Rr_rms = None, None
        else:
            Rr = (M*Rfb*Rsh)/fit_parameters.right.result[0] - Rsh - Rp
            Rr_rms = sqrt(pow2(fit_parameters.right.error[0] * (-1*M*Rfb*Rsh)/pow2(fit_parameters.right.result[0])) + pow2(-1*Rp_rms))
    elif fit_type == 'tes':
        # Here we fit something of the form iTES = a*vTES + b
        # Fundamentally iTES = vTES/rTES ...
        Rsc = 1/fit_parameters.sc.result[0]
        if Rp is None:
            Rp = Rsc
        else:
            Rsc = Rsc - Rp
        Rsc_rms = np.abs((-1*fit_parameters.sc.error[0])/pow2(fit_parameters.sc.result[0]))
        if Rp_rms is None:
            Rp_rms = Rsc_rms
        else:
            Rsc_rms = sqrt(pow2(Rsc_rms) + pow2(Rp_rms))
        if fit_parameters.left.result is None:
            Rl, Rl_rms = None, None
        else:
            Rl = 1/fit_parameters.left.result[0]
            Rl_rms = np.abs((-1*fit_parameters.left.error[0])/pow2(fit_parameters.left.result[0]))
        if fit_parameters.right.result is None:
            Rr, Rr_rms = None, None
        else:
            Rr = 1/fit_parameters.right.result[0]
            Rr_rms = np.abs((-1*fit_parameters.right.error[0])/pow2(fit_parameters.right.result[0]))
    R.parasitic.set_values(Rp, Rp_rms)
    R.left.set_values(Rl, Rl_rms)
    R.right.set_values(Rr, Rr_rms)
    R.sc.set_values(Rsc, Rsc_rms)
    return R


def fit_iv_regions(x, y, sigmaY, fittype='iv', plane='iv'):
    '''Fit the iv data regions and extract fit parameters'''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    
    fitParams = FitParameters()
    # We need to walk and fit the superconducting region first since there RTES = 0
    result, perr = fit_sc_branch(x, y, sigmaY, plane)
    # Now we can fit the rest
    left_result, left_perr, right_result, right_perr = fit_normal_branches(x, y, sigmaY)
#    # The interpretation of the fit parameters depends on what plane we are in
#    if fittype == 'iv':
#        # We fit something of the form vOut = a*iBias + b
#        Rp = Rsh * ((M*Rfb/result[0]) - 1)
#        Rp_rms = np.abs( ((-1*M*Rfb*Rsh)/(pow2(result[0])))*perr[0] )
#        Rl = (M*Rfb*Rsh/left_result[0]) - Rsh - R.parasitic
#        Rl_rms = sqrt(pow2(left_perr[0]*(-1*M*Rfb*Rsh)/pow2(left_result[0])) + pow2(-1*Rerr.parasitic))
#        Rr = (M*Rfb*Rsh/right_result[0]) - Rsh - R.parasitic
#        Rr_rms = sqrt(pow2(right_perr[0]*(-1*M*Rfb*Rsh)/pow2(right_result[0])) + pow2(-1*Rerr.parasitic))
#    elif fittype == 'tes':
#        # Here we fit something of the form iTES = a*vTES + b
#        # Fundamentally iTES = vTES/rTES ...
#        Rp = 1/result[0]
#        Rp_rms = np.abs(-1*perr[0]/pow2(result[0]))
#        Rl = 1/left_result[0]
#        Rl_rms = np.abs(-1*left_perr[0]/pow2(left_result[0]))
#        Rr = 1/right_result[0]
#        Rr_rms = np.abs(-1*right_perr[0]/pow2(right_result[0]))
#    R.parasitic.set_values(Rp, Rp_rms)
#    R.left.set_values(Rl, Rl_rms)
#    R.right.set_values(Rr, Rr_rms)
    fitParams.sc.set_values(result, perr)
    fitParams.left.set_values(left_result, left_perr)
    fitParams.right.set_values(right_result, right_perr)
    #TODO: Make sure everything is right here with the equations and error prop.
    return fitParams


def get_parasitic_resistances(iv_dictionary):
    '''Loop through IV data to obtain parasitic series resistance'''
    parasitic_dictionary = {}
    fitParams = FitParameters()
    minT = list(iv_dictionary.keys())[np.argmin([float(T) for T in iv_dictionary.keys()])]
    for temperature, iv_data in iv_dictionary.items():
        print('Attempting to fit superconducting branch for temperature: {} mK'.format(temperature))
        result, perr = fit_sc_branch(iv_data['iBias'], iv_data['vOut'], iv_data['vOut_rms'], plane='iv')
        fitParams.sc.set_values(result, perr)
        R = fit_to_resistance(fitParams, fit_type='iv')
        parasitic_dictionary[temperature] = R.parasitic
    return parasitic_dictionary, minT


def process_iv_curves(outPath, data_channel, iv_dictionary):
    '''Processes the IV data to obtain electrical parameters
    The IV curve has 3 regions -- normal, biased, superconducting
    When the temperature becomes warm enough the superconducting 
    and biased regions will vanish leaving just a normal resistance line
    
    In an ideal circuit the slope of the superconducting region will be infinity
    denoting 0 resistance. There is actually some parasitic resistance in series with the TES
    and so there is a slope in the superconducting region. This slope will gradually align with the normal
    region slope as temperature increases due to the TES becoming partially normal.
    
    At the coldest temperature we can assume the TES resistance = 0 and so the slope is purely parasitic.
    This fitted parasitic resistance will be subtracted from resistances in the normal region to give a proper
    estimate of the normal resistance and an estimate of the residual TES resistance in the 'superconducting' region.
    
    Additionally the IV curve must be corrected so that iBias = 0 corresponds to vOut = 0.
    '''
    
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    keys = ['iBias', 'iBias_rms', 'vOut', 'vOut_rms']
    fit_parameters_dictionary = {}
    
    # First we should try to obtain a measure of the parasitic series resistance. This value will be subtracted from
    # subsequent fitted values of the TES resistance
    parasitic_dictionary, minT = get_parasitic_resistances(iv_dictionary)
    Rp, Rp_rms = parasitic_dictionary[minT].value, parasitic_dictionary[minT].rms
    
    # Loop through the iv data now and obtain fit parameters and correct alignment
    for temperature, iv_data in iv_dictionary.items():
        fit_parameters_dictionary[temperature] = fit_iv_regions(x=iv_data['iBias'], y=iv_data['vOut'], sigmaY=iv_data['vOut_rms'], fittype='iv', plane='iv')
        iv_data['vOut'] -= fit_parameters_dictionary[temperature].sc.result[1]
        # Re-walk on shifted data
        fit_parameters_dictionary[temperature] = fit_iv_regions(x=iv_data['iBias'], y=iv_data['vOut'], sigmaY=iv_data['vOut_rms'], fittype='iv', plane='iv')
    # Next loop through to generate plots
    for temperature, iv_data in iv_dictionary.items():
        # Make I-V plot
        xLabel = 'Bias Current [uA]'
        yLabel = 'Output Voltage [mV]'
        titleStr = 'Channel {} Output Voltage vs Bias Current for T = {} mK'.format(data_channel, temperature)
        fName = outPath + '/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
        iv_fitplot(iv_data['iBias'], iv_data['vOut'], iv_data['iBias_rms'], iv_data['vOut_rms'], fit_parameters_dictionary[temperature], parasitic_dictionary[minT], xLabel, yLabel, titleStr, fName, xScale=1e6, yScale=1e3, logx='linear', logy='linear')
        # Let's make a ROOT style plot (yuck)
        make_root_plot(outPath, data_channel, temperature, iv_data, fit_parameters_dictionary[temperature], parasitic_dictionary[minT], xScale=1e6, yScale=1e3)
    return iv_dictionary, fit_parameters_dictionary, parasitic_dictionary


def get_PT_curves(output_path, data_channel, iv_dictionary):
    '''Generate a power vs temperature curve for a TES'''
    # Need to select power in the biased region, i.e. where P(R) ~ constant
    # Try something at 0.5*Rn
    T = np.empty(0)
    P = np.empty(0)
    P_rms = np.empty(0)
    for temperature, iv_data in iv_dictionary.items():
        # Create cut to select only data going in the Normal to SC mode
        # This happens in situations as follows:
        # if iBias > 0 and iBias is decreasing over time
        # if iBias < 0 and iBias is increasing
        # Basically whenever iBias is approaching 0
        diBias = np.gradient(iv_data['iBias'], edge_order=2)
        cNtoSC_pos = np.logical_and(iv_data['iBias'] > 0, diBias < 0)
        cNtoSC_neg = np.logical_and(iv_data['iBias'] <= 0, diBias > 0)
        cNtoSC = np.logical_or(cNtoSC_pos, cNtoSC_neg)
        # Also select data that is some fraction of the normal resistance, say 20%
        cut = np.logical_and(iv_data['rTES'] > 0.3*0.540 - 20e-3, iv_data['rTES'] < 0.3*0.540 + 20e-3)
        cut = np.logical_and(cut, cNtoSC)
        if nsum(cut) > 0:
            T = np.append(T, float(temperature)*1e-3)
            P = np.append(P, np.mean(iv_data['pTES'][cut]))
            P_rms = np.append(P_rms, np.std(iv_data['pTES'][cut]))
    # Attempt to fit it to a power function
    lBounds = [1e-15, 0, 10e-3]
    uBounds = [1e-5, 10, 100e-3]
    cutT = T < 35e-3
    x0 = [5e-06, 5, 35e-3]
    results, pcov = curve_fit(tes_power_polynomial, T[cutT], P[cutT], sigma=P_rms[cutT], p0=x0, bounds=(lBounds, uBounds), absolute_sigma=True, method='trf', max_nfev=1e4)
    #results, pcov = curve_fit(tes_power_polynomial, T[cutT], P[cutT], sigma=P_rms[cutT], absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))
    #results = [results[0], 5, results[1]]
    #perr = [perr[0], 0, perr[1]]
    #x0 = [x0[0], 5, x0[1]]
    #results = minimize(nll_error, x0=np.array(x0), args=(P,P_rms,T,tes_power_polynomial), method='Nelder-Mead', options={'fatol':1e-15})
    #perr = results.x/100
    #results = results.x
    #results, pcov = curve_fit(tes_power_polynomial, T[cutT], P[cutT], method='dogbox')
    
    fitResult = FitParameters()
    fitResult.left.set_values(results, perr)
    fitResult.right.set_values(x0, [0,0,0])
    # Next make a P-T plot
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e15
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': P_rms*yScale}
    labels = {'xlabel': 'Temperature [mK]', 'ylabel': 'TES Power [fW]', 'title': 'Channel {} TES Power vs Temperature'.format(data_channel)}
    
    ax = generic_fitplot_with_errors(ax=ax, x=T, y=P, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    #ax.set_ylim((-1,1))
    ax = add_model_fits(ax=ax, x=T, y=P, model=fitResult, xScale=xScale, yScale=yScale, model_function=tes_power_polynomial)
    ax = add_power_temperature_textbox(ax=ax, model=fitResult)
    
    fName = output_path + '/' + 'pTES_vs_T_ch_' + str(data_channel)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    print('Results: k = {}, n = {}, Tb = {}'.format(*results))
    return None
    

def get_RT_curves(output_path, data_channel, iv_dictionary):
    '''Generate a resistance vs temperature curve for a TES'''
    # Rtes = R(i,T) really so select a fixed i and across multiple temperatures obtain values for R and then plot
    T = np.empty(0)
    R = np.empty(0)
    for temperature, iv_data in iv_dictionary.items():
        cut = np.logical_and(iv_data['iBias'] > 0.0e-6, iv_data['iBias'] < 1e-6)
        if nsum(cut) > 0:
            T = np.append(T, float(temperature)*1e-3) # T in K
            R = np.append(R, np.mean(iv_data['rTES'][cut]))
    # Next make an R-T plot
    # R vs T
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    labels = {'xlabel': 'Temperature [mK]', 'ylabel': 'TES Resistance [m' + r'$\Omega$' +']', 'title': 'Channel {} TES Resistance vs Temperature'.format(data_channel)}
    
    ax = generic_fitplot_with_errors(ax=ax, x=T, y=R, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    #ax.set_ylim((-1,1))
    #ax = add_model_fits(ax=ax, x=data['vTES'], y=data['iTES'], model=fit_parameters, sc_bounds=sc_bounds, xScale=xScale, yScale=yScale, model_function=lin_sq)
    #ax = add_fit_textbox(ax=ax, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)
    
    fName = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    
    # Make R vs T only for times we are going from higher iBias to lower iBias values
    # One way, assuming noise does not cause overlaps, is to only select points where iBias[i] > iBias[i+1]
    # If I take the diff array I get the following: diBias[i] = iBias[i] - iBias[i-1]. If diBias[i] < 0 then iBias is descending
    # So let's use that then.
    
    # Rtes = R(i,T) really so select a fixed i and across multiple temperatures obtain values for R and then plot
    T = np.empty(0)
    R = np.empty(0)
    Rrms = np.empty(0)
    T_desc = np.empty(0)
    R_desc = np.empty(0)
    Rrms_desc = np.empty(0)
    fitResult = FitParameters()
    i_select = 1e-6
    for temperature, iv_data in iv_dictionary.items():
        diBias = np.gradient(iv_data['iBias'], edge_order=2)
        cut = np.logical_and(iv_data['iBias'] > i_select - 0.2e-6, iv_data['iBias'] < i_select + 0.2e-6)
        print('the sum of cut is: {}'.format(nsum(cut)))
        cut1 = np.logical_and(iv_data['iBias'] > 0, diBias < 0)
        cut2 = np.logical_and(iv_data['iBias'] <= 0, diBias > 0)
        dcut = np.logical_or(cut1, cut2)
        cut_desc = np.logical_and(cut, dcut)
        cut_asc = np.logical_and(cut, ~dcut)
        if nsum(cut_asc) > 0:
            T = np.append(T, float(temperature)*1e-3) # T in K
            R = np.append(R, np.mean(iv_data['rTES'][cut_asc]))
            Rrms = np.append(Rrms, np.std(iv_data['rTES'][cut_asc]))
        if nsum(cut_desc) > 0:
            T_desc = np.append(T_desc, float(temperature)*1e-3) # T in K
            R_desc = np.append(R_desc, np.mean(iv_data['rTES'][cut_desc]))
            Rrms_desc = np.append(Rrms_desc, np.std(iv_data['rTES'][cut_desc]))
    # Next make an R-T plot
    # Try a fit?
    # [Rn, Rp, Tc, Tw]
    # In new fit we have [C, D, B, A] --> A = 1/Tw, B = -Tc/Tw
    sortKey = np.argsort(T)
    x0 = [1, 0, T[sortKey][np.gradient(R[sortKey], T[sortKey], edge_order=2).argmax()]*1.1, 1e-3]
    #x0 = [1, 0, -T[sortKey][np.gradient(R[sortKey], T[sortKey], edge_order=2).argmax()]/1e-3,  1/1e-3]
    print('For ascending fit initial guess is {}'.format(x0))
    result, pcov = curve_fit(tanh_tc, T, R, sigma=Rrms, absolute_sigma=True, p0=x0, method='trf')
    perr = np.sqrt(np.diag(pcov))
    print('Ascending (SC -> N): Rn = {} mOhm, Rp = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fitResult.left.set_values(result, perr)
    
    # Try a fit?
    sortKey = np.argsort(T_desc)
    x0 = [1, 0, T_desc[sortKey][np.gradient(R_desc[sortKey], T_desc[sortKey], edge_order=2).argmax()]*1.1, 1e-3]
    #x0 = [1, 0, -T_desc[sortKey][np.gradient(R_desc[sortKey], T_desc[sortKey], edge_order=2).argmax()]/1e-3, 1/1e-3]
    print('For descending fit (N->S) initial guess is {}'.format(x0))
    result_desc, pcov_desc = curve_fit(tanh_tc, T_desc, R_desc, sigma=Rrms_desc, p0=x0, absolute_sigma=True, method='trf')
    perr_desc = np.sqrt(np.diag(pcov_desc))
    print('Descending (N -> SC): Rn = {} mOhm, Rp = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result_desc]))
    fitResult.right.set_values(result_desc, perr_desc)
    # R vs T
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    sortKey = np.argsort(T)
    
    labels = {'xlabel': 'Temperature [mK]', 'ylabel': 'TES Resistance [m' + r'$\Omega$' +']', 'title': 'Channel {}'.format(data_channel) + ' TES Resistance vs Temperature for Bias Current = {}'.format(i_select*1e6)  + r'$\mu$' + 'A'}
    
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': Rrms[sortKey]*yScale}
    ax = generic_fitplot_with_errors(ax=ax, x=T[sortKey], y=R[sortKey], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    
    sortKey = np.argsort(T_desc)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green', 'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': Rrms_desc[sortKey]*yScale}
    ax = generic_fitplot_with_errors(ax=ax, x=T_desc[sortKey], y=R_desc[sortKey], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    #ax.set_ylim((-1,1))
    ax = add_model_fits(ax=ax, x=T, y=R, model=fitResult, xScale=xScale, yScale=yScale, model_function=tanh_tc)
    ax = add_resistance_temperature_textbox(ax=ax, model=fitResult)
    ax.legend(['SC to N', 'N to SC'])
    fName = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_descending_iBias_' + str(i_select*1e6) + 'uA'
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    
    
    # We can try to plot alpha vs R as well why not
    # alpha = To/Ro * dR/dT --> dln(R)/dln(T)
    #alpha = np.gradient(np.log(R), np.log(T), edge_order=2)
    modelT = np.linspace(T.min(), T.max(), 100)
    modelR = tanh_tc(modelT, *fitResult.right.result)
    model_sortKey = np.argsort(modelT)
    #model_alpha = (modelT[model_sortKey]/modelR[model_sortKey])*np.gradient(modelR[model_sortKey], modelT[model_sortKey], edge_order=1)
    #model_alpha = np.gradient(np.log(modelR[model_sortKey]), np.log(modelT[model_sortKey]), edge_order=1)
    model_alpha = (modelT[model_sortKey]/modelR[model_sortKey])*np.gradient(modelR[model_sortKey], modelT[model_sortKey], edge_order=2)
    print('The max alpha is: {}'.format(np.max(model_alpha)))
    sortKey = np.argsort(T)
    alpha = (T[sortKey]/R[sortKey])*np.gradient(R[sortKey], T[sortKey], edge_order=1)/1e3
    #alpha = np.gradient(np.log(R[sortKey]), np.log(T[sortKey]), edge_order=1)
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': '-', 'xerr': None, 'yerr': None}
    labels = {'xlabel': 'TES Resistance [m' + 'r$\Omega$' + ']', 'ylabel': r'$\alpha$', 'title': 'Channel {} TES '.format(data_channel) + r'$\alpha$' +' vs Resistance'}
    ax = generic_fitplot_with_errors(ax=ax, x=modelR[model_sortKey], y=model_alpha, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    ax = generic_fitplot_with_errors(ax=ax, x=R[sortKey], y=alpha, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    #ax.set_ylim((-1,1))
    #ax = add_model_fits(ax=ax, x=data['vTES'], y=data['iTES'], model=fit_parameters, sc_bounds=sc_bounds, xScale=xScale, yScale=yScale, model_function=lin_sq)
    #ax = add_fit_textbox(ax=ax, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)
    fName = output_path + '/' + 'alpha_vs_rTES_ch_' + str(data_channel)
    #ax.set_ylim((0,150))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    
    # alpha vs T
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    labels = {'xlabel': 'Temperature [mK]', 'ylabel': r'$\alpha$', 'title': 'Channel {} TES '.format(data_channel) + r'$\alpha$' +' vs Temperature'}
    
    ax = generic_fitplot_with_errors(ax=ax, x=T[sortKey], y=alpha, labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    #ax.set_ylim((-1,1))
    #ax = add_model_fits(ax=ax, x=data['vTES'], y=data['iTES'], model=fit_parameters, sc_bounds=sc_bounds, xScale=xScale, yScale=yScale, model_function=lin_sq)
    #ax = add_fit_textbox(ax=ax, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)
    fName = output_path + '/' + 'alpha_vs_T_ch_' + str(data_channel)
    ax.set_ylim((0,150))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    
    # We can get R-T curves for multiple current selections as well :)
    # Proceed to do 0-1, 1-2, 2-3, up to 9-10
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': '-', 'xerr': None, 'yerr': None}
    labels = {'xlabel': 'Temperature [mK]', 'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']', 'title': 'Channel {} TES Resistance vs Temperature'.format(data_channel)}
    for i in range(10):
        T = np.empty(0)
        R = np.empty(0)
        for temperature, iv_data in iv_dictionary.items():
            diBias = np.gradient(iv_data['iBias'], edge_order=2)
            cut = np.logical_and(iv_data['iBias'] > i*1e-6, iv_data['iBias'] < (i+1)*1e-6)
            cut1 = np.logical_and(iv_data['iBias'] > 0, diBias < 0)
            cut2 = np.logical_and(iv_data['iBias'] <= 0, diBias > 0)
            dcut = np.logical_or(cut1, cut2)
            cut = np.logical_and(cut, dcut)
            if nsum(cut) > 0:
                T = np.append(T, float(temperature)*1e-3)
                R = np.append(R, np.mean(iv_data['rTES'][cut]))
        sortKey = np.argsort(T)
        ax = generic_fitplot_with_errors(ax=ax, x=T[sortKey], y=R[sortKey], labels=labels, params=params, xScale=xScale, yScale=yScale, logx='linear', logy='linear')
    fName = output_path + '/' + 'rTES_vs_T_multi_ch_' + str(data_channel)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    save_plot(fig, ax, fName)
    
    return None


def process_tes_curves(outPath, data_channel, iv_dictionary):
    '''Take TES data and find Rp and Rn values.'''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    fit_dictionary = {}
    for temperature, iv_data in iv_dictionary.items():
        print('Processing TES for T = {} mK'.format(temperature))
        fitParams = fit_iv_regions(x=iv_data['vTES'], y=iv_data['iTES'], sigmaY=iv_data['iTES_rms'], fittype='tes', plane='tes')
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
        #iv_data['R'] = R
        #iv_data['Rerr'] = Rerr
        # Obtain R and P values
        fit_dictionary[temperature] = fitParams
    return iv_dictionary, fit_dictionary


def get_TES_values(outPath, data_channel, iv_dictionary, parasitic_dictionary):
    '''From I-V data values compute the TES values for iTES and vTES, ultimately yielding rTES'''
    squid_parameters = get_squid_parameters(2)
    Rsh = squid_parameters['Rsh']
    M = squid_parameters['M']
    Rfb = squid_parameters['Rfb']
    # Test: Select the parasitic resistance from the lowest temperature fit to use for everything
    minT = list(parasitic_dictionary.keys())[np.argmin([float(T) for T in parasitic_dictionary.keys()])]
    Rp, Rp_rms = parasitic_dictionary[minT].value, parasitic_dictionary[minT].rms
    for temperature, iv_data in iv_dictionary.items():
        iv_data['iTES'] = get_iTES(iv_data['vOut'], Rfb, M)
        iv_data['iTES_rms'] = get_iTES(iv_data['vOut_rms'], Rfb, M)
        
        iv_data['vTES'] = get_vTES(iv_data['iBias'], iv_data['vOut'], Rfb, M, Rsh, Rp)
        iv_data['vTES_rms'] = get_vTES_rms(iv_data['iBias_rms'], iv_data['vOut'], iv_data['vOut_rms'], Rfb, M, Rsh, Rp, Rp_rms)
        
        iv_data['rTES'] = get_rTES(iv_data['iTES'], iv_data['vTES'])
        iv_data['rTES_rms'] = get_rTES_rms(iv_data['iTES'], iv_data['iTES_rms'], iv_data['vTES'], iv_data['vTES_rms'])
        
        iv_data['pTES'] = get_pTES(iv_data['iTES'], iv_data['vTES'])
        iv_data['pTES_rms'] = get_pTES_rms(iv_data['iTES'], iv_data['iTES_rms'], iv_data['vTES'], iv_data['vTES_rms'])
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
        vOut = mean_waveforms[data_channel][cut]
        vOut_rms = rms_waveforms[data_channel][cut]
        # Let us toss out T values wherein the digitizer rails
        if np.any(vOut_rms < 1e-9):
            print('Invalid digitizer response for T: {} mK'.format(np.round(mean_temperature*1e3, 3)))
            continue
        else:
            T = str(np.round(mean_temperature*1e3, 3))
            # Proceed to correct for SQUID Jumps
            # We should SORT everything by increasing time....
            sortKey = np.argsort(times)
            times = times[sortKey]
            iBias = iBias[sortKey]
            iBias_rms = iBias_rms[sortKey]
            vOut = vOut[sortKey]
            vOut_rms = vOut_rms[sortKey]
            print('Attempting to correct for SQUID jumps for temperature {}'.format(T))
            times, iBias, iBias_rms, vOut, vOut_rms = correct_squid_jumps(outPath, T, times, iBias, iBias_rms, vOut, vOut_rms)
            # We can technically get iTES at this point too since it is proportional to vOut but since it is let's not.
            print('Creating dictionary entry for T: {} mK'.format(T))
            # Make gradient to save as well
            # Try to do this: dV/dt and di/dt and then (dV/dt)/(di/dt) --> (dV/di)
            dvdt = np.gradient(vOut, times, edge_order=2)
            didt = np.gradient(iBias, times, edge_order=2)
            dvdi = dvdt/didt
            index_vector = np.asarray([i for i in range(dvdi.size)])
            iv_dictionary[T] = {'iBias': iBias, 'iBias_rms': iBias_rms, 'vOut': vOut, 'vOut_rms': vOut_rms, 't': times, 'dvdi': dvdi, 'dvdt': dvdt, 'didt': didt, 'index': index_vector}
    return iv_dictionary


def get_iv_data(input_path, output_path, squid_run, bias_channel, data_channel):
    '''Function that returns an iv dictionary from waveform root file'''
    time_values, temperatures, mean_waveforms, rms_waveforms, timeList = get_pyIV_data(input_path, output_path, squid_run, bias_channel)
    # Next chop up waveform data into an iv dictionary
    iv_dictionary = generate_iv_curves(output_path, time_values, temperatures, mean_waveforms, rms_waveforms, timeList, bias_channel, data_channel)
    return iv_dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', default='/Users/bwelliver/cuore/bolord/squid/', help='Specify full file path of input data')
    parser.add_argument('-o', '--outputPath', help='Path to put the plot directory. Defaults to input directory')
    parser.add_argument('-r', '--run', type=int, help='Specify the SQUID run number to use')
    parser.add_argument('-b', '--biasChannel', type=int, default=5, help='Specify the digitizer channel that corresponds to the bias channel. Defaults to 5')
    parser.add_argument('-d', '--dataChannel', type=int, default=7, help='Specify the digitizer channel that corresponds to the output channel. Defaults to 7')
    parser.add_argument('-s', '--makeRoot', action='store_true', help='Specify whether to write data to a root file')
    parser.add_argument('-L', '--readROOT', action='store_true', help='Read IV data from processed root file. Stored in outputPath /root/iv_data.root')
    parser.add_argument('-T', '--readTESROOT', action='store_true', help='Read IV and TES data from processed root file. Stored in outputPath /root/iv_data.root')
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
    
    # First step is to get basic IV data into a dictionary format. Either read raw files or load from a saved root file
    if args.readROOT is False and args.readTESROOT is False:
        iv_dictionary = get_iv_data(input_path=args.inputFile, output_path=outPath, squid_run=args.run, bias_channel=args.biasChannel, data_channel=args.dataChannel)
        # Next save the iv_curves
        save_to_root(outPath, iv_dictionary)
    if args.readROOT is True and args.readTESROOT is False:
        # If we saved the root file and want to load it do so here
        iv_dictionary = read_from_ivroot(outPath + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 't'])

    # Next we can process the IV curves to get Rn and Rp values. Once we have Rp we can obtain vTES and go onward
    if args.readTESROOT is False:
        iv_dictionary, fit_parameters_dictionary, parasitic_dictionary = process_iv_curves(outPath, args.dataChannel, iv_dictionary)
        iv_dictionary = get_TES_values(outPath, args.dataChannel, iv_dictionary, parasitic_dictionary)
        save_to_root(outPath, iv_dictionary)
        print('Obtained TES values')
    if args.readTESROOT is True:
        iv_dictionary = read_from_ivroot(outPath + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 't', 'iTES', 'iTES_rms', 'vTES', 'vTES_rms', 'rTES', 'rTES_rms', 'pTES', 'pTES_rms'])
        # Note: We would need to also save or re-generate the fit_parameters dictionary?
    
    # This step onwards assumes iv_dictionary contains TES values
    iv_dictionary, fit_dictionary = process_tes_curves(outPath, args.dataChannel, iv_dictionary)
    # Make TES Plots
    #make_tes_plots(output_path=outPath, data_channel=args.dataChannel, iv_dictionary=iv_dictionary, fit_dictionary=fit_dictionary)
    
    # Next let's do some special processing...R vs T, P vs T type of thing
    get_RT_curves(outPath, args.dataChannel, iv_dictionary)
    get_PT_curves(outPath, args.dataChannel, iv_dictionary)
    print('done')