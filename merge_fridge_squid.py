from os.path import isabs
from os.path import dirname
from os.path import basename
import glob
import socket
import getpass
import argparse
import re
import datetime
import time

import ROOT as rt
import numpy as np
import pandas as pan

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from readROOT import readROOT
from writeROOT import writeROOT as wR

import matplotlib as mp
from matplotlib import pyplot as plt


def gen_plot(x, y, xlab, ylab, title, fName, log='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(8, 6))
    ax = fig2.add_subplot(111)
    ax.plot(x, y)
    ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    fig2.savefig(fName, dpi=100)
    plt.close('all')

def timeParse(tString, tPatr='%Y-%m-%d_%H-%M-%S'):
    '''Function to parse local time into tzone free UTC'''
    print(tString)
    tUTC = datetime.datetime.strptime(tString, tPatr).timetuple()
    
    return time.mktime(tUTC)


def getNTdata(fileName):
    '''Function that returns noise thermometer info'''
    
    # Data stored within the file is given as deltaT from when the file was initialized
    # So we have to parse the file name and convert to unix time.
    pat = '(?:\w*\-)+\d\d'
    match = re.search(pat, fileName)
    match = match.group(0) if match else None
    
    # Our date is a LOCAL date, so we should not act as if it is already UTC
    tPatr = '%Y-%m-%d_%H-%M-%S'
    tZero = timeParse(match, tPatr)
    
    # Now let's obtain the data from the file itself.
    tData = tmfParser(fileName)
    tData['time'] = tData['time'] + tZero
    
    # If the file does contain an interrupt that is probably the real tZero
    
    cNan = ~np.isnan(tData['T'])
    if ~np.all(cNan):
        # We have at least one NaN present
        iTest = cNan.argmin()
        tData['time'] = tData['time'] - tZero # revert back so undo tZero addition.
        delta = tData['time'][iTest] - tData['time'][iTest-1] if iTest > 0 else tData['time'][iTest] - tData['time'][iTest+1]
        #print('delta is {0}'.format(delta))
        tData['time'] = tData['time'] + delta
    return tData


def tmfParser(fileName):
    '''Pandas allows us to open the file and correctly parse the file by padding with nan'''
    
    # First parse the filename and get the new fileName
    
    fileName = tmfPretty(fileName)
    
    # Read the first line and see what we have
    fR = pan.read_csv(fileName, delimiter='\t', header=None, nrows=1)
    if fR.shape[1] == 1:
        # First line is 1 entry, a date. Read rest of lines and prepend this to the start.
        nT = pan.read_csv(fileName, delimiter='\t', header=None, skiprows=1)
        nT = {'time': np.append(fR.values[:,0], nT.values[:,0]), 'T': np.append(np.NaN, nT.values[:,1]), 'dT': np.append(np.NaN, nT.values[:,2])}
    else:
        # First row is data so go on
        nT = pan.read_csv(fileName, delimiter='\t', header=None)
        nT = {'time': nT.values[:,0], 'T': nT.values[:,1], 'dT': nT.values[:,2]}

    return nT

def tmfPretty(fileName):
    '''Load a tmf file and locate any untoward time strings in it'''
    
    tPatr='%Y-%m-%d_%H-%M-%S'
    patr = '(?:\w*\-)+\d\d'
    with open(fileName, 'r') as f:
        lines = f.readlines()
    
    # Next scan through the lines to see if there are any matches
    for linenum in range(len(lines)):
        match = re.match(patr, lines[linenum])
        
        if match:
            newLine = timeParse(match.group(0), tPatr)
            newLine = str(newLine) + '\n'
            lines[linenum] = newLine
        
    with open(fileName + '.parse', 'w') as f:
        for linenum in lines:
            f.write(linenum)

    return fileName + '.parse'


def timeInterp(nTime, nTemp, rootTime, interpDegree):
    '''Function to do an interpolation of time and temperature values'''
    if interpDegree in ['1','2','3']:
        interpDegree = ['linear','quadratic','cubic'][int(interpDegree)-1]
        print(interpDegree)
    intT = interp1d(nTime, nTemp, kind=interpDegree, bounds_error=False)
    new_T = intT(rootTime)
    return new_T
    

def timeSpline(nTime, nTemp, rootTime, interpDegree):
    '''Function to do a spline interpolation of time and temperature values'''
    intT = InterpolatedUnivariateSpline(nTime, nTemp, k=interpDegree)
    
    new_T = intT(rootTime)
    return new_T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--inputSQUIDFile', help='Specify the full path to the directory with the SQUID root files you wish to merge with fridge data')
    parser.add_argument('-r', '--squidrun', help='Specify the SQUID run number you wish to grab files from')
    parser.add_argument('-f', '--inputFridgeFile', help='Specify the full path of the Fridge root file you wish to merge with SQUID data')
    parser.add_argument('-n', '--inputNTFile', default='', help='Specify the full path of the Noise Thermometer TMF file you wish to merge with SQUID data')
    parser.add_argument('-o', '--outputFile', help='Specify output root file. If not a full path, it will be output in the same directory as the input SQUID file')    
    parser.add_argument('-i', '--interpType', default='spline', help='Specify interpolation method: "interp" for standard interpolation and "spline" for interpolated univariate spline. Default is "spline"')
    parser.add_argument('-d', '--interpDegree', default=3, help='Specify the interpolation degree (default 3 for cubic)')
    parser.add_argument('-c', '--newFormat', action='store_true', help='Specify whether or not to process with new file format')
    args = parser.parse_args()
    
    inSQUIDFile = args.inputSQUIDFile
    squidRun = args.squidrun
    inFridgeFile = args.inputFridgeFile
    inNTFile = args.inputNTFile
    outFile = args.outputFile
    interpType = args.interpType
    interpDegree = args.interpDegree
    if not isabs(outFile):
        outFile = dirname(inSQUIDFile) + '/' + outFile
    # Load desired Fridge data
    fTree = 'FridgeLogs'
    fBranches = ['Time_secs', 'EPCal_t_s', 'EPCal_T_K']
    fData = readROOT(inFridgeFile, fTree, fBranches, 'single')
    # We also need to obtain the UnixTimestamps from the SQUID ROOT files.
    # This is tricky because they must be chained together
    lof = glob.glob('{}/*{}*.root'.format(inSQUIDFile, squidRun))
    #lof = glob.glob('/Users/bwelliver/cuore/bolord/squid/*{0}*.root'.format(run))
    if args.newFormat is False:
        tree = 'data_tree'
        branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform']
        method = 'chain'
        sData = readROOT(lof, tree, branches, method)
    else:
        chlist = 'ChList'
        channels = readROOT(lof[0], None, None, method='single', tobject=chlist)
        channels = channels['data'][chlist]
        tree = 'data_tree'
        branches = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s'] + ['Waveform' + '{:03d}'.format(int(i)) for i in channels]
        method = 'chain'
        sData = readROOT(lof, tree, branches, method)
    # Now make life easier
    fData = fData['data']
    sData = sData['data']
    if args.newFormat is True:
        sData['Channels'] = channels
    print('The first entry in Waveform005 is: {}'.format(sData['Waveform005'][0]))
    # OK now get the SQUID event timestamp vector...this is annoyingly tricky.
    # The following will represent the time stamp of the FIRST entry in the waveform vector
    # Individual samples inside a waveform
    print(len(sData['Waveform'][0])) if args.newFormat is False else print(len(sData['Waveform' + '{:03d}'.format(int(sData['Channels'][0]))]))
    # will have actual timestamps of 'Timestamp_s + Timestamp_mus/1e6 + SamplingWidth_s'
    sTunix = sData['Timestamp_s'] + sData['Timestamp_mus']/1e6
    print('First sTunix is {}'.format(sTunix[0]))
    print('First fTunix is {}'.format(fData['EPCal_t_s'][0]))
    # Now interpolate fridge data to align with these start times...generally we will not worry about
    # inter-waveform alignment
    # These are EPCal data that are above 0K, not missing Temp data, and that occur within the SQUID timestamps.
    # Warning! timestamps are duplicated N(unique channels) times because of the way entries vs events are recorded
    uTunix, idx, rdx = np.unique(sTunix, return_index=True, return_inverse=True)
    fBranch = 'EPCal_T_K'
    # During times the sensor is off the time is duplicated
    fBranch_t = 'EPCal_t_s'
    print('Original sizes: {}, {}'.format(sTunix.size, fData[fBranch].size))
    # get unique fridge times and keys to get same events
    uTemp_t, idxT, rdxT = np.unique(fData[fBranch_t], return_index=True, return_inverse=True)
    uTemp = fData[fBranch][idxT]
    # First get only valid temperatures
    cValidT = np.logical_and(uTemp > 0, ~np.isnan(uTemp))
    # Next select temperatures that occur during SQUID data
    cTSQUID = np.logical_and(uTemp_t >= uTunix[0], uTemp_t <= uTunix[-1])
    # Interpolation will only work also if the new values are not outside of what data is there
    cNF = np.logical_and(uTunix >= uTemp_t[cValidT][0], uTunix <= uTemp_t[cValidT][-1])
    print('Number of useable values inside squid data is {}'.format(np.sum(cNF)))
    cUseT = np.logical_and(cValidT, cTSQUID)
    print('Number to use from fridge is {}'.format(np.sum(cUseT)))
    # Fill everything else with Nan?
    new_T = np.zeros(uTunix.size) - 1
    
    gen_plot(uTemp_t[cUseT], uTemp[cUseT], 'Unix Time', 'EPCal Temp', 'EPCal Temp vs Time', 'epc_vs_time', 'linear')
    # Interpolate power over the uniqued timestamps
    if interpType == 'interp':
        new_T[cNF] = timeInterp(uTemp_t[cUseT], uTemp[cUseT], uTunix[cNF], interpDegree)
    elif interpType == 'spline':
        new_T[cNF] = timeSpline(uTemp_t[cUseT], uTemp[cUseT], uTunix[cNF], interpDegree)
    print('New size is {0} and root size is {1}'.format(new_T.size, sTunix.size))
    #gen_plot(nTunix[branch][cNF], new_spline_Terr[cNF], 'Unix Time', 'Spline Temp', 'Spline Temp vs Time', 'splineT_vs_time', 'linear')
    gen_plot(uTunix[cNF], new_T[cNF], 'Unix Time', 'Interp EPCal Temp', 'Interp Temp vs Time', 'interpT_vs_time', 'linear')
    # Now re-expand to full size to make duplicates based on unfolding the unique time grid
    new_T = new_T[rdx]
    
    # Now we can also try to add in NT data if so desired
    # nt_data contains 3 keys: time, T, dT
    if inNTFile != '':
        nt_data = getNTdata(inNTFile)
        print('First SQUID Unix Time is {}'.format(sTunix[0]))
        print('First NT Unix Time is {}'.format(nt_data['time'][0]))

        # Now interpolate NT data to align with these start times...generally we will not worry about
        # inter-waveform alignment
        # These are NT data that are above 0K, not missing Temp data, and that occur within the SQUID timestamps.
        # Warning! timestamps are duplicated N(unique channels) times because of the way entries vs events are recorded
        uTunix, idx, rdx = np.unique(sTunix, return_index=True, return_inverse=True)
        fBranch = 'T'
        # During times the sensor is off the time is duplicated
        fBranch_t = 'time'
        print('Original sizes: {}, {}'.format(sTunix.size, nt_data[fBranch].size))
        # get unique fridge times and keys to get same events
        uTemp_t, idxT, rdxT = np.unique(nt_data[fBranch_t], return_index=True, return_inverse=True)
        uTemp = nt_data[fBranch][idxT]
        # First get only valid temperatures
        cValidT = np.logical_and(uTemp > 0, ~np.isnan(uTemp))
        # Next select temperatures that occur during SQUID data
        cTSQUID = np.logical_and(uTemp_t >= uTunix[0], uTemp_t <= uTunix[-1])
        # Interpolation will only work also if the new values are not outside of what data is there
        cNF = np.logical_and(uTunix >= uTemp_t[cValidT][0], uTunix <= uTemp_t[cValidT][-1])
        print('Number of useable values inside squid data is {}'.format(np.sum(cNF)))
        cUseT = np.logical_and(cValidT, cTSQUID)
        print('Number to use from NT is {}'.format(np.sum(cUseT)))
        # Fill everything else with Nan?
        new_NT = np.zeros(uTunix.size) - 1
        gen_plot(uTemp_t[cUseT], uTemp[cUseT], 'Unix Time', 'NT Temp', 'NT Temp vs Time', 'nt_vs_time', 'linear')
        # Interpolate power over the uniqued timestamps
        if interpType == 'interp':
            new_NT[cNF] = timeInterp(uTemp_t[cUseT], uTemp[cUseT], uTunix[cNF], interpDegree)
        elif interpType == 'spline':
            new_NT[cNF] = timeSpline(uTemp_t[cUseT], uTemp[cUseT], uTunix[cNF], interpDegree)
        print('New NT size is {0} and root size is {1}'.format(new_NT.size, sTunix.size))
        #gen_plot(nTunix[branch][cNF], new_spline_Terr[cNF], 'Unix Time', 'Spline Temp', 'Spline Temp vs Time', 'splineT_vs_time', 'linear')
        gen_plot(uTunix[cNF], new_NT[cNF], 'Unix Time', 'Interp NT Temp', 'Interp NT Temp vs Time', 'interpNT_vs_time', 'linear')
        # Now re-expand to full size to make duplicates based on unfolding the unique time grid
        new_NT = new_NT[rdx]
    
    
    #gen_plot(nTunix[branch][cNF], new_Terr[cNF] - new_spline_Terr[cNF], 'Unix Time', 'delta Temp', 'delta Temp vs Time', 'deltaT_vs_time', 'linear')
    # Now comes the "fun" part...write a new ROOT file that contains everything
    # Create dictionary with correct format
    rootDict = {'TTree': {'data_tree': {'TBranch': {} } } }
    if args.newFormat is False:
        squidNames = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform']
        # Waveform is a bit tricky...originally it is a std::vector<double>. So if I load in entry 0 that will give us
        # these branches for some channel with some values. entry 1 will be for a different value but same time and so on
        # until entry N (where N = unique(channels)). This collection of entries can be considered an "event".
        # readROOT will still return array[entry] = data but here now type(data) == array too.
        # make a diagnostic output plot
        for branch in squidNames:
            rootDict['TTree']['data_tree']['TBranch'][branch] = sData[branch]
        rootDict['TTree']['data_tree']['TBranch']['EPCal_K'] = new_T
        if inNTFile != '':
            rootDict['TTree']['data_tree']['TBranch']['NT'] = new_NT
    else:
        squidNames = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s'] + ['Waveform' + '{:03d}'.format(int(i)) for i in sData['Channels']]
        for branch in squidNames:
            rootDict['TTree']['data_tree']['TBranch'][branch] = sData[branch]
        rootDict['TTree']['data_tree']['TBranch']['EPCal_K'] = new_T
        if inNTFile != '':
            rootDict['TTree']['data_tree']['TBranch']['NT'] = new_NT
        # How do we write the single ChList TVectorT<double> ?
        rootDict['TVectorT'] = {'ChList': sData['Channels']}
    result = wR(outFile, rootDict)
    if result:
        if inNTFile != '':
            print("Noise thermometer and EPCal data have been interpolated and added to SQUID data")
        else:
            print('EPCal data has been interpolated and added to SQUID data.')

