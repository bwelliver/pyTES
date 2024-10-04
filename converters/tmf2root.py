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

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from readROOT import readROOT
import cloneROOT as cr

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
    parser.add_argument('-r', '--inputROOTFile', help='Specify the full path of the ROOT file you wish to inject NT data into')
    parser.add_argument('-n', '--inputNTFile', help='Specify the full path of the Noise Thermometer file you wish to use')
    parser.add_argument('-o', '--outputFile', help='Specify output root file. If not a full path, it will be output in the same directory as the input file')    
    parser.add_argument('-i', '--interpType', default='spline', help='Specify interpolation method: "interp" for standard interpolation and "spline" for interpolated univariate spline. Default is "spline"')
    parser.add_argument('-d', '--interpDegree', default=3, help='Specify the interpolation degree (default 3 for cubic)')
    args = parser.parse_args()
    
    inROOTFile = args.inputROOTFile
    inNTFile = args.inputNTFile
    outFile = args.outputFile
    interpType = args.interpType
    interpDegree = args.interpDegree
    if not isabs(outFile):
        outFile = dirname(inROOTFile) + '/' + outFile
        
    # Get NT data into numpy arrays
    tData = getNTdata(inNTFile)
    #print(tData['time'])
    # We also need to obtain the UnixTimestamps from the ROOT file.
    #tree = 'BridgeLog'
    #branch = 'Time_sec'
    tree = 'btree'
    branch = 'MeasTime'
    nTunix = readROOT(inROOTFile, tree, branch, 'single')
    nTunix = nTunix['data']
    
    # Now let's run an interpolation over the NT data sampled at nTunix but only for data that is valid.
    # These are NT data that are above 0K, not missing Temp data, and that occur within the ROOT unixtimestamps.
    cValid = np.logical_and(tData['T'] > 0, ~np.isnan(tData['T']))
    #print(tData['time'])
    cROOT = np.logical_and(tData['time'] >= nTunix[branch][0], tData['time'] <= nTunix[branch][-1])
    # Interpolation will only work also if the new values are not outside of what data is there
    cNF = np.logical_and(nTunix[branch] >= tData['time'][cValid][0], nTunix[branch] <= tData['time'][cValid][-1])
    cUse = np.logical_and(cValid, cROOT)
    # Fill everything else with Nan?
    new_T = np.zeros(nTunix[branch].size) -1
    new_Terr = np.zeros(nTunix[branch].size) -1
    
    gen_plot(tData['time'][cUse], tData['T'][cUse], 'Unix Time', 'NT Temp', 'NT Temp vs Time', 'ntT_vs_time', 'linear')
    if interpType == 'interp':
        new_T[cNF] = timeInterp(tData['time'][cUse], tData['T'][cUse], nTunix[branch][cNF], interpDegree)
        new_Terr[cNF] = timeInterp(tData['time'][cUse], tData['dT'][cUse], nTunix[branch][cNF], interpDegree)
    elif interpType == 'spline':
        new_T[cNF] = timeSpline(tData['time'][cUse], tData['T'][cUse], nTunix[branch][cNF], interpDegree)
        new_Terr[cNF] = timeSpline(tData['time'][cUse], tData['dT'][cUse], nTunix[branch][cNF], interpDegree)
    
    print('New size is {0} and root size is {1}'.format(new_T.size, nTunix[branch].size))
    
    #gen_plot(nTunix[branch][cNF], new_spline_Terr[cNF], 'Unix Time', 'Spline Temp', 'Spline Temp vs Time', 'splineT_vs_time', 'linear')
    gen_plot(nTunix[branch][cNF], new_T[cNF], 'Unix Time', 'Interp Temp', 'INterp Temp vs Time', 'interpT_vs_time', 'linear')
    #gen_plot(nTunix[branch][cNF], new_Terr[cNF] - new_spline_Terr[cNF], 'Unix Time', 'delta Temp', 'delta Temp vs Time', 'deltaT_vs_time', 'linear')

    # Now create an injection dictionary and make the new ROOT file
    # Create file information dictionary
    fInfo = {'ProgName': rt.TNamed('ProgName', basename(__file__)), 'ProgVersion': rt.TNamed('ProgVersion','0.5.2'),
            'CreatedByUser': rt.TNamed('CreatedByUser', getpass.getuser()), 'BridgeFile': rt.TNamed('BridgeFile', basename(inROOTFile)), 'NTFile': rt.TNamed('NTFile', basename(inNTFile)), 'CreatedOnServer': rt.TNamed('CreatedOnServer', socket.gethostname()), 'TInterp': rt.TNamed('TInterp', interpType + ' ' + str(interpDegree))}
    injection = {'TTree': {tree: {'NTemp': new_T, 'NTemp_err': new_Terr}}, 'TDirectory': {'FileInfo': {'TObject': fInfo}}}
    cr.CopyFile_Inject(inROOTFile, outFile, injection)
    
    print('Noise thermometry data has been interpolated and added to the specified tree.')

