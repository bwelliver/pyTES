'''Module to merge fridge ROOT file, SQUID data root file and optionally noise thermometer
data into a single ROOT file similar in structure to that of SQUID data
'''

from os.path import isabs
from os.path import dirname
import glob
import argparse
import re
import datetime
import time

import numpy as np
import pandas as pan

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from readROOT import readROOT
from writeROOT import writeROOT as wR
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

def gen_plot(xval, yval, xlab, ylab, title, fname, log='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(8, 6))
    ax = fig2.add_subplot(111)
    ax.plot(xval, yval)
    ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    fig2.savefig(fname, dpi=100)
    plt.close('all')
    return None


def gen_line(xval, yval, xlab, ylab, title, fname, log='log'):
    """Create generic plots that may be semilogx (default)"""
    fig2 = plt.figure(figsize=(32, 16))
    ax = fig2.add_subplot(111)
    ax.plot(xval, yval, marker='None', linewidth=1)
    ax.set_xscale(log)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    fig2.savefig(fname, dpi=100)
    plt.close('all')
    return None


def timeParse(tString, tPatr='%Y-%m-%d_%H-%M-%S'):
    '''Function to parse local time into tzone free UTC'''
    print(tString)
    tUTC = datetime.datetime.strptime(tString, tPatr).timetuple()
    return time.mktime(tUTC)


def getNTdata(fileName):
    '''Function that returns noise thermometer info'''

    # Data stored within the file is given as deltaT from when the file was initialized
    # So we have to parse the file name and convert to unix time.
    pat = r'(?:\w*\-)+\d\d'
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
        tData['time'] = tData['time'] - tZero  # revert back so undo tZero addition.
        delta = tData['time'][iTest] - tData['time'][iTest-1] if iTest > 0 else tData['time'][iTest] - tData['time'][iTest+1]
        # print('delta is {0}'.format(delta))
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
        nT = {'time': np.append(fR.values[:, 0], nT.values[:, 0]), 'T': np.append(np.NaN, nT.values[:, 1]), 'dT': np.append(np.NaN, nT.values[:, 2])}
    else:
        # First row is data so go on
        nT = pan.read_csv(fileName, delimiter='\t', header=None)
        nT = {'time': nT.values[:, 0], 'T': nT.values[:, 1], 'dT': nT.values[:, 2]}
    return nT


def tmfPretty(fileName):
    '''Load a tmf file and locate any untoward time strings in it'''

    tPatr = '%Y-%m-%d_%H-%M-%S'
    patr = r'(?:\w*\-)+\d\d'
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
    if interpDegree in ['1', '2', '3']:
        interpDegree = ['linear', 'quadratic', 'cubic'][int(interpDegree)-1]
        print(interpDegree)
    intT = interp1d(nTime, nTemp, kind=interpDegree, bounds_error=False)
    new_T = intT(rootTime)
    return new_T


def timeSpline(nTime, nTemp, rootTime, interpDegree):
    '''Function to do a spline interpolation of time and temperature values'''
    intT = InterpolatedUnivariateSpline(nTime, nTemp, k=interpDegree)
    new_T = intT(rootTime)
    return new_T


def get_fridge_data(inFridgeFile, thermometer='EP'):
    '''Function to load fridge root data'''
    fTree = 'FridgeLogs'
    if thermometer == 'EP':
        fBranches = ['Time_secs', 'EPCal_t_s', 'EPCal_T_K']
    elif thermometer == 'ExpRuOx':
        fBranches = ['Time_secs', 'ExpRuOx_Jig_t_s', 'ExpRuOx_Jig_T_K']
    fData = readROOT(inFridgeFile, fTree, fBranches, 'single')
    return fData


def get_squid_data(inSQUIDFile, newFormat):
    '''Function to load squid data'''
    # This is tricky because they must be chained together
    # lof = glob.glob('{}/*{}*.root'.format(inSQUIDFile, squidRun))
    # No run number
    lof = []
    for inDir in inSQUIDFile:
        lof += glob.glob('{}/*.root'.format(inDir))
    # NATURAL SORT
    print('Before sorting there are {} files to use'.format(len(lof)))
    dre = re.compile(r'(\d+)')
    lof.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in re.split(dre, l)])
    # lof = glob.glob('/Users/bwelliver/cuore/bolord/squid/*{0}*.root'.format(run))
    print('After sorting there are {} files to merge'.format(len(lof)))
    if newFormat is False:
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
        sData['data']['Channels'] = channels
        # Try a sanity check plot
#        n5 = np.empty(0)
#        n7 = np.empty(0)
#        index = np.empty(0)
#        print('starting append loop')
#        for ev in range(3000):
#            print('Performing append for event: {}'.format(ev))
#            n5 = np.append(n5, sData['data']['Waveform005'][ev])
#            n7 = np.append(n7, sData['data']['Waveform007'][ev])
#        index = [i for i in range(n7.size)]
#        print('Making plot...')
#        gen_line(index, n5, 'Index', 'vBias', 'vBias vs index', dirname(inSQUIDFile) + '/test_startLoad_vBias_sr' + str(squidRun) + '.png', log='linear')
#        gen_line(index, n7, 'Index', 'vOut', 'vOut vs index', dirname(inSQUIDFile) + '/test_startLoad_vOut_sr' + str(squidRun) + '.png', log='linear')
    return sData


def get_interpolated_temperature(fData, sTunix, interpType, interpDegree, sensor_keys):
    '''Interpolate the fridge temperature'''
    # Now interpolate fridge data to align with these start times...generally we will not worry about
    # inter-waveform alignment
    # These are EPCal data that are above 0K, not missing Temp data, and that occur within the SQUID timestamps.
    # Warning! timestamps are duplicated N(unique channels) times because of the way entries vs events are recorded
    uTunix, rdx = np.unique(sTunix, return_index=False, return_inverse=True)
    #fBranch = 'EPCal_T_K'
    fBranch = sensor_keys['temp']
    # During times the sensor is off the time is duplicated
    #fBranch_t = 'EPCal_t_s'
    fBranch_t = sensor_keys['time']
    print('Original sizes: {}, {}'.format(sTunix.size, fData[fBranch].size))
    # get unique fridge times and keys to get same events
    uTemp_t, idxT = np.unique(fData[fBranch_t], return_index=True, return_inverse=False)
    uTemp = fData[fBranch][idxT]
    # First get only valid temperatures
    # cValidT = np.logical_and(uTemp > 0, ~np.isnan(uTemp))
    cValidT = ~np.isnan(uTemp)
    # Next select temperatures that occur during SQUID data
    cTSQUID = np.logical_and(uTemp_t >= uTunix[0], uTemp_t <= uTunix[-1])
    # Interpolation will only work also if the new values are not outside of what data is there
    cNF = np.logical_and(uTunix >= uTemp_t[cValidT][0], uTunix <= uTemp_t[cValidT][-1])
    print('Number of useable values inside squid data is {}'.format(np.sum(cNF)))
    cUseT = np.logical_and(cValidT, cTSQUID)
    print('Number to use from fridge is {}'.format(np.sum(cUseT)))
    # Fill everything else with Nan?
    new_T = np.zeros(uTunix.size) - 1

    gen_plot(uTemp_t[cUseT], uTemp[cUseT], 'Unix Time', 'Temp', 'Temp vs Time', 'temp_vs_time', 'linear')
    # Interpolate power over the uniqued timestamps
    if interpType == 'interp':
        new_T[cNF] = timeInterp(uTemp_t[cUseT], uTemp[cUseT], uTunix[cNF], interpDegree)
    elif interpType == 'spline':
        new_T[cNF] = timeSpline(uTemp_t[cUseT], uTemp[cUseT], uTunix[cNF], interpDegree)
    print('New size is {0} and root size is {1}'.format(new_T.size, sTunix.size))
    # gen_plot(nTunix[branch][cNF], new_spline_Terr[cNF], 'Unix Time', 'Spline Temp', 'Spline Temp vs Time', 'splineT_vs_time', 'linear')
    gen_plot(uTunix[cNF], new_T[cNF], 'Unix Time', 'Interp Temp', 'Interp Temp vs Time', 'interpT_vs_time', 'linear')
    # Now re-expand to full size to make duplicates based on unfolding the unique time grid
    new_T = new_T[rdx]
    return new_T


def get_interpolated_noise_thermometer(nt_data, sTunix, interpType, interpDegree):
    '''Interpolate the noise thermometer data'''
    # Now interpolate NT data to align with these start times...generally we will not worry about
    # inter-waveform alignment
    # These are NT data that are above 0K, not missing Temp data, and that occur within the SQUID timestamps.
    # Warning! timestamps are duplicated N(unique channels) times because of the way entries vs events are recorded
    uTunix, rdx = np.unique(sTunix, return_index=False, return_inverse=True)
    fBranch = 'T'
    # During times the sensor is off the time is duplicated
    fBranch_t = 'time'
    print('Original sizes: {}, {}'.format(sTunix.size, nt_data[fBranch].size))
    # get unique fridge times and keys to get same events
    uTemp_t, idxT = np.unique(nt_data[fBranch_t], return_index=True, return_inverse=False)
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
    # gen_plot(nTunix[branch][cNF], new_spline_Terr[cNF], 'Unix Time', 'Spline Temp', 'Spline Temp vs Time', 'splineT_vs_time', 'linear')
    gen_plot(uTunix[cNF], new_NT[cNF], 'Unix Time', 'Interp NT Temp', 'Interp NT Temp vs Time', 'interpNT_vs_time', 'linear')
    # Now re-expand to full size to make duplicates based on unfolding the unique time grid
    new_NT = new_NT[rdx]
    return new_NT


def write_new_merged_file(outFile, sData, new_T, sensor_keys, new_NT=None, newFormat=False):
    # Now comes the "fun" part...write a new ROOT file that contains everything
    # Create dictionary with correct format
    rootDict = {'TTree': {'data_tree': {'TBranch': {}}}}
    if newFormat is False:
        squidNames = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform']
        # Waveform is a bit tricky...originally it is a std::vector<double>.
        # So if I load in entry 0 that will give us these branches for some channel with some values.
        # Entry 1 will be for a different value but same time and so on until entry N (where N = unique(channels)).
        # This collection of entries can be considered an "event".
        # readROOT will still return array[entry] = data but here now type(data) == array too.
        # make a diagnostic output plot
        for branch in squidNames:
            rootDict['TTree']['data_tree']['TBranch'][branch] = sData[branch]
        rootDict['TTree']['data_tree']['TBranch'][sensor_keys['branch']] = new_T
        if new_NT is not None:
            rootDict['TTree']['data_tree']['TBranch']['NT'] = new_NT
    else:
        # Make a test plot please
        # Convert sData to a single array
        # n = np.empty(0)
        # index = np.empty(0)
        # print('starting append loop')
        # for ev in range(1000):
        #     print('Performing append for event: {}'.format(ev))
        #     n = np.append(n, sData['Waveform005'][ev])
        # index = [i for i in range(n.size)]
        # print('Making plot...')
        # gen_line(index, n, 'Index', 'vBias', 'vBias vs index', dirname(inSQUIDFile) + '/test_EndLoad_vBias_sr' + str(squidRun) + '.png', log='linear')
        squidNames = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s'] + ['Waveform' + '{:03d}'.format(int(i)) for i in sData['Channels']]
        for branch in squidNames:
            rootDict['TTree']['data_tree']['TBranch'][branch] = sData[branch]
        rootDict['TTree']['data_tree']['TBranch'][sensor_keys['branch']] = new_T
        if new_NT is not None:
            rootDict['TTree']['data_tree']['TBranch']['NT'] = new_NT
        # How do we write the single ChList TVectorT<double> ?
        rootDict['TVectorT'] = {'ChList': sData['Channels']}
    result = wR(outFile, rootDict)
    return result


def merge_fridge_squid_data(inputSQUIDFile, outputFile, inputFridgeFile, squidrun, interpType='spline', interpDegree=3, inputNTFile='', newFormat=True, thermometer='EP'):
    '''Main function to merge data from fridge and squid files, along with possibly NT data'''
    # Load desired Fridge data
    if thermometer == 'EP':
        time_key = 'EPCal_t_s'
        temp_key = 'EPCal_T_K'
        branch = 'EPCal_K'
    elif thermometer == 'ExpRuOx':
        time_key = 'ExpRuOx_Jig_t_s'
        temp_key = 'ExpRuOx_Jig_T_K'
        branch = 'ExpRuOx_K'
    sensor_keys = {'time': time_key, 'temp': temp_key, 'branch': branch}
    fData = get_fridge_data(inputFridgeFile, thermometer)

    # Load SQUID Data
    # We also need to obtain the UnixTimestamps from the SQUID ROOT files.
    sData = get_squid_data(inputSQUIDFile, newFormat)
    # Now make life easier
    fData = fData['data']
    sData = sData['data']

    # OK now get the SQUID event timestamp vector...this is annoyingly tricky.
    # The following will represent the time stamp of the FIRST entry in the waveform vector
    # Individual samples inside a waveform
    if newFormat is False:
        print(len(sData['Waveform'][0]))
    else:
        print(len(sData['Waveform' + '{:03d}'.format(int(sData['Channels'][0]))]))
    # will have actual timestamps of 'Timestamp_s + Timestamp_mus/1e6 + SamplingWidth_s'
    sTunix = sData['Timestamp_s'] + sData['Timestamp_mus']/1e6
    print('First sTunix is {}'.format(sTunix[0]))
    print('First fTunix is {}'.format(fData[sensor_keys['time']][0]))

    new_T = get_interpolated_temperature(fData, sTunix, interpType, interpDegree, sensor_keys)

    # Now we can also try to add in NT data if so desired
    # nt_data contains 3 keys: time, T, dT
    new_NT = None
    if inputNTFile != '':
        nt_data = getNTdata(inputNTFile)
        print('First SQUID Unix Time is {}'.format(sTunix[0]))
        print('First NT Unix Time is {}'.format(nt_data['time'][0]))
        new_NT = get_interpolated_noise_thermometer(nt_data, sTunix, interpType, interpDegree)

    result = write_new_merged_file(outputFile, sData, new_T, sensor_keys, new_NT, newFormat)
    return result


def get_args():
    '''Function to get input arguments when module is called'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--inputSQUIDFile', action='append',
                        help='Specify the full path to the directory with the SQUID root files you wish to merge with fridge data')
    parser.add_argument('-r', '--squidrun',
                        help='Specify the SQUID run number you wish to grab files from')
    parser.add_argument('-f', '--inputFridgeFile',
                        help='Specify the full path of the Fridge root file you wish to merge with SQUID data')
    parser.add_argument('-n', '--inputNTFile', default='',
                        help='Specify the full path of the Noise Thermometer TMF file you wish to merge with SQUID data')
    parser.add_argument('-o', '--outputFile',
                        help='Specify output root file. If not a full path, it will be output in the same directory as the input SQUID file')
    parser.add_argument('-i', '--interpType', default='spline',
                        help='Specify interpolation method: "interp" for standard interpolation and "spline" for\
                        interpolated univariate spline. Default is "spline"')
    parser.add_argument('-d', '--interpDegree', default=3,
                        help='Specify the interpolation degree (default 3 for cubic)')
    parser.add_argument('-c', '--newFormat', action='store_true',
                        help='Specify whether or not to process with new file format')
    parser.add_argument('-T', '--thermometer', default='EP',
                        help='Specify whether to interpolate the EPCal sensor or the ExpRuOx')
    args = parser.parse_args()
    if not isabs(args.outputFile):
        args.outputFile = dirname(args.inputSQUIDFile) + '/' + args.outputFile
    return args


if __name__ == '__main__':
    ARGS = get_args()
    RESULT = merge_fridge_squid_data(ARGS.inputSQUIDFile, ARGS.outputFile, ARGS.inputFridgeFile,
                                     ARGS.squidrun, ARGS.interpType, ARGS.interpDegree, ARGS.inputNTFile, ARGS.newFormat, ARGS.thermometer)
    if RESULT:
        if ARGS.inputNTFile != '':
            print("Noise thermometer and RuOx data have been interpolated and added to SQUID data")
        else:
            print('RuOx data has been interpolated and added to SQUID data.')
