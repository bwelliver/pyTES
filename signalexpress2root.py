from os.path import isabs
from os.path import dirname
from os.path import basename
import argparse
import struct
import glob
import re
import numpy as np
import pandas as pan

from writeROOT import writeROOT as write_root


def load_signal_express_file(fileName, sampleDuration):
    '''Pandas allows us to open the file and correctly parse the file by padding with nan'''        
    # Read the file. The first row is header information. The first column should be 'Time'
    # Subsequent columns are the channel information. Specifically they are at the ai# line.
    unix_offset = -2082844800 # Time from start of labview time relative to start of unix time (in seconds)
    data = pan.read_csv(fileName, delimiter='\t')
    headers = data.columns
    branchNames = []
    pat = 'ai\d'
    for header in headers:
        match = re.search(pat, header)
        match = match.group(0) if match else None
        branchNames.append(match if match is not None else header)
    # Next we must convert the data to an array.
    #['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s'] + ['Waveform' + '{:03d}'.format(int(i)) for i in sData['Channels']]
    sample_rate = data.Time.size / sampleDuration
    waveform_duration = 1
    nEntries = int(waveform_duration * sampleDuration)
    waveform_size = int(waveform_duration * sample_rate)
    data_dictionary = {}
    print('Converting to data dictionary')
    for header in headers:
        #TODO: FIX THIS SO IT WORKS DAMNIT. EVERY 1 SECOND IS AN ENTRY.
        event_data = data[header]
        if header == 'Time':
            event_data += unix_offset
            # Split into second and microsecond part
            timestamp_s = np.array(np.floor(event_data), dtype=int)
            timestamp_mus = np.array(event_data*1e6 - timestamp_s*1e6, dtype=int)
            # These are the timestamp info for every single sample. But I want only the ones at the start of entries
            entry_timestamp_s = np.zeros(nEntries)
            entry_timestamp_mus = np.zeros(nEntries)
            for entry in range(nEntries):
                entry_timestamp_s[entry] = timestamp_s[entry*waveform_size]
                entry_timestamp_mus[entry] = timestamp_mus[entry*waveform_size]
            data_dictionary['Timestamp_s'] = entry_timestamp_s
            data_dictionary['Timestamp_mus'] = entry_timestamp_mus
        else:
            # A channel. Get the channel number
            # Note: Here we must make nEntries arrays containing waveform_duration * sample_rate sized arrays
            match = re.search(pat, header)
            match = match.group(0)
            channel = int(match.strip('ai'))
            data_dictionary['Waveform' + '{:03d}'.format(channel)] = {}
            for entry in range(nEntries):
                waveform = np.zeros(waveform_size)
                for sample_index in range(waveform_size):
                    waveform[sample_index] = event_data[entry*waveform_size + sample_index]
                data_dictionary['Waveform' + '{:03d}'.format(channel)][entry] = waveform
    # Now the data from the file is in the data_dictionary. Add in the extras we need to manually create
    data_dictionary['NumberOfSamples'] = np.zeros(nEntries) + waveform_size
    data_dictionary['SamplingWidth_s'] = np.zeros(nEntries) + 1/sample_rate
    return data_dictionary


def write_to_root(output_file, data_dictionary):
    '''Format and write the data dictionary into a root file'''
    rootDict = {'TTree': {'data_tree': {'TBranch': {} } } }
    # The keys of the data_dictionary are the branch names
    for key, value in data_dictionary.items():
        rootDict['TTree']['data_tree']['TBranch'][key] = value
    # Add in the ChList Tvector
    rootDict['TVectorT'] = {'ChList': np.array([5,7])}
    write_root(output_file, rootDict)
    return True


def convert_logfile(inputDirectory, outputDirectory, runNumber, sampleDuration):
    '''Main function to convert a signal express logfile into a ROOT file of the format used in the PXIDAQ'''
    list_of_files = glob.glob('{}/*{}*.txt'.format(inputDirectory, runNumber))
    list_of_files.sort()
    for logfile in list_of_files:
        data_dictionary = load_signal_express_file(logfile, sampleDuration)
        output_file = basename(logfile)
        output_file = output_file.split('.')[0]
        output_file = outputDirectory + '/' + output_file + '.root'
        write_to_root(output_file, data_dictionary)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inputDirectory', help='Specify the full path of the directory containing the log files to convert')
    parser.add_argument('-o', '--outputDirectory', help='Specify output directory. If not a full path, it will be output in the same directory as the input directory')
    parser.add_argument('-s', '--sampleDuration', type=float, help='Specify the duration (in seconds) that a file lasts')
    parser.add_argument('-r', '--runNumber', help='Specify the run number in the log file to convert')
    args = parser.parse_args()
    inputDirectory = args.inputDirectory
    outputDirectory = args.outputDirectory
    if not isabs(outputDirectory):
        outputDirectory = inputDirectory    
    convert_logfile(inputDirectory, outputDirectory, args.runNumber, args.sampleDuration)
    print('All done')