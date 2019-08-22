'''Module to convert signal express files into root files'''

from os.path import isabs
from os.path import basename
from os import makedirs
import argparse
import glob
import re
import multiprocessing

import numpy as np
import pandas as pan
from joblib import Parallel, delayed
from writeROOT import writeROOT as write_root


def mkdpaths(dirpath):
    '''Function to make a directory path if it is not present'''
    makedirs(dirpath, exist_ok=True)
    return True


def load_signal_express_file(fname, sample_duration):
    '''
    Pandas allows us to open the file and correctly parse the file by padding with nan
    '''
    # Read the file. The first row is header information. The first column should be 'Time'
    # Subsequent columns are the channel information. Specifically they are at the ai# line.
    unix_offset = -2082844800  # Time from start of labview time relative to start of unix time (in seconds)
    data = pan.read_csv(fname, delimiter='\t')
    headers = data.columns
    branches = []
    # TODO: header may not be ai# format, could be text such as Voltage - Vin
    pat = r'ai\d'
    for header in headers:
        if header.lower().find('vin') > -1:
            header = 'ai0'
        if header.lower().find('vout') > -1:
            header = 'ai1'
        match = re.search(pat, header)
        match = match.group(0) if match else None
        branches.append(match if match is not None else header)
    # Next we must convert the data to an array.
    # ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s'] + ['Waveform' + '{:03d}'.format(int(i)) for i in sData['Channels']]
    sample_rate = data.Time.size / sample_duration
    waveform_duration = 1
    num_entries = int(waveform_duration * sample_duration)
    waveform_size = int(waveform_duration * sample_rate)
    data_dictionary = {}
    # print('Converting to data dictionary')
    for header in headers:
        # TODO: FIX THIS SO IT WORKS DAMNIT. EVERY 1 SECOND IS AN ENTRY.
        event_data = data[header]
        if header == 'Time':
            event_data += unix_offset
            # Split into second and microsecond part
            timestamp_s = np.array(np.floor(event_data), dtype=int)
            timestamp_mus = np.array(event_data*1e6 - timestamp_s*1e6, dtype=int)
            # These are the timestamp info for every single sample. But I want only the ones at the start of entries
            entry_timestamp_s = np.zeros(num_entries)
            entry_timestamp_mus = np.zeros(num_entries)
            for entry in range(num_entries):
                entry_timestamp_s[entry] = timestamp_s[entry*waveform_size]
                entry_timestamp_mus[entry] = timestamp_mus[entry*waveform_size]
            data_dictionary['Timestamp_s'] = entry_timestamp_s
            data_dictionary['Timestamp_mus'] = entry_timestamp_mus
        else:
            # A channel. Get the channel number
            # Note: Here we must make num_entries arrays containing waveform_duration * sample_rate sized arrays
            if header.lower().find('vin') > -1:
                header = 'ai0'
            if header.lower().find('vout') > -1:
                header = 'ai1'
            match = re.search(pat, header)
            match = match.group(0)
            channel = int(match.strip('ai'))
            data_dictionary['Waveform' + '{:03d}'.format(channel)] = {}
            for entry in range(num_entries):
                waveform = np.zeros(waveform_size)
                for sample_index in range(waveform_size):
                    waveform[sample_index] = event_data[entry*waveform_size + sample_index]
                data_dictionary['Waveform' + '{:03d}'.format(channel)][entry] = waveform
    # Now the data from the file is in the data_dictionary. Add in the extras we need to manually create
    data_dictionary['NumberOfSamples'] = np.zeros(num_entries) + waveform_size
    data_dictionary['SamplingWidth_s'] = np.zeros(num_entries) + 1/sample_rate
    return data_dictionary


def write_to_root(output_file, data_dictionary):
    '''Format and write the data dictionary into a root file'''
    root_dict = {'TTree': {'data_tree': {'TBranch': {}}}}
    # The keys of the data_dictionary are the branch names
    for key, value in data_dictionary.items():
        root_dict['TTree']['data_tree']['TBranch'][key] = value
    # Add in the ChList Tvector
    chArray = [int(st.split('Waveform')[1]) for st in data_dictionary.keys() if st.startswith('Waveform')]
    root_dict['TVectorT'] = {'ChList': np.array(chArray)}
    write_root(output_file, root_dict)
    return True


def logfile_converter(output_directory, logfile, sample_duration):
    '''The actual function that converts a particular logfile into a root file
    We should avoid putting all this into a for loop so we can parallelize it perhaps
    '''
    data_dictionary = load_signal_express_file(logfile, sample_duration)
    output_file = basename(logfile)
    output_file = output_file.split('.')[0]
    output_file = output_directory + '/' + output_file + '.root'
    print('Passing data to root file {} for writing...'.format(output_file))
    write_to_root(output_file, data_dictionary)
    return True


def convert_logfile(input_directory, output_directory, run_number, sample_duration, use_parallel=False):
    '''Main function to convert a signal express logfile into a ROOT file of the format used in the PXIDAQ'''
    print('run number {}'.format(run_number))
    # list_of_files = glob.glob('{}/*{}*.txt'.format(inputDirectory, runNumber))
    list_of_files = glob.glob('{}/*.txt'.format(input_directory))
    print('After gobbing, the number of files is {}'.format(len(list_of_files)))
    # NATURAL SORT
    dre = re.compile(r'(\d+)')
    list_of_files.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in re.split(dre, l)])
    print('The list of files after sorting is: {}'.format(list_of_files))
    print('The size of the file list is: {}'.format(len(list_of_files)))
    if use_parallel is False:
        print('Performing conversions serially')
        results = []
        for logfile in list_of_files:
            print('Converting file {}'.format(logfile))
            result = logfile_converter(output_directory, logfile, sample_duration)
            results.append(result)
    else:
        # Attempt at using joblib
        print('Performing conversions in parallel')
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(logfile_converter)(output_directory, logfile, sample_duration) for logfile in list_of_files)
    if np.all(results):
        if len(results) == len(list_of_files):
            print('All files converted')
        else:
            print('Every file that was executed was converted, but not all files were recorded...')
    else:
        if len(results) == len(list_of_files):
            print('All files have a record in the results array but not all of these files were actually converted')
        else:
            print('Not all files have a record and of those that were, not all were converted')
    return True


def get_args():
    '''Function to get and parse input arguments when calling module'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inputDirectory',
                        help='Specify the full path of the directory containing the log files to convert')
    parser.add_argument('-o', '--outputDirectory',
                        help='Specify output directory. If not a full path, it will be output in the same directory as the input directory')
    parser.add_argument('-s', '--sample_duration', type=float,
                        help='Specify the duration (in seconds) that a file lasts')
    parser.add_argument('-r', '--runNumber',
                        help='Specify the run number in the log file to convert')
    parser.add_argument('-p', '--useParallel', action='store_true',
                        help='If flag is set use parallel dispatcher to process files as opposed to performing conversion serially')
    args = parser.parse_args()
    if not isabs(args.outputDirectory):
        args.outputDirectory = args.inputDirectory
    if not mkdpaths(args.outputDirectory):
        raise Exception('Could not make output directory {}'.format(args.outputDirectory))
    return args


if __name__ == '__main__':
    ARGS = get_args()
    convert_logfile(ARGS.inputDirectory, ARGS.outputDirectory, ARGS.runNumber, ARGS.sample_duration, ARGS.useParallel)
    print('All done')
