"""Module to convert csv from Matlab denoise files into root files."""

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
    """Make a directory path if it is not present."""
    makedirs(dirpath, exist_ok=True)
    return True


def load_data_file(fname, sample_rate, tz_offset, t0=0, header_names=None, delimiter=',', waveform_duration=1):
    """Pandas allows us to open the file and correctly parse the file by padding with nan.

    Expect that the format of the data is simply N columns where each column is a set of samples for a
    single channel. Time information is not present.
    """
    # Read the file. The first row is header information. The first column should be 'Time'
    # Subsequent columns are the channel information. Specifically they are at the ai# line.
    tz_correction = tz_offset * 3600  # The timezone correction to apply
    unix_offset = -2082844800  # Time from start of labview time relative to start of unix time (in seconds)
    time_correction = unix_offset + tz_correction
    if header_names is None:
        header_names = ['ai0', 'ai1']
    data = pan.read_csv(fname, delimiter=delimiter, names=header_names)
    headers = data.columns
    # print('The headers are: {}'.format(headers))
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
    # data_duration: how many seconds long is the total amount of data in our file
    data_duration = data[headers[0]].size / sample_rate
    # num_entries: how many 'events' of waveform_duration seconds are present in the data file
    # waveform_size: how many samples comprise 1 waveform
    num_entries = int(waveform_duration * data_duration)
    waveform_size = int(waveform_duration * sample_rate)
    # print('The num of entries and waveform size are {} and {}'.format(num_entries, waveform_size))
    data_dictionary = {}
    # Next we need to start doing our conversions. We don't have a timestamp column so use t0 and bootstrap a time vector
    # data file stores the time per event, split into seconds + microseconds. We will need to compute starting from t0 each event timestamp
    start_seconds = int(t0)
    start_microseconds = int((t0 - start_seconds)*1e6)
    entry_timestamp_s =  np.zeros(num_entries)
    entry_timestamp_mus = np.zeros(num_entries)
    entry_timestamp_s[0] = start_seconds
    entry_timestamp_mus[0] = start_microseconds
    for entry in range(1, num_entries):
        t1 = entry_timestamp_s[entry-1] + entry_timestamp_mus[entry-1]/1e6 + entry*waveform_duration
        entry_timestamp_s[entry] = int(t1)
        entry_timestamp_mus[entry] = int((t1 - entry_timestamp_s[entry])*1e6)
    data_dictionary['Timestamp_s'] = entry_timestamp_s
    data_dictionary['Timestamp_mus'] = entry_timestamp_mus
    # Now run through actual channel data
    for header in headers:
        event_data = data[header]
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
    # print('The number of entries is: {}'.format(num_entries))
    data_dictionary['NumberOfSamples'] = np.zeros(num_entries) + waveform_size
    data_dictionary['SamplingWidth_s'] = np.zeros(num_entries) + 1/sample_rate
    return data_dictionary


def write_to_root(output_file, data_dictionary):
    """Format and write the data dictionary into a root file."""
    root_dict = {'TTree': {'data_tree': {'TBranch': {}}}}
    # The keys of the data_dictionary are the branch names
    for key, value in data_dictionary.items():
        root_dict['TTree']['data_tree']['TBranch'][key] = value
    # Add in the ChList Tvector
    chArray = [int(st.split('Waveform')[1]) for st in data_dictionary.keys() if st.startswith('Waveform')]
    root_dict['TVectorT'] = {'ChList': np.array(chArray)}
    write_root(output_file, root_dict)
    return True


def logfile_converter(output_directory, logfile, sample_rate, tz_offset, t0=0, header_names=None, delimiter=',', waveform_duration=1):
    """Convert a particular logfile into a root file.

    We should avoid putting all this into a for loop so we can parallelize it perhaps.
    """
    data_dictionary = load_data_file(logfile, sample_rate, tz_offset, t0=t0, header_names=header_names, delimiter=delimiter, waveform_duration=waveform_duration)
    output_file = basename(logfile)
    output_file = output_file.split('.')[0]
    output_file = output_directory + '/' + output_file + '.root'
    print('Passing data to root file {} for writing...'.format(output_file))
    write_to_root(output_file, data_dictionary)
    return True


def convert_logfile(input_directory, output_directory, run_number, sample_rate, tz_offset, t0=0, header_names=None, delimiter=',', waveform_duration=1, use_parallel=False):
    """Convert a signal express logfile into a ROOT file of the format used in the PXIDAQ."""

    print('run number {}'.format(run_number))
    # list_of_files = glob.glob('{}/*{}*.txt'.format(inputDirectory, runNumber))
    list_of_text_files = glob.glob('{}/*.txt'.format(input_directory))
    list_of_csv_files = glob.glob('{}/*.csv'.format(input_directory))
    list_of_files = [*list_of_text_files, *list_of_csv_files]
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
            result = logfile_converter(output_directory, logfile, sample_rate, tz_offset, t0=t0, header_names=header_names, delimiter=delimiter, waveform_duration=waveform_duration)
            results.append(result)
    else:
        # Attempt at using joblib
        print('Performing conversions in parallel')
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(logfile_converter)(output_directory, logfile, sample_rate, tz_offset, t0=t0, header_names=header_names, delimiter=delimiter, waveform_duration=waveform_duration) for logfile in list_of_files)
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
    """Get and parse input arguments when calling module."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inputDirectory',
                        help='Specify the full path of the directory containing the files to convert')
    parser.add_argument('-o', '--outputDirectory',
                        help='Specify output directory. If not a full path, it will be output in the same directory as the input directory')
    parser.add_argument('-R', '--sample_rate', type=float,
                        help='Specify the sample rate (in Hz) of data in the file')
    parser.add_argument('-r', '--runNumber',
                        help='Specify the run number in the log file to convert')
    parser.add_argument('-H', '--header', type=str,
                        help='Specify data file headers as a single comma separated string ("ai0, ai2,ai1, ai3")')
    parser.add_argument('-t', '--t0', type=float,
                        help='Specify the timestamp the data starts at in unixformat')
    parser.add_argument('-w', '--waveform_duration', type=float,
                        help='Specify the duration in seconds that a single event waveform should last. Default is 1s')
    parser.add_argument('-z', '--tzOffset', default=0.0, type=float,
                        help='The number of hours of timezone offset to use.\
                        Default is 0 and assumes timestamps to convert are from the same timezone.\
                        If you need to convert to an earlier timezone use a negative number.')
    parser.add_argument('-p', '--useParallel', action='store_true',
                        help='If flag is set use parallel dispatcher to process files as opposed to performing conversion serially')
    args = parser.parse_args()
    if not isabs(args.outputDirectory):
        args.outputDirectory = args.inputDirectory
    if not mkdpaths(args.outputDirectory):
        raise Exception('Could not make output directory {}'.format(args.outputDirectory))
    if args.header is not None:
        args.header = args.header.split(',')
    return args


if __name__ == '__main__':
    ARGS = get_args()
    convert_logfile(ARGS.inputDirectory, ARGS.outputDirectory, ARGS.runNumber, ARGS.sample_rate, ARGS.tzOffset, t0=ARGS.t0, header_names=ARGS.header, waveform_duration=ARGS.waveform_duration, use_parallel=ARGS.useParallel)
    print('All done')
