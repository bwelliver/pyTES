"""Convert fast digitizer files into the usual ROOT format for the analysis code."""
import struct
from os.path import isabs
from os.path import basename
from os import makedirs
import re
import argparse
import glob
import multiprocessing
import numpy as np
import deepdish as dd
from joblib import Parallel, delayed
from writeROOT import writeROOT as write_root


def mkdpaths(dirpath):
    """Make a directory path if it is not present."""
    makedirs(dirpath, exist_ok=True)
    return True


def read_filtered_file(hfile):
    """Parse a filtered AC bias file via deepdish."""
    # The format of the resulting output is basically some type of triggered events
    # It has the following keys: ch_info, data, header_info
    # ch_info -- specific information about the channel if needed
    # header_info -- specific information about the data, mainly window size and sample rate
    # data -- contains subkeys
    #   eventNumber -- contains key 1 (possibly channel number?)
    #       key 1 (ch?) -- header, data keys
    #           header: uncertain what this really is (time, ?, ch, size?)
    #           data: contains the actual waveform
    data_dictionary = dd.io.load(hfile)
    header_dict, ch_info = read_new_header_file(None, data_dictionary['header_info'])
    # data = {ch: {} for ch in ch_info.keys()}
    # for eventNumber, ch_dictionary in data_dictionary['data'].items():
    #    for channel, waveform in ch_dictionary.items():
    #        data[channel][eventNumber] = waveform['data']
    return header_dict, ch_info, data_dictionary['data']

def read_new_header_file(hfile, header_dictionary=None):
    """Parse new header file into a reference dictionary."""
    # New header is basically a tab separated key-value document.
    if header_dictionary is None:
        header_dict = {}
        with open(hfile, mode='r') as hf:
            lines = hf.readlines()
        for line in lines:
            parse = line.strip('\n').split('\t')
            header_dict[parse[0]] = parse[1]
    else:
        header_dict = header_dictionary
    # Let us also make a specific channel info dictionary
    ch_info = {}
    # keys in the dictionary are of the form CH# something
    seen_channel = -1
    for key, value in header_dict.items():
        if key.startswith("CH"):
            channel = int(key.split('CH')[1].split(' ')[0])
            if channel != seen_channel:
                keyprefix = 'CH{}'.format(channel)
                ch_info[channel] = {'range': float(header_dict['{} Vertical range'.format(keyprefix)]),
                                    'inputZ': float(header_dict['{} Input impedance'.format(keyprefix)]),
                                    'attenuation': float(header_dict['{} Probe attenuation'.format(keyprefix)])
                                    }
                seen_channel = channel
            else:
                continue
        else:
            continue
    # ok channel info should be ready
    return header_dict, ch_info


def parse_header_time(header_info, tz_offset, manual_tstart=None):
    """Convert header time into unix time."""
    tz_correction = tz_offset * 3600
    unix_offset = -2082844800
    time_correction = unix_offset + tz_correction
    if manual_tstart is None:
        header_info['Start time UNIX'] = float(header_info['Start time UNIX']) + time_correction
        header_info['Stop time UNIX'] = float(header_info['Stop time UNIX']) + time_correction
    else:
        header_info['Start time UNIX'] = manual_tstart + time_correction
        header_info['Stop time UNIX'] = manual_tstart + time_correction
    return header_info


def unroll_binary_event(ch_data, num_root_per_bin, sample_rate):
    """Unroll a single binary event into the appropriate number of ROOT events."""
    # For all channels everything except the Waveforms should be the same for a given ROOT event
    # ch_data[channel]['header'] = [time, gain, channel, nsamples]
    root_event = {}
    unix_offset = -2082844800
    for idx in range(num_root_per_bin):
        root_event[idx] = {}
    for channel, values in ch_data.items():
        wf_size = values['data'].size
        subsize = int(wf_size/num_root_per_bin)
        t0 = values['header'][0]
        if t0 < 0:
            t0 = t0 - unix_offset
        for idx in range(num_root_per_bin):
            wf_name = 'Waveform{:03d}'.format(channel)
            root_event[idx][wf_name] = values['data'][idx*subsize:(idx+1)*subsize]
            timestamp = int(np.floor(t0)) + (idx*subsize/sample_rate)
            timestamp_mu = int(t0*1e6 - int(np.floor(t0))*1e6)  # assume the same microsecond offset
            root_event[idx]['Timestamp_s'] = timestamp
            root_event[idx]['Timestamp_mus'] = timestamp_mu
    return root_event


def convert_to_root(header_info, ch_info, parsed_data, waveform_duration=None):
    """Convert the data into ROOT format now."""
    # Here we need to make 1 entry per second for the ROOT file and it needs to be such that
    # it contains all channel waveforms as need be.
    # parsed_data has as keys the binary entry number
    # For each parsed_data[key] we have a dictionary for each channel.
    # parsed_data[key][channel][data] will contain the actual data we want.
    # The goal here will be to get a dictionary whose key is a ROOT entry and whose values will be the branches
    # Each root entry must contain: Timestamp_s, Timestamp_mus, NumberOfSamples, SamplingWidth_s, and Waveform%03d(vector)
    # The data dictionary format is keys: Branch, values: nEntries arrays of what we want
    # The waveform one is itself a dictionary whose keys are the actual root entry

    nSamples = float(header_info['Acquired points'])
    sample_freq = float(header_info['Sampling Frequency'])
    sample_duration = nSamples/sample_freq  # This indicates how many seconds our data is and hence how many divisions to make
    waveform_duration = sample_duration if waveform_duration is None else waveform_duration
    waveform_size = int(waveform_duration * sample_freq)
    num_root_per_bin = int(sample_duration/waveform_duration)
    num_entries = len(parsed_data)*num_root_per_bin
    print('The number of bin entries is {} and the number of root entries then is: {}'.format(len(parsed_data), num_entries))
    data_dictionary = {'Timestamp_s': np.zeros(num_entries), 'Timestamp_mus': np.zeros(num_entries)}
    for channel in ch_info.keys():
        if channel not in parsed_data[0].keys():
            continue
        data_dictionary['Waveform{:03d}'.format(channel)] = {}
    for bin_entry, ch_dict in parsed_data.items():
        root_events = unroll_binary_event(ch_dict, num_root_per_bin, sample_freq)
        for entry, value in root_events.items():
            for subkey, subvalue in value.items():
                data_dictionary[subkey][num_root_per_bin*bin_entry + entry] = subvalue
                # if subkey in ['Timestamp_s', 'Timestamp_mus']:
                #     data_dictionary[subkey][num_root_per_bin*bin_entry + entry] = subvalue
                # else:
                #     data_dictionary[subkey][[num_root_per_bin*bin_entry + entry]] = subvalue
    # Add the last things manually
    data_dictionary['NumberOfSamples'] = np.zeros(num_entries) + waveform_size
    data_dictionary['SamplingWidth_s'] = np.zeros(num_entries) + 1/sample_freq
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


def hdf5_converter(output_directory, logfile, waveform_duration=None):
    """Full processing of an hdf5 file after denoising."""
    # Each file contains ch_info, header_info and data so we parse it all here
    header_info, ch_info, data = read_filtered_file(logfile)
    header_info = parse_header_time(header_info, tz_offset=0, manual_tstart=None)
    data_dictionary = convert_to_root(header_info, ch_info, data, waveform_duration)
    output_file = basename(logfile)
    output_file = output_file.split('.')[0]
    output_file = output_directory + '/' + output_file + '.root'
    print('Passing data to root file {} for writing...'.format(output_file))
    result = write_to_root(output_file, data_dictionary)
    return result


def process_digifile(input_directory, output_directory, run_number, tz_offset=0, waveform_duration=None, use_parallel=False):
    """Actually parse log files."""
    list_of_h5_files = glob.glob('{}/*.hdf5'.format(input_directory))
    list_of_files = [*list_of_h5_files]
    print('After globbing, the number of files is {}'.format(len(list_of_files)))
    # NATURAL SORT
    dre = re.compile(r'(\d+)')
    list_of_files.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in re.split(dre, l)])
    print('The list of files after sorting is: {}'.format(list_of_files))
    print('The size of the file list is: {}'.format(len(list_of_files)))
    # Now each h5 file contains a header_info key so just batch process each h5 file
    if use_parallel is False:
        print('Performing conversions serially')
        results = []
        for logfile in list_of_files:
            print('Converting file {}'.format(logfile))
            result = hdf5_converter(output_directory, logfile, waveform_duration)
            results.append(result)
    else:
        # Attempt at using joblib
        print('Performing conversions in parallel')
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(hdf5_converter)(output_directory, logfile, waveform_duration) for logfile in list_of_files)
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
                        help='Specify the full path of the directory containing the log files to convert')
    parser.add_argument('-o', '--outputDirectory',
                        help='Specify output directory. If not a full path, it will be output in the same directory as the input directory')
    parser.add_argument('-r', '--runNumber',
                        help='Specify the run number in the log file to convert')
    parser.add_argument('-w', '--waveformDuration', default=None, type=float,
                        help='Specify the duration (in seconds) a root waveform should be. Defaults to same size as the digitizer header specifies. \
                            Only values < digitizer binary file duration are supported.')
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
    return args


if __name__ == '__main__':
    ARGS = get_args()
    process_digifile(ARGS.inputDirectory, ARGS.outputDirectory, ARGS.runNumber, ARGS.tzOffset, ARGS.waveformDuration, ARGS.useParallel)
    print('All done')
