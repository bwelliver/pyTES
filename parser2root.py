"""Convert fast digitizer files into the usual ROOT format for the analysis code."""
import struct
from os.path import isabs
from os.path import basename
from os.path import getsize
from os.path import exists as opexists
from os import makedirs
import re
import argparse
import glob
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from array import array
import writeROOT as wr
import ROOT as rt


def mkdpaths(dirpath):
    """Make a directory path if it is not present."""
    makedirs(dirpath, exist_ok=True)
    return True


def all_bytes_from_file(filename):
    """Open and store entire binary file into memory."""
    with open(filename, mode='rb') as f:
        byteFile = f.read()
    return byteFile



def binfile_converter(output_directory, binary_files, sample_freq, run_number, event_duration=1, events_per_partial=50, scale_factor=0.00390625, endian='<'):
    """Processing a mux binary file to root"""
    # Since the files are just one long contiguous binary file
    # we cannot open them all at once into memory
    # rather we must proceed line by line.
    # Assumptions:
    # - All binary files are encoded as int32
    # - all binary files have the same sample rate
    
    # First get the file size of each binary file
    size_dictionary = {}
    minsize = np.inf
    io_dictionary = {}
    for ch,iqfiles in binary_files.items():
        size_dictionary[ch] = {'i': 0, 'q': 0}
        io_dictionary[ch] = {'i': None, 'q': None}
        for iq, file in iqfiles.items():
            fsize = getsize(file)
            size_dictionary[ch][iq] = fsize
            minsize = fsize if fsize < minsize else minsize
            io_dictionary[ch][iq] = open(file, mode='rb')
    #int32 = 4 bytes
    total_samples = minsize/8
    # now the tricky part
    # We have to get a waveform out of doing sqrt(i^2 + q^2) for each sample
    # The standard root format associates all channels with a given file
    # Now we have to build up waveforms for every damn thing
    # Because we have to grow things in a way writeROOT isn't able to handle yet, we form the ROOT stuff here.
    root_dict = {'TTree': {'data_tree': {'TBranch': {}}}}
    root_dict['TVectorT'] = {'ChList': np.array(chArray)}
    #['Timestamp_s', 'Timestamp_mus', 'NumberOfSamples']
    #data_dictionary['NumberOfSamples'] = np.zeros(num_entries) + waveform_size
    #data_dictionary['SamplingWidth_s'] = np.zeros(num_entries) + 1/sample_freq
    chArray = np.array(binary_files.keys())
    left_to_read = total_samples
    entry = 0
    output_name = output_directory + 'MUX_Run{:06d}'.format(run_number)
    partial = 0
    timestamp = 0
    timestamp_mus = 0
    partial_name = output_name + "_p{:06d}.root".format(partial)
    while left_to_read > 0:
        if entry%events_per_partial == 0:
            # new file
            partial += 1
            entry = 0
            partial_name = output_name + "_p{:06d}.root".format(partial)
            print("Creating partial: {}".format(partial_name))
            tfile = rt.TFile(partial_name, 'RECREATE')
            wr.writeTVectorT(chArray)
            tree = rt.TTree('data_tree', 'data_tree')
            AddressOf = getattr(rt, 'AddressOf')
            std = getattr(rt, 'std')
            dloc = {}
            # Assign branch addresses and names/types
            for ch in binary_files.keys():
                branchkey = 'Waveform{:03d}'.format(ch)
                dloc[branchkey] = std.vector('double')(sample_freq*event_duration)
                tree.Branch(branchkey, 'std::vector<double>', AddressOf(dloc[branchkey]))
            dloc['Timestamp_s'] = array('I', [0])
            tree.Branch('Timestamp_s', dloc['Timestamp_s'], 'Timestamp_s' + '/i')
            dloc['Timestamp_mus'] = array('I', [0])
            tree.Branch('Timestamp_mus', dloc['Timestamp_mus'], 'Timestamp_mus' + '/i')
            dloc['NumberOfSamples'] = array('I', [0])
            tree.Branch('NumberOfSamples', dloc['NumberOfSamples'], 'NumberOfSamples' + '/i')
            dloc['SamplingWidth_s'] = array('d', [0.0])
            tree.Branch('SamplingWidth_s', dloc['SamplingWidth_s'], 'SamplingWidth_s' + '/d')
        # Now read in data from binary files
        for ch in binary_files.keys():
            ibuff = io_dictionary[ch]['i'].read(4*sample_freq*event_duration)
            qbuff = io_dictionary[ch]['q'].read(4*sample_freq*event_duration)
            ivec = np.array(struct.unpack(endian + 'i'*sample_freq*event_duration, ibuff))*scale_factor
            qvec = np.array(struct.unpack(endian + 'i'*sample_freq*event_duration, qbuff))*scale_factor
            dloc['Waveform{:03d}'.format(ch)].assign(-1*np.sqrt(ivec*ivec + qvec*qvec))
        # For this entry write the 'common' stuff
        dloc['Timestamp_s'][0] = int(timestamp)
        dloc['Timestamp_mus'][0] = int(timestamp_mus)
        dloc['NumberOfSamples'][0] = int(sample_freq*event_duration)
        dloc['SamplingWidth_s'][0] = 1/sample_freq
        tree.Fill()
        entry += 1
        timestamp += event_duration
        if entry == events_per_partial:
            # when we have written all entries catch the tree write and cleanup
            tree.Write()
            del tree
            del tfile
            print("Done with file: {}".format(partial_name))
        left_to_read -= 4*sample_freq*event_duration
    # done with all reading?
    return True

def process_binfile(input_directory, output_directory, run_number, sample_freq, channels=None, tz_offset=0, use_parallel=False):
    """Actually parse log files."""
    #list_of_header_files = glob.glob('{}/*.hdr'.format(input_directory))  # should be just one
    if channels is None:
        list_of_i_files = glob.glob('{}/_bolo_c*_i_raw32'.format(input_directory))
        list_of_q_files = glob.glob('{}/_bolo_c*_q_raw32'.format(input_directory))
        if (len(list_of_i_files) != len(list_of_q_files)):
            print("Mismatch in number of i and q files!")
            print("There are {} i files and {} q files".format(len(list_of_i_files), len(list_of_q_files)))
            return False
        # Convert to dictionary now splitting by channel number
        dict_of_files = {int(ifile.split('ch')[1].split('_')[0]): {'i': ifile} for ifile in list_of_i_files}
        for qfile in list_of_q_files:
            dict_of_files[int(qfile.split('ch')[1].split("_")[0])] |= {'q': qfile}
    else:
        dict_of_files = {ch: {'i': input_directory + "/_bolo_c{:03d}_i_raw32".format(ch), 'q': input_directory + "/_bolo_c{:03d}_q_raw32".format(ch)} for ch in channels}
        # Check existence
        for ch in dict_of_files.keys():
            if not opexists(dict_of_files[ch]['i']):
                print("Warning! The file {} does not exist!".format(dict_of_files[ch]['i']))
                return False
            if not opexists(dict_of_files[ch]['q]']):
                print("Warning! The file {} does not exist!".format(dict_of_files[ch]['q']))
                return False
    # If we are here then our files are found and exist for both i and q
    print('After gobbing, the number of files is {}'.format(2*len(dict_of_files)))
    # NATURAL SORT
    #dre = re.compile(r'(\d+)')
    #list_of_files.sort(key=lambda l: [int(s) if s.isdigit() else s.lower() for s in re.split(dre, l)])
    #print('The list of files after sorting is: {}'.format(list_of_files))
    #print('The size of the file list is: {}'.format(len(list_of_files)))
    # We need to group together i and q files for a given path so do so by channel

    #header_info, ch_info = read_header_file(list_of_header_files[0])
    #header_info = parse_header_time(header_info, tz_offset, manual_tstart=None)
    result = binfile_converter(output_directory, dict_of_files, sample_freq=sample_freq, run_number=run_number, event_duration=1, events_per_partial=50)
    return result


def get_args():
    """Get and parse input arguments when calling module."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inputDirectory',
                        help='Specify the full path of the directory containing the binary files to convert')
    parser.add_argument('-o', '--outputDirectory',
                        help='Specify output directory. If not a full path, it will be output in the same directory as the input directory')
    parser.add_argument('-r', '--runNumber',
                        help='Specify the run number in the log file to convert')
    parser.add_argument('-s', '--sampleRate', default=4000, type=int,
                        help='Specify the sampling rate in Hz')
    parser.add_argument('-z', '--tzOffset', default=0.0, type=float,
                        help='The number of hours of timezone offset to use.\
                        Default is 0 and assumes timestamps to convert are from the same timezone.\
                        If you need to convert to an earlier timezone use a negative number.')
    parser.add_argument('-c', '--channels', nargs='+', type=int, default=None, help="Specify a list of channels to convert")
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
    process_binfile(ARGS.inputDirectory, ARGS.outputDirectory, ARGS.runNumber, ARGS.sampleRate, channels=ARGS.channels, tz_offset=ARGS.tzOffset, use_parallel=ARGS.useParallel)
    print('All done')
