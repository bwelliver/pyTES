import os
import time
from os.path import isabs, dirname, basename
import argparse

import numpy as np
import pandas as pan
from numba import jit

from iv_input_arguments import InputArguments
import iv_processor
import iv_plots as ivp
from ring_buffer import RingBuffer
import squid_info

import ROOT as rt
from readROOT import readROOT
from writeROOT import writeROOT


def mkdpaths(dirpath):
    '''Function to make a directory path if it is not present'''
    os.makedirs(dirpath, exist_ok=True)
    return True


def get_tree_names(input_file):
    '''Quick and dirty function to get name of trees'''
    rt_tfile = getattr(rt, 'TFile')  # avoids pylint complaining rt.TFile doesn't exist
    tfile = rt_tfile.Open(input_file)
    # tFile = TFile.Open(input_file)
    # tDir = tFile.Get(directory)
    keys = tfile.GetListOfKeys()
    key_list = [key.GetName() for key in keys]
    del tfile
    return key_list


def read_from_ivroot(filename, branches):
    '''Read data from special IV root file and put it into a dictionary
    TDir - iv
        TTrees - temperatures
            TBranches - iv properties
    '''
    print('Trying to open {}'.format(filename))
    iv_dictionary = {}
    tree_names = get_tree_names(filename)
    # branches = ['iv/' + branch for branch in branches]
    method = 'single'
    for tree_name in tree_names:
        print('Trying to get tree {} and branches {}'.format(tree_name, branches))
        rdata = readROOT(filename, tree_name, branches, method)
        temperature = tree_name.strip('T')
        iv_dictionary[temperature] = rdata['data']
    return iv_dictionary


def save_iv_to_root(output_directory, iv_dictionary):
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
    data = {'TTree': {}}
    for temperature, iv_data in iv_dictionary.items():
        data['TTree']['T' + temperature] = {'TBranch': {}}
        for key, value in iv_data.items():
            data['TTree']['T' + temperature]['TBranch'][key] = value
    # print(data)
    # We should also make an object that tells us what the other tree values are
    # data['TDirectory']['iv']['TTree']['names']['TBranch'] =
    mkdpaths(output_directory + '/root')
    out_file = output_directory + '/root/iv_data.root'
    status = writeROOT(out_file, data)
    return status


@jit(nopython=True)
def waveform_processor(samples, number_of_windows, process_type):
    '''Basic waveform processor'''
#    subsamples = np.split(samples, number_of_windows, axis=0)
#    mean_samples = np.mean(subsamples, 1)
#    std_samples = np.std(subsamples, 1)
    # If using numba try for loops again since np.split is not supported
    # Also numpy.append is not supported >:(
    if process_type == 'mean':
        mean_samples = np.zeros(number_of_windows)
        std_samples = np.zeros(number_of_windows)
        sz = samples.size
        window_size = sz//number_of_windows
        for idx in range(number_of_windows):
            start_idx = idx*window_size
            end_idx = (idx+1)*window_size
            mean_samples[idx] = np.mean(samples[start_idx:end_idx])
            std_samples[idx] = np.std(samples[start_idx:end_idx])
    if process_type == 'serr_mean':
        mean_samples = np.zeros(number_of_windows)
        std_samples = np.zeros(number_of_windows)
        sz = samples.size
        window_size = sz//number_of_windows
        for idx in range(number_of_windows):
            start_idx = idx*window_size
            end_idx = (idx+1)*window_size
            mean_samples[idx] = np.mean(samples[start_idx:end_idx])
            std_samples[idx] = np.std(samples[start_idx:end_idx])/np.sqrt(window_size)
    if process_type == 'median':
        mean_samples = np.zeros(number_of_windows)
        std_samples = np.zeros(number_of_windows)
        sz = samples.size
        window_size = sz//number_of_windows
        for idx in range(number_of_windows):
            start_idx = idx*window_size
            end_idx = (idx+1)*window_size
            mean_samples[idx] = np.median(samples[start_idx:end_idx])
            std_samples[idx] = nmad(samples[start_idx:end_idx], mean_samples[idx])
    return mean_samples, std_samples


def process_waveform(waveform, time_values, sample_length, number_of_windows=1, process_type='serr_mean'):
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
    # Here we will use numpy.split which will split an array into N equal length sections
    # The return is a list of the sub-arrays.
    # process_type = 'median'
    print('Processing waveform with {} windows'.format(number_of_windows))
    print('The len of the waveform is {} and the len of time is {}'.format(len(waveform), time_values.size))
    # Step 1: How many actual entries will we wind up with?
    number_of_entries = len(waveform) * number_of_windows
    #processed_waveform = {'mean_waveform': np.empty(number_of_entries), 'rms_waveform': np.empty(number_of_entries), 'new_time': np.empty(number_of_entries)}
    processed_waveform = {'mean_waveform': [], 'rms_waveform': [], 'new_time': []}
    if process_type == 'mean' or process_type == 'serr_mean':
        for event, samples in waveform.items():
            event_metadata = {'base_index': samples.size//number_of_windows, 'event_time': time_values[event]}
            sub_times = [event_metadata['event_time'] + sample_length/(2*number_of_windows) + idx/number_of_windows for idx in range(number_of_windows)]
            mean_samples, std_samples = waveform_processor(samples, number_of_windows, process_type=process_type)
            #mean_samples, std_samples = average_groups(samples, number_of_windows)
            #processed_waveform['mean_waveform'].extend(mean_samples)
            #processed_waveform['rms_waveform'].extend(std_samples)
            processed_waveform['new_time'].extend(sub_times)
            start_index = event*number_of_windows
            end_index = start_index + number_of_windows
            processed_waveform['mean_waveform'][start_index:end_index] = mean_samples
            processed_waveform['rms_waveform'][start_index:end_index] = std_samples
            # upper_index + lower_index = n*base_index + base_index + n*base_index = (2n+1)*base_index
            #processed_waveform['new_time'][start_index:end_index] = sub_times
    if process_type == 'median':
        for event, samples in waveform.items():
            event_metadata = {'base_index': samples.size//number_of_windows, 'event_time': time_values[event]}
            # warning: time_values[event] is just the starting timestamp of the event. To get the timestamp of the particular window requires using sample length
            # What we know: 1 event is 1 second long. Therefore each sample is sample_length s separated from the previous sample in a given event.
            # If we had 1 window we should collapse the timestamp to event_time + (samples.size/2)*sample_length
            # If we have 2 windows now each should be the middle of their respective blocks. event_time + (samples.size)/(2*2) and event_time + (2+1)*(samples.size)/(2*2)
            # If we have 3 windows then should be in middle of each third.
            # event_time + (sample_length/(2*3)), event_time + ((sample_length/(2*3))
            # So times are:
            sub_times = [event_metadata['event_time'] + sample_length/(2*number_of_windows) + idx/number_of_windows for idx in range(number_of_windows)]
            median_samples, mad_samples = waveform_processor(samples, number_of_windows, process_type=process_type)
            # this array is number_of_windows long
            # In principle now we have an array of sub-samples associated with this event
            # We can simply append them now to an existing array. But growing by appending is slow
            # so again we associated subsamples[i] with main[k] through some map of k:<->i
            # Or we can do slice assignment. Clever accounting for the slice of main and subsample will allow this.
            processed_waveform['mean_waveform'].extend(median_samples)
            processed_waveform['rms_waveform'].extend(mad_samples)
            processed_waveform['new_time'].extend(sub_times)
            #start_index = event*number_of_windows
            #end_index = start_index + number_of_windows
            #processed_waveform['mean_waveform'][start_index:end_index] = median_samples
            #processed_waveform['rms_waveform'][start_index:end_index] = mad_samples
            # upper_index + lower_index = n*base_index + base_index + n*base_index = (2n+1)*base_index
            #processed_waveform['new_time'][start_index:end_index] = sub_times
    return processed_waveform

def format_and_make_output_path(path, output_path):
    '''This function will format the output path into an absolute path and make
    the directory if needed.
    '''
    output_path = output_path if output_path else dirname(path) + '/' + basename(path).replace('.root', '')
    if not isabs(output_path):
        output_path = dirname(path) + '/' + output_path
    mkdpaths(output_path)
    return output_path


def get_iv_data_from_file(input_path, new_format=False, thermometer='EP'):
    '''Load IV data from specified directory'''
    if thermometer == 'EP':
        thermometer_name = 'EPCal_K'
    else:
        thermometer_name = 'NT'
    if new_format is False:
        tree = 'data_tree'
        branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform', thermometer_name]
        method = 'chain'
        iv_data = readROOT(input_path, tree, branches, method)
        # Make life easier:
        iv_data = iv_data['data']
    else:
        chlist = 'ChList'
        channels = readROOT(input_path, tree=None, branches=None, method='single', tobject=chlist)
        channels = channels['data'][chlist]
        branches = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', thermometer_name]
        branches = branches + ['Waveform' + '{:03d}'.format(int(i)) for i in channels]
        print('Branches to be read are: {}'.format(branches))
        tree = 'data_tree'
        method = 'chain'
        iv_data = readROOT(input_path, tree, branches, method)
        iv_data = iv_data['data']
        iv_data['Channel'] = channels
    return iv_data


def get_pyiv_data(input_path, output_path, new_format=True, number_of_windows=1, thermometer='EP'):
    '''Function to gather iv data in correct format
    Returns time values, temperatures, mean waveforms, rms waveforms and the list of times for temperature jumps
    '''
    iv_data = get_iv_data_from_file(input_path, new_format=new_format, thermometer=thermometer)
    formatted_data = format_iv_data(iv_data, output_path, new_format=new_format, number_of_windows=number_of_windows, thermometer=thermometer)
    return formatted_data


def parse_temperature_steps(output_path, time_values, temperatures, pid_log, tz_correction):
    '''Run through the PID log and parse temperature steps
    The PID log has as the first column the timestamp a PID setting STARTS
    The second column is the power or temperature setting point
    '''
    times = pan.read_csv(pid_log, delimiter='\t', header=None)
    times = times.values[:, 0]
    times = times + tz_correction  # adjust for any timezone issues
    # Each index of times is now the starting time of a temperature step.
    # Include an appropriate offset for mean computation BUT only a softer one for time boundaries
    # time_list is a list of tuples.
    time_list = []
    start_offset = 1*60
    end_offset = 45
    if times.size > 1:
        for index in range(times.size - 1):
            cut = np.logical_and(time_values > times[index]+start_offset, time_values < times[index+1] - end_offset)
            mean_temperature = np.mean(temperatures[cut])
            serr_temperature = np.std(temperatures[cut])/np.sqrt(temperatures[cut].size)
            start_time = times[index] + start_offset
            stop_time = times[index + 1] - end_offset
            time_list.append((start_time, stop_time, mean_temperature, serr_temperature))
        # Handle the last step
        # How long was the previous step?
        d_t = time_list[0][1] - time_list[0][0]
        start_time = time_list[-1][1] + start_offset
        end_time = start_time + d_t - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mean_temperature = np.mean(temperatures[cut])
        serr_temperature = np.std(temperatures[cut])/np.sqrt(temperatures[cut].size)
        time_list.append((start_time, end_time, mean_temperature, serr_temperature))
        d_t = time_values - time_values[0]
    else:
        # Only 1 temperature step defined
        start_time = times[0] + start_offset
        end_time = time_values[-1] - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mean_temperature = np.mean(temperatures[cut])
        time_list.append((start_time, end_time, mean_temperature, serr_temperature))
        d_t = time_values - time_values[0]
    ivp.test_steps(d_t, temperatures, time_list, time_values[0], 'Time', 'T', output_path + '/' + 'test_temperature_steps.png')
    return time_list


def get_temperature_steps(output_path, time_values, temperatures, pid_log, thermometer='EP', tz_correction=0):
    '''Returns a list of tuples that corresponds to temperature steps
    Depending on the value of pid_log we will either parse an existing pid log file or if it is None
    attempt to find the temperature steps
    '''
    if pid_log is None:
        timelist = find_temperature_steps(output_path, time_values, temperatures, thermometer)
    else:
        timelist = parse_temperature_steps(output_path, time_values, temperatures, pid_log, tz_correction)
    return timelist


def chop_data_by_temperature_steps(iv_data, timelist, thermometer_name, bias_channel, data_channel, thermometer, squid):
    '''Chop up waveform data based on temperature steps'''
    squid_parameters = squid_info.SQUIDParameters(squid)
    r_bias = squid_parameters.Rbias
    time_buffer = 0
    iv_dictionary = {}
    # The following defines a range of temperatures to reject. That is:
    # reject = cut_temperature_min < T < cut_temperature_max
    #FIXME:
    # Put these in units of mK for now...this is a hack!
    cut_temperature_max = 1  # Should be the max rejected temperature
    cut_temperature_min = 0  # Should be the minimum rejected temperature
    expected_duration = 4800  # TODO: make this an input argument or auto-determined somehow
    # Now chop up the IV data into steps keyed by the mean temperature
    print('The size of the timelist is: {}'.format(timelist))
    for values in timelist:
        start_time, stop_time, mean_temperature, serr_temperature = values
        print('The mean temperature is: {}'.format(mean_temperature))
        times = iv_data['Timestamp_s'] + iv_data['Timestamp_mus']/1e6
        cut = np.logical_and(times >= start_time + time_buffer, times <= stop_time)
        # Warning: iv_data[WaveformXYZ] is a dictionary! Its keys are event numbers and its values are the samples.
        n_events = len(iv_data['Waveform' + '{:03d}'.format(int(bias_channel))])
        sz_array = iv_data['Waveform' + '{:03d}'.format(int(bias_channel))][0].size
        print('Size check: The size of times is: {}, the number of events in waveform is: {}, the size of the waveform itself is: {}'.format(times.size, n_events, sz_array))
        bias = np.empty((n_events, sz_array))
        for event, sample in iv_data['Waveform' + '{:03d}'.format(int(bias_channel))].items():
            bias[event] = sample
        bias = bias/r_bias
        n_events = len(iv_data['Waveform' + '{:03d}'.format(int(data_channel))])
        sz_array = iv_data['Waveform' + '{:03d}'.format(int(data_channel))][0].size
        v_out = np.empty((n_events, sz_array))
        for event, sample in iv_data['Waveform' + '{:03d}'.format(int(data_channel))].items():
            v_out[event] = sample
        times = times[cut]
        bias = bias[cut]
        v_out = v_out[cut]
        temperatures = iv_data[thermometer_name][cut]
        sampling_width = iv_data['SamplingWidth_s'][cut]
        temperature_key = str(np.round(mean_temperature*1e3, 3))
        # Toss out any T values where the digitizer has railed
        if np.std(v_out)/np.sqrt(v_out.size) < 1e-15:
            print('Invalid digitizer response for T: {} mK'.format(temperature_key))
            continue
        if stop_time - start_time > expected_duration:
            print('Temperature step is too long for T: {} mK. End: {}, Start: {}, Duration: {}'.format(temperature_key, stop_time, start_time, stop_time - start_time))
            continue
        if cut_temperature_min < mean_temperature*1e3 < cut_temperature_max:
            continue
        else:
            # Make sure things are sorted by increasing time
            sort_key = np.argsort(times)
            times = times[sort_key]
            bias = bias[sort_key]
            v_out = v_out[sort_key]
            temperatures = temperatures[sort_key]
            print('Creating dictionary entry for T: {} mK'.format(temperature_key))
            iv_dictionary[temperature_key] = {
                    'iBias': bias,
                    'vOut': v_out,
                    'timestamps': times,
                    'temperatures': temperatures,
                    'sampling_width': sampling_width
                    }
    return iv_dictionary


def get_iv_temperature_data(argin):
    '''Based on requested input data, load the IV data into memory and split by temperatures'''
    # Step 1: Load IV data into memory
    iv_data = get_iv_data_from_file(input_path=argin.inputPath, new_format=argin.newFormat, thermometer=argin.thermometer)
    # Step 2: Get temperature steps
    time_values = iv_data['Timestamp_s'] + iv_data['Timestamp_mus']/1e6
    if argin.thermometer == 'EP':
        thermometer_name = 'EPCal_K'
    else:
        thermometer_name = 'NT'
    timelist = get_temperature_steps(argin.outputPath, time_values=time_values, temperatures=iv_data[thermometer_name], pid_log=argin.pidLog, thermometer=argin.thermometer, tz_correction=argin.tzOffset)
    iv_dictionary = chop_data_by_temperature_steps(iv_data, timelist, thermometer_name, argin.biasChannel, argin.dataChannel, thermometer_name, argin.squid)
    return iv_dictionary


def process_iv_curves(output_path, data_channel, squid, iv_dictionary, number_of_windows=0):
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

    # First determine if we need to split these data up into windows
    iv_curves = {}
    if number_of_windows > 0:
        for temperature, iv_data in sorted(iv_dictionary.items()):
            st = time.time()
            processed_waveforms = process_waveform(iv_data['iBias'], iv_data['timestamps'], iv_data['sampling_width'][0], number_of_windows=number_of_windows)
            print('Process function took: {} s to run'.format(time.time() - st))
            iBias, iBias_err, time_values = processed_waveforms.values()
            st = time.time()
            processed_waveforms = process_waveform(iv_data['vOut'], iv_data['timestamps'], iv_data['sampling_width'][0], number_of_windows=number_of_windows)
            print('Process function took: {} s to run'.format(time.time() - st))
            vOut, vOut_err, time_values = processed_waveforms.values()
            iv_curves[temperature] = {'iBias': iBias, 'iBias_rms': iBias_err, 'vOut': vOut, 'vOut_rms': vOut_err, 'timestamps': time_values}
    else:
        for temperature, iv_data in sorted(iv_dictionary.items()):
            # Take an estimate for the distribution width using 5% of the total data
            end_idx = int(0.05*iv_data['iBias'].size)
            iv_curves[temperature] = {
                    'iBias': iv_data['iBias'], 'iBias_rms':  np.std(iv_data['iBias'].flatten()[0:end_idx])/np.sqrt(end_idx),
                    'vOut': iv_data['vOut'], 'vOut_rms': np.std(iv_data['vOut'].flatten()[0:end_idx])/np.sqrt(end_idx), 'timestamps': iv_data['timestamps']
                    }
    # Next try to obtain a measure of the parasitic series resistance. Note that this value is subtracted
    # from subsquent fitted values of the TES resistance and is assumed to be T indepdent.
    parasitic_dictionary, min_temperature = iv_processor.get_parasitic_resistances(iv_curves, squid)
    # Loop through the iv data now and obtain fit parameters and correct alignment
    fit_parameters_dictionary = {}
    for temperature, iv_data in sorted(iv_curves.items()):
        fit_parameters_dictionary[temperature] = iv_processor.fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigma_y=iv_data['vOut_rms'], plane='iv')
        # Make it pass through zero. Correct offset.
        # i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'interceptbalance')
        i_offset, v_offset = iv_processor.correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'dual')
        print('The maximum iBias={} and the minimum iBias={} with a total size={}'.format(iv_data['iBias'].max(), iv_data['iBias'].min(), iv_data['iBias'].size))
        print('For temperature {} the normal offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
        iv_data['vOut'] -= v_offset
        iv_data['iBias'] -= i_offset
        fit_parameters_dictionary[temperature] = iv_processor.fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigma_y=iv_data['vOut_rms'], plane='iv')
    # Next loop through to generate plots
    for temperature, iv_data in sorted(iv_curves.items()):
        # Make I-V plot
        file_name = output_path + '/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
        plt_data = [iv_data['iBias'], iv_data['vOut'], iv_data['iBias_rms'], iv_data['vOut_rms']]
        axes_options = {'xlabel': 'Bias Current [uA]',
                        'ylabel': 'Output Voltage [mV]',
                        'title': 'Channel {} Output Voltage vs Bias Current for temperatures = {} mK'.format(data_channel, temperature)
                        }
        model_resistance = iv_processor.convert_fit_to_resistance(fit_parameters_dictionary[temperature], squid, fit_type='iv', r_p=parasitic_dictionary[min_temperature].value, r_p_rms=parasitic_dictionary[min_temperature].rms)
        ivp.iv_fitplot(plt_data, fit_parameters_dictionary[temperature], model_resistance, parasitic_dictionary[min_temperature], file_name, axes_options, xscale=1e6, yscale=1e3)
        # Let's make a ROOT style plot (yuck)
        # ivp.make_root_plot(output_path, data_channel, temperature, iv_data, fit_parameters_dictionary[temperature], parasitic_dictionary[min_temperature], xscale=1e6, yscale=1e3)
        iv_dictionary = iv_curves
        iv_dictionary['fit_parameters'] = fit_parameters_dictionary
        iv_dictionary['parasitic'] = parasitic_dictionary
    return iv_data


def iv_main(argin):
    '''The main IV processing function that will call other functions in order
    to process the entire IV dataset
    '''
    argin.outputPath = format_and_make_output_path(argin.inputPath, argin.outputPath)
    print('We will run with the following options:')
    print('The squid run is {}'.format(argin.run))
    print('The SQUID is: {}'.format(argin.squid))
    print('The output path is: {}'.format(argin.outputPath))

    if argin.readROOT is False and argin.readTESROOT is False:
        # Step 1: Load IV data into memory from the base data files and split it up into distinct temperature steps
        # It will be a dictionary whose keys are temperatures (in mK). The sub-dictionary has keys that are iBias, vOut, timestamps, temperatures, sampling_width
        iv_dictionary = get_iv_temperature_data(argin)
        # Step 2: Save these chopped up IV data
        # save_iv_to_root(argin.outputPath, iv_dictionary)
    if argin.readROOT is True and argin.readTESROOT is False:
        # This loads saved data from steps 1 and 2 if it has been performed already and will put us in a state to proceed with TES quantity computations
        iv_dictionary = read_from_ivroot(argin.outputPath + '/root/iv_data.root', branches=['iBias', 'vOut', 'timestamps', 'temperatures', 'sampling_width'])
    if argin.readTESROOT is False:
        # Step 3:
        iv_curves = process_iv_curves(argin.outputPath, argin.dataChannel, argin.squid, iv_dictionary, argin.numberOfWindows)
        plot_iv_curves(argin.outputPath,)



    ######### old stuff below here ##########
#    iv_curves = {}
#    # First step is to get basic IV data into a dictionary format. Either read raw files or load from a saved root file
#    if argin.readROOT is False and argin.readTESROOT is False:
#        iv_curves['iv'] = get_iv_data(argin)
#        # Next try to correct squid jumps
#        # iv_curves['iv'] = correct_squid_jumps(argin.outputPath, iv_curves['iv'])
#        iv_curves = compute_extra_quantities(iv_curves)
#        # Next save the iv_curves
#        save_iv_to_root(argin.outputPath, iv_curves['iv'])
#    if argin.readROOT is True and argin.readTESROOT is False:
#        # If we saved the root file and want to load it do so here
#        iv_curves['iv'] = read_from_ivroot(argin.outputPath + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 'timestamps'])
#    # Next we can process the IV curves to get Rn and r_p values. Once we have r_p we can obtain vTES and go onward
#    if argin.readTESROOT is False:
#        iv_curves = process_iv_curves(argin.outputPath, argin.dataChannel, argin.squid, iv_curves)
#        save_iv_to_root(argin.outputPath, iv_curves['iv'])
#        iv_curves = get_tes_values(iv_curves, argin.squid)
#        save_iv_to_root(argin.outputPath, iv_curves['iv'])
#        print('Obtained TES values')
#    if argin.readTESROOT is True:
#        iv_curves['iv'] = read_from_ivroot(argin.outputPath + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 'timestamps', 'iTES', 'iTES_rms', 'vTES', 'vTES_rms', 'rTES', 'rTES_rms', 'pTES', 'pTES_rms'])
#        # Note: We would need to also save or re-generate the fit_parameters dictionary?
#    # This step onwards assumes iv_dictionary contains TES values
#    iv_curves = process_tes_curves(iv_curves)
#    # Make TES Plots
#    if argin.plotTES is True:
#        make_tes_plots(output_path=argin.outputPath,  data_channel=argin.dataChannel, squid=argin.squid, iv_dictionary=iv_curves['iv'], fit_dictionary=iv_curves['tes_fit_parameters'], individual=True)
#    # Next let's do some special processing...R vs T, P vs T type of thing
#    get_power_temperature_curves(argin.outputPath, argin.dataChannel, iv_curves['iv'])
#    get_resistance_temperature_curves_new(argin.outputPath, argin.dataChannel, iv_curves['iv'])
    return True


def input_parser():
    '''Parse input arguments and return an argument object'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputPath', default='/Users/bwelliver/cuore/bolord/squid/',
                        help='Specify full file path of input data')
    parser.add_argument('-o', '--outputPath',
                        help='Path to put the plot directory. Defaults to input directory')
    parser.add_argument('-r', '--run', type=int,
                        help='Specify the SQUID run number to use')
    parser.add_argument('-b', '--biasChannel', type=int, default=5,
                        help='Specify the digitizer channel that corresponds to the bias channel. Defaults to 5')
    parser.add_argument('-d', '--dataChannel', type=int, default=7,
                        help='Specify the digitizer channel that corresponds to the output channel. Defaults to 7')
    parser.add_argument('-s', '--makeROOT', action='store_true', help='Specify whether to write data to a root file')
    parser.add_argument('-L', '--readROOT', action='store_true',
                        help='Read IV data from processed root file. Stored in outputPath /root/iv_data.root')
    parser.add_argument('-l', '--readTESROOT', action='store_true',
                        help='Read IV and TES data from processed root file. Stored in outputPath /root/iv_data.root')
    parser.add_argument('-g', '--plotTES', action='store_true',
                        help='Make plots with computed TES values.')
    parser.add_argument('-n', '--newFormat', action='store_true',
                        help='Specify whether or not to use the new ROOT format for file reading')
    parser.add_argument('-p', '--pidLog', default=None,
                        help='Specify an optional PID log file to denote the step timestamps. If not supplied program will try to find steps automatically')
    parser.add_argument('-w', '--numberOfWindows', default=1, type=int,
                        help='Specify the number of windows to divide one waveform sample up into for averaging. Default is 1 window per waveform.')
    parser.add_argument('-T', '--thermometer', default='EP',
                        help='Specify the name of the thermometer to use. Can be either EP for EPCal (default) or NT for the noise thermometer')
    parser.add_argument('-S', '--squid', help='Specify the serial number of the SQUID being used.')
    parser.add_argument('-z', '--tzOffset', default=0.0, type=float,
                        help='The number of hours of timezone offset to use.\
                        Default is 0 and assumes timestamps to convert are from the same timezone.\
                        If you need to convert to an earlier timezone use a negative number.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = input_parser()
    argin = InputArguments()
    argin.set_from_args(ARGS)
    iv_main(argin)
    print('done')
