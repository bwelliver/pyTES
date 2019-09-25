import os
import time
from os.path import isabs, dirname, basename
import argparse
import numpy as np
from numpy import square as pow2
from numpy import sqrt as npsqrt
from numpy import sum as nsum

from scipy.optimize import curve_fit
from scipy.signal import detrend
from scipy.stats import median_absolute_deviation as mad

from matplotlib import pyplot as plt

import pandas as pan
from numba import jit, prange

import iv_results
import iv_resistance
import squid_info

import IVPlots as ivp
import pyTESFitFunctions as fitfuncs
from ring_buffer import RingBuffer

import ROOT as rt

from readROOT import readROOT
from writeROOT import writeROOT

#from pycallgraph import PyCallGraph
#from pycallgraph import Config
#from pycallgraph.output import GraphvizOutput

EPS = np.finfo(float).eps

# mp.use('agg')


class InputArguments:
    '''Class to store input arguments for use with py_iv'''
    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.inputPath = ''
        self.outputPath = ''
        self.run = 0
        self.biasChannel = 0
        self.dataChannel = 0
        self.makeROOT = False
        self.readROOT = False
        self.readTESROOT = False
        self.plotTES = False
        self.newFormat = False
        self.pidLog = ''
        self.numberOfWindows = 1
        self.squid = ''
        self.tzOffset = 0
        self.thermometer = 'EP'

    def set_from_args(self, args):
        '''Set InputArguments properties from argparse properties'''
        self.set_from_dict(vars(args))

    def set_from_dict(self, dictionary):
        '''Set InputArguments properties from dictionary properties'''
        for key, value in dictionary.items():
            if getattr(self, key, None) is not None:
                setattr(self, key, value)
            else:
                print('The requested argument attribute {} is not a defined property of class InputArguments'.format(key))


class ArrayIsUnsortedException(Exception):
    '''Exception to raise if an array is unsorted'''


class InvalidChannelNumberException(Exception):
    '''Exception to raise if a channel number is invalid'''


class InvalidObjectTypeException(Exception):
    '''Exception to raise if the object passed into a function
    is not the correct type
    '''


class RequiredValueNotSetException(Exception):
    '''Exception to raise if a required value is missing'''


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

def average_groups(a, N):  # N is number of groups and a is input array
    n = len(a)
    m = n//N
    w = np.full(N, m)
    w[:n-m*N] += 1
    sums = np.add.reduceat(a, np.r_[0, w.cumsum()[:-1]])
    means = np.true_divide(sums, w)
    sqsums = np.add.reduceat((a - np.repeat(means, m))**2, np.r_[0, w.cumsum()[:-1]])
    variances = np.true_divide(sqsums, w)
    stdevs = np.sqrt(variances)
    return means, stdevs


@jit(nopython=True)
def nmad(arr, arr_median=None):
    '''Compute MAD using numpy'''
    if arr_median is not None:
        arr_median = np.median(arr)
    return np.median(np.abs(arr - arr_median))


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
            # print('The equality is: {}'.format(np.allclose(mean_samples, mean_samples2)))
            # print('The equality is: {}'.format(np.allclose(std_samples, std_samples2)))
            # this array is number_of_windows long
            # In principle now we have an array of sub-samples associated with this event
            # We can simply append them now to an existing array. But growing by appending is slow
            # so again we associated subsamples[i] with main[k] through some map of k:<->i
            # Or we can do slice assignment. Clever accounting for the slice of main and subsample will allow this.
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
    # Wrap up into numpy
    processed_waveform['mean_waveform'] = np.array(processed_waveform['mean_waveform'])
    processed_waveform['rms_waveform'] = np.array(processed_waveform['rms_waveform'])
    processed_waveform['new_time'] = np.array(processed_waveform['new_time'])
    return processed_waveform


def process_waveform_old(waveform, time_values, sample_length, number_of_windows=1, process_type='mean'):
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
    # print('The first entry contents of waveform are: {}'.format(waveform[0]))
    print('Processing waveform with {} windows...'.format(number_of_windows))
    # Pre-allocation size
    number_of_entries = len(waveform) * number_of_windows
    # mean_waveform = np.empty(number_of_entries)
    # rms_waveform = np.empty(number_of_entries)
    # new_time_values = np.empty(number_of_entries)
    process_type = 'median'
    processed_waveform = {'mean_waveform': np.empty(number_of_entries), 'rms_waveform': np.empty(number_of_entries), 'new_time': np.empty(number_of_entries)}
    if process_type == 'mean':
        for event, samples in waveform.items():
            event_metadata = {'base_index': samples.size//number_of_windows, 'event_time': time_values[event]}
            # base_index = samples.size // number_of_windows
            # event_time = time_values[event]
            for windex in range(number_of_windows):
                # lower_index = n*base_index
                # upper_index = (n+1)*base_index
                subsample = samples[(windex*event_metadata['base_index']):(windex + 1)*event_metadata['base_index']]
                entry_index = windex + number_of_windows*event
                processed_waveform['mean_waveform'][entry_index] = np.mean(subsample)
                processed_waveform['rms_waveform'][entry_index] = np.std(subsample)
                # upper_index + lower_index = n*base_index + base_index + n*base_index = (2n+1)*base_index
                processed_waveform['new_time'][entry_index] = event_metadata['event_time'] + (((2*windex + 1)*event_metadata['base_index'])/2)*sample_length
    if process_type == 'median':
        for event, samples in waveform.items():
            event_metadata = {'base_index': samples.size//number_of_windows, 'event_time': time_values[event]}
            # base_index = samples.size // number_of_windows
            # event_time = time_values[event]
            for windex in range(number_of_windows):
                # lower_index = n*base_index
                # upper_index = (n+1)*base_index
                subsample = samples[(windex*event_metadata['base_index']):(windex + 1)*event_metadata['base_index']]
                entry_index = windex + number_of_windows*event
                processed_waveform['mean_waveform'][entry_index] = np.median(subsample)
                processed_waveform['rms_waveform'][entry_index] = mad(subsample)
                # upper_index + lower_index = n*base_index + base_index + n*base_index = (2n+1)*base_index
                processed_waveform['new_time'][entry_index] = event_metadata['event_time'] + (((2*windex + 1)*event_metadata['base_index'])/2)*sample_length
    return processed_waveform


def correct_squid_jumps(output_path, iv_dictionary):
    '''Attempt to correct squid jumps for various IV data collections'''
    for temperature, iv_data in iv_dictionary.items():
        print('Attempting to correct for SQUID jumps for temperature {}'.format(temperature))
        iv_data = find_and_fix_squid_jumps_new(output_path, temperature, iv_data)
    return iv_dictionary


def find_and_fix_squid_jumps_new(output_path, temperature, iv_data, event_start=0, buffer_size=5):
    '''A function to try and correct SQUID jumps'''
    # This is a bit of a tricky situation. Sudden changes in output voltage can be the result of a SQUID jump
    # or it can be the result of a transition between the SC and N state

    # Let's outline the process here
    # Look in the vOut vs time plane
    # Look for a jump in the data and flag it
    # Examine data on either side of the jump. If the slope has changed then
    # it is probably a TES state change and is OK. If the slope is the same then
    # it is probably a SQUID jump. This should be OK to use but it fails for the case
    # where a SQUID jump happens during a TES state change. This is probably the most likely
    # type of jump to occur.

    # Note: detrend just does not work well.
    # One option is to use normal branch interpolation. Here we detect any type of jump
    # and make a note of whether the slope is smaller or larger in the new region
    # If the slope is larger, it is a N --> SC jump and if it is smaller it is a SC --> N jump
    # This then is used to label the regions
    # Next jump that occurs we ensure the slope is correct (should be the next type of state)
    # and then compare the y-intercept of the fit. If the y-intercepts do not match, move everything up
    # Finally reset the window with the correct label again
    timestamps = iv_data['timestamps']
    ydata = iv_data['vOut']
    dy_dt = np.gradient(ydata, timestamps, edge_order=2)
    return iv_data




def find_and_fix_squid_jumps(output_path, temperature, iv_data, event_start=0, buffer_size=5):
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
    timestamps = iv_data['timestamps']
    # x = iv_data['iBias']
    # xrms = iv_data['iBias_rms']
    ydata = iv_data['vOut']
    # yrms = iv_data['vOut_rms']
    if np.all(timestamps[:-1] <= timestamps[1:]) is False:
        raise ArrayIsUnsortedException('Input argument timestamps is unsorted')
    # First let us compute the gradients with respect to time
    dyd_t = np.gradient(ydata, timestamps, edge_order=2)
    # dxd_t = np.gradient(xdata, timestamps, edge_order=2)
    # Next construct dydx in time ordered sense
    # dydx = dyd_t/dxd_t
    # So we will walk along and compute the average of N elements at a time.
    # If the new average differs from the previous by some amount mark that as the boundary of a SQUID jump
    # This should not be a subtle thing.
    # Make a plot of what we are testing
    ivp.test_plot(timestamps, ydata, 'time', 'vOut', output_path + 'uncorrected_squid_jumps_' + str(temperature) + 'event_start_' + str(event_start) + '_' + 'mK.png')
    dbuff = RingBuffer(buffer_size, dtype=float)
    buffer_size = buffer_size if event_start + buffer_size < dyd_t.size - 1 else dyd_t.size - event_start - 1
    for event in range(buffer_size):
        dbuff.append(dyd_t[event_start + event])
    # Now our buffer is initialized so loop over all events until we find a change
    event = event_start + buffer_size
    difference_of_means = 0
    print('The first y value is: {} and the location of the max y value is: {}'.format(ydata[0], np.argmax(ydata)))
    while difference_of_means < 2 and event < dyd_t.size - 1:
        current_mean = dbuff.get_mean()
        dbuff.append(dyd_t[event])
        new_mean = dbuff.get_mean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        # print('The current y value is: {}'.format(y[event]))
        # print('event {}: currentMean = {}, newMean = {}, dMean = {}'.format(event, currentMean, newMean, dMean))
        event += 1
    # We have located a potential jump at this point (event)
    # So compute the slope on either side of event and compare
    print('The event and size of the data are {} and {}'.format(event, timestamps.size))
    if event >= timestamps.size - 1:
        print('No jumps found after walking all of the data...')
    else:
        # We have found something and are not yet at the end of the array
        # Let's see if we have something...
        step_away = 10
        distance_ahead = np.min([step_away, dyd_t.size - event])
        # Compute distance to look behind.
        # Basically we need to ensure that event - distance_behind >= 0
        if event - step_away < 0:
            distance_behind = event
        else:
            distance_behind = step_away
        # slope_before = np.mean(dyd_t[event-distance_behind:event])
        # slope_after = np.mean(dyd_t[event+1:event+distance_ahead])
        result, pcov = curve_fit(fitfuncs.lin_sq, timestamps[event-distance_behind:event-1], ydata[event-distance_behind:event-1])
        fit_before = result[0]
        result, pcov = curve_fit(fitfuncs.lin_sq, timestamps[event + 1:event + distance_ahead], ydata[event + 1:event + distance_ahead])
        fit_after = result[0]
        print('The event and size are {} and {} and The slope before is: {}, slope after is: {}'.format(event, dyd_t.size, fit_before, fit_after))
        slope_ratio1 = 1 - fit_after/fit_before
        slope_ratio2 = 1 - fit_before/fit_after
        # Slope ratio values:
        # If the slopes are approximately the same, slope_ratio1 ~ slope_ratio2
        # If the slopes are very different, then one slope ratio will be very big and the other close to 1
        # Diff is 1 - fa/fb - (1 - fb/fa) == fb/fa - fa/fb
        # In the limit that both are the same this tends towards 0. If they are very different it will be big.
        # Note that there should not be a slope change at a squid jump (well...shouldn't be likely)
        # Thus we should require the slope_ratios to be also of different signs. If they are the same sign then we have
        # one of either fa or fb with the opposite sign of the other.
        print('slope_ratio 1 (1 - fa/fb): {}'.format(slope_ratio1))
        print('slope_ratio 2 (1 - fb/fa): {}'.format(slope_ratio2))
        if np.abs(slope_ratio1 - slope_ratio2) < 0.5 and (np.sign(slope_ratio1) != np.sign(slope_ratio2)):
            # This means the slopes are the same and so we have a squid jump
            # Easiest thing to do then is to determine what the actual value was prior to the jump
            # and then shift the offset values up to it
            print('Potential SQUID jump found for temperature {} at event {}'.format(temperature, event))
            result, pcov = curve_fit(fitfuncs.lin_sq, timestamps[event-distance_behind:event], ydata[event-distance_behind:event])
            extrapolated_value_of_post_jump_point = fitfuncs.lin_sq(timestamps[event+5], *result)
            print('The extrapolated value for the post jump point should be: {}. \
                  It is actually right now {}'.format(extrapolated_value_of_post_jump_point, ydata[event-10:event+10]))
            # Now when corrected y' - event = 0 so we need y' = y - dy where dy = y - event
            dydata = ydata[event+5] - extrapolated_value_of_post_jump_point
            ydata[event+5:] = ydata[event+5:] - dydata
            print('After correction the value of y is {}'.format(ydata[event+5]))
            # Finally we must remove the point event from the data. Let's be safe and remove the surrounding points
            points_to_remove = [event + (i-4) for i in range(9)]
            for key in iv_data.keys():
                iv_data[key] = np.delete(iv_data[key], points_to_remove)
            # Test
            print('Does a view still equal the actual array? t is iv_data[timestamps] == {}'.format(timestamps is iv_data['timestamps']))
            ivp.test_plot(timestamps, ydata, 'time', 'vOut', output_path + 'corrected_squid_jumps_' + str(temperature) + 'event_' + str(event) + '_mK.png')
        else:
            # Slopes are not the same...this is not a squid jump.
            print('No SQUID Jump detected up to event {}'.format(event))
        # Cycle through again until we hit the end of the list.
        iv_data = find_and_fix_squid_jumps(output_path, temperature, iv_data, event_start=event)
    return iv_data


def power_temp(T, t_tes, k, n):
    '''Function defining thermal power'''
    # pylint: disable=invalid-name
    P = k*(np.power(t_tes, n) - np.power(T, n))
    return P


def power_temp5(T, t_tes, k):
    '''Function defining thermal power with thermal resistance exponent fixed to 5'''
    # pylint: disable=invalid-name
    P = k*(np.power(t_tes, 5) - np.power(T, 5))
    return P


def fill_ringbuffer(buffer, offset, data):
    '''Fill a buffer until it is full'''
    for idx in range(buffer.size_max):
        buffer.append(data[idx + offset])
    return buffer


def find_end(buffers, temperature_step, temperatures):
    '''find an ending step boundary'''
    # Compute dMu_ij pairs
    step_event = None
    while buffers['event'] < temperatures.size - 1:
        dmu_pc = np.abs(buffers['past'].get_mean() - buffers['current'].get_mean())
        dmu_cf = np.abs(buffers['current'].get_mean() - buffers['future'].get_mean())
        # dMu_pf = np.abs(past_buffer.get_mean() - future_buffer.get_mean())
        # print(past_buffer.get_mean(), current_buffer.get_mean(), future_buffer.get_mean())
        # print(dmu_pc, dmu_cf, dMu_pf)
        # Test if past and future are similar...if these are then we assume current is similar and so advance
        if dmu_pc < 0.5*temperature_step and not dmu_cf < 0.5*temperature_step:
            # This is the return case...current similar to past but not future
            # Make the step_event be the first event of the future
            step_event = buffers['event'] - buffers['future'].get_size()
            print('The step event is {}'.format(step_event))
            return_list = [step_event, buffers]
            break
        else:
            # past and future are within 1/2 temperature_step so increment all things by 1 event
            # other conditions too but for now increment by 1 event
            buffers['past'].append(buffers['current'].get_all()[-1])
            buffers['current'].append(buffers['future'].get_all()[-1])
            buffers['future'].append(temperatures[buffers['event']])
            buffers['event'] += 1
        return_list = [step_event, buffers]
    return return_list


def find_start(buffers, temperature_step, temperatures):
    '''find a starting step boundary'''
    # Compute dMu_ij pairs
    step_event = None
    while buffers['event'] < temperatures.size - 1:
        dmu_pc = np.abs(buffers['past'].get_mean() - buffers['current'].get_mean())
        dmu_cf = np.abs(buffers['current'].get_mean() - buffers['future'].get_mean())
        # dMu_pf = np.abs(past_buffer.get_mean() - future_buffer.get_mean())
        # print(past_buffer.get_mean(), current_buffer.get_mean(), future_buffer.get_mean())
        # print(dmu_pc, dmu_cf, dMu_pf)
        # Test if past and future are similar...if these are then we assume current is similar and so advance
        if dmu_cf < 0.5*temperature_step and not dmu_pc < 0.5*temperature_step:
            # This is the return case...current similar to future but not past
            # Make the step_event be the first event of the current
            step_event = buffers['event'] - buffers['future'].get_size() - buffers['current'].get_size()
            # print('The step event is {}'.format(step_event))
            return_list = [step_event, buffers]
            break
        else:
            # past and future are within 1/2 temperature_step so increment all things by 1 event
            # other conditions too but for now increment by 1 event
            # print('The 3 dMu values for start search are: {}'.format([dmu_pc, dmu_cf, dMu_pf]))
            buffers['past'].append(buffers['current'].get_all()[-1])
            buffers['current'].append(buffers['future'].get_all()[-1])
            buffers['future'].append(temperatures[buffers['event']])
            buffers['event'] += 1
        return_list = [step_event, buffers]
    return return_list


def get_stable_temperature_steps(timestamps, temperatures, buffer_length=10, temperature_step=5e-5):
    '''Function that attempts to find temperature steps and select periods of relatively stable T
    This method will create 3 sliding windows and compute the mean in each window.
    buffer_length - the length of the ring buffer (seconds)
    temperature_step - the step change in temperature (K) to be sensitive to
    past_buffer - the leftmost window and what defines "the past"
    current_buffer - the center window and what defines the window of interest
    future_buffer - the rightmost window and what defines "the future"
    Note that smaller buffer_length means we generally lose sensitivity to larger values of temperature_step
    Values that seem OK are (10, 5e-5), (60, 2e-4), (90, 5e-4)
    We compute pMu, cMu and fMu and then decide which region cMu is similar to
    metric: dMu_ij = |mu_i - mu_j|
    compute for all pairs: p-c, c-f, p-f
    if dMu_ij < 0.5*T_step consider mu_i ~ mu_j
    otherwise means are not the same so a boundary has occurred.
    We start with t = 0 as one boundary and create boundaries in pairs
    First boundary we find will be the end of the first region
    Once an end boundary is found we set past_buffer = future_buffer
    Note: if all dMu combinations are dissimilar we're in a transition region
    '''
    buffers = {
            'past': RingBuffer(buffer_length, dtype=float),
            'current': RingBuffer(buffer_length, dtype=float),
            'future': RingBuffer(buffer_length, dtype=float),
            'event': 0
            }
    # Fill the buffers initially
    time_list = []
    triplet = {'start_time': timestamps[0], 'end_time': None, 'mean_temperature': None}
    d_t = 0
    while buffers['event'] < temperatures.size and buffers['event'] + buffer_length < temperatures.size:
        # We start with assuming the first window starts at t = timestamps[0]
        if time_list == []:
            triplet['start_time'] = timestamps[0]
            # Start past_buffer at ev = 0
            buffers['past'] = fill_ringbuffer(buffers['past'], buffers['event'], temperatures)
            buffers['event'] += buffer_length
            buffers['current'] = fill_ringbuffer(buffers['current'], buffers['event'], temperatures)
            buffers['event'] += buffer_length
            buffers['future'] = fill_ringbuffer(buffers['future'], buffers['event'], temperatures)
            buffers['event'] += buffer_length
            # Now proceed to find an end
        else:
            # we have now found an end so now need to find a new start
            # We need new windows first
            # When we find an end point past ~ current !~ future
            # So adjust past <-- current, current <-- future, future <-- temperatures
            buffers['past'].insert_array(buffers['current'].get_all(), flipped=True)
            buffers['current'].insert_array(buffers['future'].get_all(), flipped=True)
            buffers['future'] = fill_ringbuffer(buffers['future'], buffers['event'], temperatures)
            buffers['event'] += buffer_length
            # common things are: [event, past_buffer, current_buffer, future_buffer]
            step_event, buffers = find_start(buffers, temperature_step, temperatures)
            triplet['start_time'] = timestamps[step_event]
        # Now we have a start_time so we need to find a end_time.
        # When we find a start point, past !~ current ~ future
        # we can keep sliding forward until we reach past ~ current !~ future
        step_event, buffers = find_end(buffers, temperature_step, temperatures)
        triplet['end_time'] = timestamps[step_event]
        # Check validity of this temperature step: It must last longer than some amount of time
        # Also if the temperature is flagged as bad, ignore it.
        bad_temperatures = [(10e-3, 10.8e-3)]
        if triplet['end_time'] - triplet['start_time'] > d_t:
            cut = np.logical_and(timestamps >= triplet['start_time'] + d_t, timestamps <= triplet['end_time'])
            triplet['mean_temperature'] = np.mean(temperatures[cut])
            ctemp = False
            for bad_temp_range in bad_temperatures:
                cbad = np.logical_and(triplet['mean_temperature'] >= bad_temp_range[0], triplet['mean_temperature'] <= bad_temp_range[1])
                if cbad:
                    print('Temperature {} is flagged as a bad temperature and will not be included onward'.format(triplet['mean_temperature']))
                ctemp = np.logical_or(ctemp, cbad)
            if not ctemp:
                time_list.append((triplet['start_time'], triplet['end_time'], triplet['mean_temperature']))
    return time_list


def walk_normal(xdata, ydata, side, buffer_size=40*16):
    '''Function to walk the normal branches and find the line fit
    To do this we will start at the min or max input current and compute a walking derivative
    If the derivative starts to change then this indicates we entered the biased region and should stop
    NOTE: We assume data is sorted by voltage values
    '''
    # Ensure we have the proper sorting of the data
    if not np.all(xdata[:-1] <= xdata[1:]):
        raise ArrayIsUnsortedException('Input argument x is unsorted')
    # We should select only the physical data points for examination
    di_bias = np.gradient(xdata, edge_order=2)
    c_normal_to_sc_pos = np.logical_and(xdata > 0, di_bias < 0)
    c_normal_to_sc_neg = np.logical_and(xdata <= 0, di_bias > 0)
    c_normal_to_sc = np.logical_or(c_normal_to_sc_pos, c_normal_to_sc_neg)

    # First let us compute the gradient (dy/dx)
    dydx = np.gradient(ydata, xdata, edge_order=2)
    # Set data that is in the SC to N transition to NaN in here
    xdata[~c_normal_to_sc] = np.nan
    ydata[~c_normal_to_sc] = np.nan
    dydx[~c_normal_to_sc] = np.nan

    if side == 'right':
        # Flip the array
        dydx = dydx[::-1]
    # In the normal region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time.
    # If the new average differs from the previous by some amount mark that as the boundary to the bias region
    dbuff = RingBuffer(buffer_size, dtype=float)
    for event in range(buffer_size):
        dbuff.append(dydx[event])
    # Now our buffer is initialized so loop over all events until we find a change
    event = buffer_size
    difference_of_means = 0
    d_event = 0
    while difference_of_means < 1e-2 and event < dydx.size - 1:
        current_mean = dbuff.get_nanmean()
        dbuff.append(dydx[event])
        new_mean = dbuff.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        event += 1
        d_event += 1
    if side == 'right':
        # Flip event index back the right way
        event = dydx.size - 1 - event
    #print('The {} deviation occurs at ev = {} with current = {} and voltage = {} with dMean = {}'.format(side, ev, current[ev], voltage[ev], dMean))
    return event


def get_sc_endpoints(buffer_size, index_min_x, dydx):
    '''A function to try and determine the endpoints for the SC region'''
    # Look for rightmost endpoint, keeping in mind it could be our initial point
    if buffer_size + index_min_x >= dydx.size:
        # Buffer size and offset would go past end of data
        right_buffer_size = np.nanmax([dydx.size - index_min_x - 1, 0])
    else:
        right_buffer_size = buffer_size
    slope_buffer = RingBuffer(right_buffer_size, dtype=float)
    # Now fill the buffer
    for event in range(right_buffer_size):
        slope_buffer.append(dydx[index_min_x + event])
    # The buffer is full with initial values. NOw walk along
    ev_right = index_min_x + right_buffer_size
    difference_of_means = 0
    while difference_of_means < 1e-2 and ev_right < dydx.size - 1:
        current_mean = slope_buffer.get_nanmean()
        slope_buffer.append(dydx[ev_right])
        new_mean = slope_buffer.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        ev_right = ev_right + 1
    # Now we must check the left direction. Again keep in mind we might start there.
    if index_min_x - buffer_size <= 0:
        # The buffer would go past the array edge
        left_buffer_size = index_min_x
    else:
        left_buffer_size = buffer_size
    if left_buffer_size == 0:
        # Implies index_min_x is 0. Fit 1 point in?
        left_buffer_size = 1
    print('We will create a ringbuffer with size: {}'.format(left_buffer_size))
    slope_buffer = RingBuffer(left_buffer_size, dtype=float)
    # Do initial appending
    for event in range(left_buffer_size):
        slope_buffer.append(dydx[index_min_x - event])
    # Walk to the left
    ev_left = index_min_x - left_buffer_size
    difference_of_means = 0
    print('The value of ev_left to start is: {}'.format(ev_left))
    while difference_of_means < 1e-2 and ev_left >= 0:
        current_mean = slope_buffer.get_nanmean()
        slope_buffer.append(dydx[ev_left])
        new_mean = slope_buffer.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        ev_left -= 1
    ev_left = ev_left if ev_left >= 0 else ev_left + 1
    return (ev_left, ev_right)


def get_sc_endpoints_old(buffer_size, index_min_x, dydx):
    '''Function to actually locate the end of a SC region'''
    slope_buffer = RingBuffer(buffer_size, dtype=float)
    # First is going from midpoint to the right
    if buffer_size + index_min_x >= dydx.size:
        right_buffer_size = np.nanmax([dydx.size - index_min_x - 1, 0])
    else:
        right_buffer_size = buffer_size
    for event in range(right_buffer_size):
        slope_buffer.append(dydx[index_min_x + event])
    # Now our buffer is initialized so loop over all events until we find a change
    ev_right = index_min_x + right_buffer_size
    difference_of_means = 0
    while difference_of_means < 1e-2 and ev_right < dydx.size - 1:
        current_mean = slope_buffer.get_nanmean()
        slope_buffer.append(dydx[ev_right])
        new_mean = slope_buffer.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        ev_right += 1
    # Now repeat but go to the left from the minimum abs. voltage
    if index_min_x - buffer_size <= 0:
        left_buffer_size = index_min_x
    else:
        left_buffer_size = buffer_size
    slope_buffer = RingBuffer(left_buffer_size, dtype=float)
    for event in range(left_buffer_size):
        slope_buffer.append(dydx[index_min_x - event])
    # Now our buffer is initialized so loop over all events until we find a change
    print('The min x index and buffer size are: {} and {}, with total array size {}'.format(index_min_x, buffer_size, dydx.size))
    ev_left = index_min_x - left_buffer_size
    difference_of_means = 0
    while difference_of_means < 1e-2 and ev_left >= 0:
        current_mean = slope_buffer.get_nanmean()
        slope_buffer.append(dydx[ev_left])
        new_mean = slope_buffer.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        ev_left -= 1
    ev_left = ev_left if ev_left >= 0 else ev_left + 1
    return (ev_left, ev_right)


def walk_sc(xdata, ydata, buffer_size=5*16, plane='iv'):
    '''Function to walk the superconducting region of the IV curve and get the left and right edges
    Generally when ib = 0 we should be superconducting so we will start there and go up until the bias
    then return to 0 and go down until the bias
    In order to be correct your x and y data values must be sorted by x
    '''
    # Ensure we have the proper sorting of the data
    if np.all(xdata[:-1] <= xdata[1:]) is False:
        raise ArrayIsUnsortedException('Input argument x is unsorted')
    # We should select only the physical data points for examination
    di_bias = np.gradient(xdata, edge_order=2)
    print('The size of di_bias is: {}'.format(di_bias.size))
    c_normal_to_sc_pos = np.logical_and(xdata > 0, di_bias < 0)
    c_normal_to_sc_neg = np.logical_and(xdata <= 0, di_bias > 0)
    c_normal_to_sc = np.logical_or(c_normal_to_sc_pos, c_normal_to_sc_neg)
    print('The number of normal to sc points is: {}'.format(np.sum(c_normal_to_sc)))

    # Also select data that is some fraction of the normal resistance, say 20%
    # First let us compute the gradient (i.e. dy/dx)
    dydx = np.gradient(ydata, xdata, edge_order=2)

    # Set data that is in the SC to N transition to NaN in here
    if plane == 'iv':
        #xdata[~c_normal_to_sc] = np.nan
        #ydata[~c_normal_to_sc] = np.nan
        #dydx[~c_normal_to_sc] = np.nan
        print('Setting things to nan')

    # In the sc region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time.
    # If the new average differs from the previous by some amount mark that as the end.

    # First we should find whereabouts of (0,0)
    # This should roughly correspond to x = 0 since if we input nothing we should get out nothing. In reality there are parasitics of course
    if plane == 'tes':
        #ioffset = -1e-8
        # try minimizing ydata here (since it is current)
        #index_min_x = np.nanargmin(np.abs(ydata))
        # Try to find the point closest to 0,0 and use that as our starting
        #cutX = np.logical_and(xdata > -0.5e-6, xdata < 0.5e-6)
        #cutY = np.logical_and(ydata > -0.5e-6, ydata < 0.5e-6)
        #cut = np.logical_and(cutX, cutY) # but how to get index?
        # Ideally we should look for the point that is closest to (0, 0)!
        distance = np.zeros(xdata.size)
        px, py = (0, 0)
        for idx in range(xdata.size):
            dx = xdata[idx] - px
            dy = ydata[idx] - py
            distance[idx] = np.sqrt(dx**2 + dy**2)
        index_min_x = np.nanargmin(distance)
        print('The point closest to ({}, {}) is at index {} with distance {} and is ({}, {})'.format(px, py, index_min_x, distance[index_min_x], xdata[index_min_x], ydata[index_min_x]))
        # Occasionally we may have a shifted curve that is not near 0 for some reason (SQUID jump)
        # So find the min and max iTES and then find the central point
    elif plane == 'iv':
        # Replicate tes plane...except that we have not corrected so (0,0) is not the best
#        distance = np.zeros(xdata.size)
#        px, py = (0, 0)
#        for idx in range(xdata.size):
#            dx = xdata[idx] - px
#            dy = ydata[idx] - py
#            distance[idx] = np.sqrt(dx**2 + dy**2)
#        index_min_x = np.nanargmin(distance)
#        print('The point closest to ({}, {}) is at index {} with distance {} and is ({}, {})'.format(px, py, index_min_x, distance[index_min_x], xdata[index_min_x], ydata[index_min_x]))
        # Find the point closest to 0 iBias.
        ioffset = 0
        index_min_x = np.nanargmin(np.abs(xdata + ioffset))
        # NOTE: The above will fail for small SC regions where vOut normal > vOut sc!!!!
    # First go from index_min_x and increase
    # Create ring buffer of to store signal
    # TODO: FIX THIS TO HANDLE SQUID JUMPS
    # Start by walking buffer_size events to the right from the minimum abs. voltage
    print('The size of dydx is: {}'.format(dydx.size))
    event_values = get_sc_endpoints(buffer_size, index_min_x, dydx)
    return event_values


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
    #print(data)
    # We should also make an object that tells us what the other tree values are
    #data['TDirectory']['iv']['TTree']['names']['TBranch'] =
    mkdpaths(output_directory + '/root')
    out_file = output_directory + '/root/iv_data.root'
    status = writeROOT(out_file, data)
    return status


def make_tes_multiplot(output_path, data_channel, squid, iv_dictionary, fit_parameters: dict):
    '''Make a plot of all temperatures at once
    rTES vs iBias

    '''
    # Convert fit parameters to R values
    resistance = {}
    for key, values in fit_parameters.items():
        resistance[key] = convert_fit_to_resistance(values, squid, fit_type='tes')
    # Current vs Voltage
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    idx = 0
    tmax = 57
    for temperature, data in iv_dictionary.items():
        if idx % 4 != 0 or float(temperature) > tmax:
            idx += 1
            continue
        params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
                  'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None
                  }
        axes_options = {'xlabel': 'Bias Current [uA]',
                        'ylabel': r'TES Resistance [m \Omega]',
                        'title': 'Channel {} TES Resistance vs Bias Current'.format(data_channel)
                        }
        axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['iBias'], y=data['rTES'], params=params, axes_options=axes_options, xscale=xscale, yscale=yscale)
        axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters[temperature], model_function=fitfuncs.lin_sq, xscale=xscale, yscale=yscale)
        axes = ivp.iv_fit_textbox(axes=axes, R=resistance[temperature], model=fit_parameters[temperature])
        idx += 1
    axes.set_ylim((0*yscale, 1*yscale))
    axes.set_xlim((-20, 20))
    file_name = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel)
    ivp.save_plot(fig, axes, file_name)

    # Overlay multiple IV plots
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e6
    temperature_names = []
    idx = 0
    for temperature, data in iv_dictionary.items():
        if idx % 4 != 0 or float(temperature) > tmax:
            idx += 1
            continue
        if temperature not in ['9.908']:
            temperature_names.append(temperature)
            params = {'marker': 'o', 'markersize': 4, 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
            axes_options = {'xlabel': r'TES Voltage [$\mu$V]',
                            'ylabel': r'TES Current [$\mu$A]',
                            'title': None #'Channel {} TES Current vs Voltage'.format(data_channel)
                            }
            axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['iTES'], params=params, axes_options=axes_options, xscale=xscale, yscale=yscale)
        idx += 1
    # Add legend?
    axes.legend(['T = {} mK'.format(temperature) for temperature in temperature_names], markerscale=5, fontsize=24)
    axes.set_ylim((0, 2))
    axes.set_xlim((-0.5, 2))
    file_name = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel)
    ivp.save_plot(fig, axes, file_name, dpi=200)

    # Overlay multiple IV plots
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    temperature_names = []
    idx = 0
    for temperature, data in iv_dictionary.items():
        if idx % 4 != 0 or float(temperature) > tmax:
            idx += 1
            continue
        if temperature not in ['9.908']:
            temperature_names.append(temperature)
            params = {'marker': 'o', 'markersize': 4, 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
            axes_options = {'xlabel': r'TES Voltage [$\mu$V]',
                            'ylabel': r'TES Resistance [m$\Omega$]',
                            'title': None #'Channel {} TES Resistance vs Voltage'.format(data_channel)
                            }
            axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['rTES'], params=params, axes_options=axes_options, xscale=xscale, yscale=yscale)
        idx += 1
    # Add legend?
    axes.legend(['T = {} mK'.format(temperature) for temperature in temperature_names], markerscale=5, fontsize=24)
    axes.set_ylim((0, 720))
    axes.set_xlim((0, 2))
    file_name = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel)
    ivp.save_plot(fig, axes, file_name, dpi=200)
    return True


def plot_current_vs_voltage(output_path, data_channel, squid, temperature, data, fit_parameters):
    '''Plot the current vs voltage for a TES'''
    # Convert fit parameters to R values
    resistance = convert_fit_to_resistance(fit_parameters, squid, fit_type='tes')
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e6
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
              'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': data['vTES_rms']*xscale, 'yerr': data['iTES_rms']*yscale
              }
    axes_options = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Current [uA]',
                    'title': 'Channel {} TES Current vs TES Voltage for temperatures = {} mK'.format(data_channel, temperature)}

    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['iTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, xscale=xscale, yscale=yscale)
    axes = ivp.iv_fit_textbox(axes=axes, R=resistance, model=fit_parameters)

    file_name = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, file_name)
    return True


def plot_resistance_vs_current(output_path, data_channel, temperature, data, fit_parameters):
    '''Plot the resistance vs current for a TES'''
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    ylim = (0, 1*yscale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
              'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': data['iTES_rms']*xscale, 'yerr': data['rTES_rms']*yscale
              }
    axes_options = {'xlabel': 'TES Current [uA]',
                    'ylabel': 'TES Resistance [mOhm]',
                    'title': 'Channel {} TES Resistance vs TES Current for temperatures = {} mK'.format(data_channel, temperature),
                    'ylim': ylim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['iTES'], y=data['rTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'rTES_vs_iTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, file_name)
    return True


def plot_resistance_vs_voltage(output_path, data_channel, temperature, data, fit_parameters):
    '''Plot the resistance vs voltage for a TES'''
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    ylim = (0, 1*yscale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
              'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': data['vTES_rms']*xscale, 'yerr': data['rTES_rms']*yscale
              }
    axes_options = {'xlabel': 'TES Voltage [uV]',
                    'ylabel': 'TES Resistance [mOhm]',
                    'title': 'Channel {} TES Resistance vs TES Voltage for temperatures = {} mK'.format(data_channel, temperature),
                    'ylim': ylim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['rTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, file_name)
    return True


def plot_resistance_vs_bias_current(output_path, data_channel, temperature, data, fit_parameters):
    '''Plot the resistance vs bias current for a TES'''
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    ylim = (0, 1*yscale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['iBias_rms']*xscale, 'yerr': data['rTES_rms']*yscale}
    axes_options = {'xlabel': 'Bias Current [uA]',
                    'ylabel': 'TES Resistance [mOhm]',
                    'title': 'Channel {} TES Resistance vs Bias Current for temperatures = {} mK'.format(data_channel, temperature),
                    'ylim': ylim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['iBias'], y=data['rTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, file_name)
    return True


def plot_power_vs_resistance(output_path, data_channel, temperature, data, fit_parameters):
    '''Plot the resistance vs bias current for a TES'''
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e12
    xlim = (0, 1*xscale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['rTES_rms']*xscale, 'yerr': data['pTES_rms']*yscale}
    axes_options = {'xlabel': 'TES Resistance [mOhm]',
                    'ylabel': 'TES Power [pW]',
                    'title': 'Channel {} TES Power vs TES Resistance for temperatures = {} mK'.format(data_channel, temperature),
                    'xlim': xlim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['rTES'], y=data['pTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'pTES_vs_rTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, file_name)
    return True


def plot_power_vs_voltage(output_path, data_channel, temperature, data, fit_parameters):
    '''Plot the TES power vs TES voltage'''
    # Note this ideally is a parabola
    cut = np.logical_and(data['rTES'] > 500e-3, data['rTES'] < 2*500e-3)
    if nsum(cut) < 3:
        cut = np.ones(data['pTES'].size, dtype=bool)
    vtes = data['vTES'][cut]
    ptes = data['pTES'][cut]
    prms = data['pTES_rms'][cut]
    result, pcov = curve_fit(fitfuncs.quad_sq, vtes, ptes, sigma=prms, absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))
    fit_result = iv_results.FitParameters()
    fit_result.left.set_values(result, perr)
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e12
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
              'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': data['vTES_rms']*xscale, 'yerr': data['pTES_rms']*yscale}
    axes_options = {'xlabel': 'TES Voltage [uV]',
                    'ylabel': 'TES Power [pW]',
                    'title': 'Channel {} TES Power vs TES Resistance for temperatures = {} mK'.format(data_channel, temperature)
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['pTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['pTES'], model=fit_result, model_function=fitfuncs.quad_sq, xscale=xscale, yscale=yscale)
    axes = ivp.pr_fit_textbox(axes=axes, model=fit_result)

    file_name = output_path + '/' + 'pTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, file_name)
    return True


def make_resistance_vs_temperature_plots(output_path, data_channel, fixed_name, fixed_value, norm_to_sc, sc_to_norm, model_func, fit_result):
    '''Function to make R vs T plots for a given set of values'''

    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e3
    sort_key = np.argsort(sc_to_norm['T'])
    nice_current = np.round(fixed_value*1e6, 3)
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'title': None  #'Channel {}'.format(data_channel) + ' TES Resistance vs Temperature for TES Current = {}'.format(nice_current) + r'$\mu$' + 'A'
                    }
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': sc_to_norm['rmsR'][sort_key]*yscale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=sc_to_norm['T'][sort_key], y=sc_to_norm['R'][sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)

    sort_key = np.argsort(norm_to_sc['T'])
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': norm_to_sc['rmsR'][sort_key]*yscale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=norm_to_sc['T'][sort_key], y=norm_to_sc['R'][sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = ivp.add_model_fits(axes=axes, x=norm_to_sc['T'], y=norm_to_sc['R'], model=fit_result, model_function=model_func, xscale=xscale, yscale=yscale)
    axes = ivp.rt_fit_textbox(axes=axes, model=fit_result)
    axes.legend(['SC to N', 'N to SC'], markerscale=6, fontsize=26)
    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + fixed_name + '_' + str(np.round(fixed_value*1e6, 3)) + 'uA'
    #for label in axes.get_xticklabels() + axes.get_yticklabels():
    #    label.set_fontsize(26)
    ivp.save_plot(fig, axes, file_name)

    # Make a plot of N --> SC only
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e3
    sort_key = np.argsort(norm_to_sc['T'])
    normal_to_sc_fit_result = iv_results.FitParameters()
    normal_to_sc_fit_result.right.set_values(fit_result.right.result, fit_result.right.error)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': norm_to_sc['rmsR'][sort_key]*yscale}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'title': 'Channel {}'.format(data_channel) + ' TES Resistance vs Temperature for TES Current = {}'.format(np.round(fixed_value*1e6, 3)) + r'$\mu$' + 'A'
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=norm_to_sc['T'][sort_key], y=norm_to_sc['R'][sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # Let us pad the T values so they are smoooooooth
    model_temperatures = np.linspace(norm_to_sc['T'].min(), 70e-3, 100000)
    axes = ivp.add_model_fits(axes=axes, x=model_temperatures, y=norm_to_sc['R'], model=normal_to_sc_fit_result, model_function=model_func, xscale=xscale, yscale=yscale)
    axes = ivp.rt_fit_textbox(axes=axes, model=normal_to_sc_fit_result)
    # axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + fixed_name + '_' + str(np.round(fixed_value*1e6, 3)) + 'uA_normal_to_sc_only'
    axes.set_xlim((10, 70))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)
    # We can also make plots of alpha = (T0/R0)*dR/dT
    # use model values
    #T = np.linspace(norm_to_sc['T'].min(), norm_to_sc['T'].max(), 10000)
    #R = model_func(T, *normal_to_sc_fit_result.right.result)
    R = norm_to_sc['R']
    T = norm_to_sc['T']
    sort_key = np.argsort(T)
    dR_dT = np.gradient(R, T, edge_order=2)
    # print('The input to the model for dR_dT would be: {}'.format(normal_to_sc_fit_result.right.result))
    # dR_dT = fitfuncs.dtanh_tc(T, *normal_to_sc_fit_result.right.result)
    alpha = (T/R) * dR_dT
    #alpha[alpha < 0] = 0
    #alpha = np.gradient(np.log(R), np.log(T), edge_order=2)
    cutR = R > 40e-3
    print('The largest alpha = {} at T = {} mK'.format(np.nanmax(alpha[cutR]), T[np.nanargmax(alpha[cutR])]))
    # Use a model function as well!
    model_T = np.linspace(norm_to_sc['T'].min(), norm_to_sc['T'].max(), 100000)
    model_R = model_func(model_T, *normal_to_sc_fit_result.right.result)
    model_dR_dT = fitfuncs.dtanh_tc(model_T, *normal_to_sc_fit_result.right.result)
    model_alpha = (model_T/model_R)*model_dR_dT
    #model_alpha = (model_T/model_R)*np.gradient(model_R, model_T, edge_order=2)
    model_cutR = model_R > 40e-3
    # Remove it
    #alpha[np.nanargmax(alpha)] = 0
    #print('The largest alpha is now = {} at T = {} mK'.format(np.nanmax(alpha), T[np.nanargmax(alpha)]))
    # make plot
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1
    yscale = 1
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES ' + r'$\alpha$',
                    'logy': 'linear',
                    'xlim': (0, 0.700*xscale),
                    'ylim': (0, model_alpha[model_cutR].max()),
                    'title': 'Channel {}'.format(data_channel) + ' TES ' + r'$\alpha$' + ' vs Temperature for TES Current = {}'.format(np.round(fixed_value*1e6, 3)) + r'$\mu$' + 'A'
                    }
    params = {'marker': 'o', 'markersize': 5, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': None}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=R[cutR], y=alpha[cutR], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # axes2 = axes.twinx()
    params = {'marker': 'None', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': '-',
              'xerr': None, 'yerr': None}
    ivp.generic_fitplot_with_errors(axes=axes, x=model_R[model_cutR], y=model_alpha[model_cutR], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # Let us pad the T values so they are smoooooooth
    # model_temperatures = np.linspace(norm_to_sc['T'].min(), 70e-3, 100000)
    #axes = ivp.add_model_fits(axes=axes, x=model_temperatures, y=norm_to_sc['R'], model=normal_to_sc_fit_result, model_function=model_func, xscale=xscale, yscale=yscale)
    #axes = ivp.rt_fit_textbox(axes=axes, model=normal_to_sc_fit_result)
    # axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'alpha_vs_T_ch_' + str(data_channel) + '_fixed_' + fixed_name + '_' + str(np.round(fixed_value*1e6, 3)) + 'uA_normal_to_sc_only'
    #axes.set_xlim((10, 70))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)
    return True


def tes_plots(output_path, data_channel, squid, temperature, data, fit_parameters):
    '''Helper function to generate standard TES plots
    iTES vs vTES
    rTES vs iTES
    rTES vs vTES
    rTES vs iBias
    pTES vs rTES
    pTES vs vTES
    '''
    # Current vs Voltage
    plot_current_vs_voltage(output_path, data_channel, squid, temperature, data, fit_parameters)
    # Resistance vs Current
    plot_resistance_vs_current(output_path, data_channel, temperature, data, fit_parameters)
    # Resistance vs Voltage
    plot_resistance_vs_voltage(output_path, data_channel, temperature, data, fit_parameters)
    # Resistance vs Bias Current
    plot_resistance_vs_bias_current(output_path, data_channel, temperature, data, fit_parameters)
    # Power vs rTES
    plot_power_vs_resistance(output_path, data_channel, temperature, data, fit_parameters)
    # Power vs vTES
    plot_power_vs_voltage(output_path, data_channel, temperature, data, fit_parameters)
    return True


def make_tes_plots(output_path, data_channel, squid, iv_dictionary, fit_dictionary, individual=False):
    '''Loop through data to generate TES specific plots'''

    if individual is True:
        for temperature, data in iv_dictionary.items():
            tes_plots(output_path, data_channel, squid, temperature, data, fit_dictionary[temperature])
    # Make a for all temperatures here
    make_tes_multiplot(output_path=output_path, data_channel=data_channel, squid=squid, iv_dictionary=iv_dictionary, fit_parameters=fit_dictionary)
    return True


def get_i_tes(vout, r_fb, m_ratio):
    '''Computes the TES current and TES current RMS in Amps'''
    ites = vout/(r_fb*m_ratio)
    return ites


def get_i_tes_rms(vrms, r_fb, m_ratio):
    '''Computes the TES current and TES current RMS in Amps'''
    ites_rms = np.abs(vrms/(r_fb*m_ratio))
    return ites_rms


def get_r_tes_alt(ites, vbias, r_bias, r_sh, r_p):
    '''A hack method to try and get rTES'''
    # Anywhere there is an Rbias replace with ?
    rtes = (1/(r_bias + r_sh + r_p))*(vbias*(r_sh+r_p)/ites - r_bias*(r_sh+r_p))
    return rtes


def get_r_tes_new(ibias, ites, r_sh, r_p):
    '''Compute rTES directly from currents'''
    # 1. ish = ibias - iTES
    # 2. ites*rp + ites*rtes = ish*rsh
    # 3. rtes = (ish*rsh - ites*rp)/ites
    # 4. rtes = (ibias*rsh - ites*rsh - ites*rp)/ites
    # 5. rtes = (ibias/ites)*rsh - rsh - rp
    rtes = (ibias/ites)*r_sh - r_sh - r_p
    return rtes

def get_r_tes_rms_new(ibias, ibias_rms, ites, ites_rms, r_p_err, r_sh, rtes):
    '''Compute the rTES rms value'''
    # Since we assume the error in rsh is constant it won't matter. But r_p err will
    # use the general rule of R_err/R = sqrt(sum((dR/dxi * xi_err)^2))
    # first dR/diBias, then dR/dites, and then dR/dr_p
    dr_dibias = r_sh/ites
    dr_dites = -(ibias/(ites*ites))*r_sh
    dr_drp = -1
    rtes_rms = np.sqrt(np.power(dr_dibias*ibias_rms, 2) + np.power(dr_dites*ites_rms, 2) +np.power(dr_drp*r_p_err, 2))
    return rtes_rms


def get_v_tes_new(ites, rtes):
    '''A hack method'''
    vtes = ites*rtes
    return vtes


def get_v_tes_rms_new(ites, ites_rms, rtes, rtes_rms, vtes):
    '''Compute rms on vtes from ites and rtes'''
    vtes_rms = np.abs(vtes)*np.sqrt(np.power(ites_rms/ites, 2) + np.power(rtes_rms/rtes, 2))
    return vtes_rms


def get_r_tes(ites, vtes):
    '''Computes the TES resistance in Ohms'''
    rtes = vtes/ites
    return rtes


def get_r_tes_rms(ites, ites_rms, vtes, vtes_rms):
    '''Comptues the RMS on the TES resistance in Ohms'''
    # Fundamentally this is resistance = a*iBias/vOut - b
    # a = r_sh*r_fb*M
    # b = r_sh
    # dR/di_bias = a/vOut
    # dR/dVout = -a*iBias/vOut^2
    # rTES_rms = npsqrt( (dR/di_bias * iBiasRms)^2 + (dR/dVout * vOutRms)^2 )
    d_r = npsqrt(pow2(vtes_rms/ites) + pow2(-1*vtes*ites_rms/pow2(ites)))
    return d_r


def get_p_tes(ites, vtes):
    '''Compute the TES power dissipation (Joule)'''
    ptes = ites*vtes
    return ptes


def get_p_tes_rms(ites, ites_rms, vtes, vtes_rms):
    '''Computes the RMS on the TES (Joule) power dissipation'''
    d_p = npsqrt(pow2(ites*vtes_rms) + pow2(vtes*ites_rms))
    return d_p


def get_v_tes(i_bias, v_out, r_fb, m_ratio, r_sh, r_p):
    '''computes the TES voltage in Volts
    ish*rsh = ites*(rp+rtes)
    ibas*rsh - ites*rsh = ites(rp+rtes)
    ibaias*rsh = ites*(rp+rtes+rsh)
    vTES = vSh - vPara
    vTES = r_sh*(iSh) - r_p*iTES
    vTES = r_sh*(iBias - iTES) - r_p*iTES = r_sh*iBias - iTES*(r_p+r_sh)
    '''
    v_tes = i_bias*r_sh - (v_out/(m_ratio*r_fb))*r_sh - (v_out/(m_ratio*r_fb))*r_p
    # Simple model
    # vTES = r_sh*(iBias - vOut/M/r_fb)
    return v_tes


def get_v_tes_rms(i_bias_rms, v_out, v_out_rms, r_fb, m_ratio, r_sh, r_p, r_p_err):
    '''compute the RMS on the TES voltage in Volts'''
    # Fundamentally this is V = r_sh*iBias - r_p/(Mr_fb)*vOut - r_sh/(Mr_fb)*vOut
    # errV**2 = (dV/di_bias * erriBias)**2 + (dV/dvOut * errvOut)**2 + (dV/dr_p * errr_p)**2
    # a = r_sh
    # b = r_sh/Rf/M
    # dV/di_bias = a
    # dV/dVout = -b
    # So this does as dV = npsqrt((dV/dIbias * iBiasRMS)^2 + (dV/dVout * vOutRMS)^2)
    d_v = npsqrt(pow2(r_sh*i_bias_rms) + pow2(((r_p+r_sh)/(m_ratio*r_fb))*v_out_rms) + pow2((v_out/(m_ratio*r_fb))*r_p_err))
    return d_v


def get_tes_fits(tes_values):
    '''Function to get the TES fit parameters for the normal branches and superconducting branch'''
    # TODO: determine if we want to get slopes of V = m*I+ b or I = m*V + b and if we want to plot I vs V or V vs I
    # First umpack. These are sorted by vTES value
    i_tes, i_tes_rms, v_tes, v_tes_rms, r_tes, r_tes_rms = tes_values

    # Get the superconducting branch first. This will tell us what the parasitic series resistance is
    # And will let us knwo the correction shift needed so iTES = 0 when vTES = 0
    # Get SC branch
    (event_left, event_right) = walk_sc(v_tes, i_tes)
    # From here let us make a fit of the form y = m*x+b where y = vTES and x = iTES
    # Here m = Resistance and b = voltage offset
    # So when iTES = 0, vTES = b
    # NOTE: This follows Ohm's law but our plotting plane is inverted
    result, pcov = curve_fit(fitfuncs.lin_sq, i_tes[event_left:event_right], v_tes[event_left:event_right], sigma=v_tes_rms[event_left:event_right], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    v_fit = fitfuncs.lin_sq(i_tes, result[0], result[1])
    v_fit_sc = {'result': result, 'perr': perr, 'model': v_fit*1e6}
    # Get the left side normal branch first
    lev = walk_normal(v_tes, i_tes, 'left')
    # Model is vTES = m*iTES + b
    result, pcov = curve_fit(fitfuncs.lin_sq, i_tes[0:lev], v_tes[0:lev], sigma=v_tes_rms[0:lev], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    v_fit = fitfuncs.lin_sq(i_tes, result[0], result[1])
    v_fit_left = {'result': result, 'perr': perr, 'model': v_fit*1e6}

    rev = walk_normal(v_tes, i_tes, 'right')
    result, pcov = curve_fit(fitfuncs.lin_sq, i_tes[rev:], v_tes[rev:], sigma=v_tes_rms[rev:], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    v_fit = fitfuncs.lin_sq(i_tes, result[0], result[1])
    v_fit_right = {'result': result, 'perr': perr, 'model': v_fit*1e6}

    # Adjust data based on intersection of SC and Normal data
    # V = Rn*I + Bn
    # V = Rs*I + Bs
    # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
    current_intersection = (v_fit_sc['result'][1] - v_fit_left['result'][1])/(v_fit_left['result'][0] - v_fit_sc['result'][0])
    voltage_intersection = v_fit_sc['result'][0]*current_intersection + v_fit_sc['result'][1]
    print('The current and voltage intersections are {} uA and {} uV'.format(current_intersection*1e6, voltage_intersection*1e6))
    i_tes = i_tes - current_intersection
    v_tes = v_tes - voltage_intersection

    # Redo walks
    (event_left, event_right) = walk_sc(v_tes, i_tes)
    # From here let us make a fit of the form y = m*x+b where y = vTES and x = iTES
    # Here m = Resistance and b = voltage offset
    # So when iTES = 0, vTES = b
    # NOTE: This follows Ohm's law but our plotting plane is inverted
    result, pcov = curve_fit(fitfuncs.lin_sq, i_tes[event_left:event_right], v_tes[event_left:event_right], sigma=v_tes_rms[event_left:event_right], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    v_fit = fitfuncs.lin_sq(i_tes, result[0], result[1])
    v_fit_sc = {'result': result, 'perr': perr, 'model': v_fit*1e6}
    # Get the left side normal branch first
    lev = walk_normal(v_tes, i_tes, 'left')
    # Model is vTES = m*iTES + b
    result, pcov = curve_fit(fitfuncs.lin_sq, i_tes[0:lev], v_tes[0:lev], sigma=v_tes_rms[0:lev], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    v_fit = fitfuncs.lin_sq(i_tes, result[0], result[1])
    v_fit_left = {'result': result, 'perr': perr, 'model': v_fit*1e6}

    rev = walk_normal(v_tes, i_tes, 'right')
    result, pcov = curve_fit(fitfuncs.lin_sq, i_tes[rev:], v_tes[rev:], sigma=v_tes_rms[rev:], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    v_fit = fitfuncs.lin_sq(i_tes, result[0], result[1])
    v_fit_right = {'result': result, 'perr': perr, 'model': v_fit*1e6}
    # Finally also recompute rTES
    r_tes = v_tes/i_tes
    return v_fit_left, v_fit_sc, v_fit_right, [i_tes, i_tes_rms, v_tes, v_tes_rms, r_tes, r_tes_rms]


def dump2text(resistance, temperature, file_name):
    '''Quick function to dump R and T values to a text file'''
    print('The shape of R and T are: {0} and {1}'.format(resistance.shape, temperature.shape))
    np.savetxt(file_name, np.stack((resistance, temperature), axis=1), fmt='%12.10f')
    return True


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
        r_data = readROOT(input_path, tree, branches, method)
        # Make life easier:
        r_data = r_data['data']
    else:
        chlist = 'ChList'
        channels = readROOT(input_path, None, None, method='single', tobject=chlist)
        channels = channels['data'][chlist]
        branches = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', thermometer_name] + ['Waveform' + '{:03d}'.format(int(i)) for i in channels]
        print('Branches to be read are: {}'.format(branches))
        tree = 'data_tree'
        method = 'chain'
        r_data = readROOT(input_path, tree, branches, method)
        r_data = r_data['data']
        r_data['Channel'] = channels
    return r_data


def format_iv_data(iv_data, output_path, new_format=False, number_of_windows=1, thermometer='EP'):
    '''Format the IV data into easy to use forms'''
    #graphviz = GraphvizOutput()
    #graphviz.output_file = output_path + '/' + 'basic.png'

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
    if thermometer == 'EP':
        temperatures = iv_data['EPCal_K']
    else:
        temperatures = iv_data['NT']
    time_values = iv_data['Timestamp_s'] + iv_data['Timestamp_mus']/1e6
    cut = temperatures >= -1
    # Get unique time values for valid temperatures
    time_values, idx = np.unique(time_values[cut], return_index=True)
    # Reshape temperatures to be only the valid values that correspond to unique times
    temperatures = temperatures[cut]
    temperatures = temperatures[idx]
    ivp.test_plot(time_values, temperatures, 'Unix Time', 'Temperature [K]', output_path + '/' + 'quick_look_Tvt.png')
    print('Processing IV waveforms...')
    # Next process waveforms into input and output arrays

    # Ultimately let us collapse the finely sampled waveforms into more coarse waveforms.
    mean_waveforms = {}
    rms_waveforms = {}
    #processed_waveforms = {}
    # How we process depends upon the format
    if new_format is False:
        waveforms = {ch: {} for ch in np.unique(iv_data['Channel'])}
        num_channels = np.unique(iv_data['Channel']).size
        for event, channel in enumerate(iv_data['Channel'][cut]):
            waveforms[channel][event//num_channels] = iv_data['Waveform'][event]
        # We now have a nested dictionary of waveforms.
        # waveforms[ch] consists of a dictionary of Nevents keys (actual events)
        # So waveforms[ch][ev] will be a numpy array with size = NumberOfSamples.
        # The timestamp of waveforms[ch][ev][sample] is time_values[ev] + sample*SamplingWidth_s
        for channel in waveforms.keys():
            processed_waveforms = process_waveform(waveforms[channel], time_values, iv_data['SamplingWidth_s'][0], number_of_windows=number_of_windows)
            mean_waveforms[channel], rms_waveforms[channel], mean_time_values = processed_waveforms.values()
            #  mean_waveforms[channel], rms_waveforms[channel], mean_time_values = process_waveform(waveforms[channel], time_values, iv_data['SamplingWidth_s'][0], number_of_windows=number_of_windows)
    else:
        for channel in iv_data['Channel']:
            print('Starting processing...')
            st = time.time()
            #with PyCallGraph(output=graphviz, config=Config(max_depth=1000000, include_stdlib=True)):
            processed_waveforms = process_waveform(iv_data['Waveform' + '{:03d}'.format(int(channel))], time_values, iv_data['SamplingWidth_s'][0], number_of_windows=number_of_windows)
            print('Process function took: {} s to run'.format(time.time() - st))
            mean_waveforms[channel], rms_waveforms[channel], mean_time_values = processed_waveforms.values()
            # mean_waveforms[channel], rms_waveforms[channel], mean_time_values = process_waveform(iv_data['Waveform' + '{:03d}'.format(int(channel))], time_values, iv_data['SamplingWidth_s'][0], number_of_windows=number_of_windows)
    print('The number of things in the mean time values are: {} and the number of waveforms are: {}'.format(np.size(mean_time_values), len(mean_waveforms[iv_data['Channel'][0]])))
    formatted_data = {'time_values': time_values,
                      'temperatures': temperatures,
                      'mean_waveforms': mean_waveforms,
                      'rms_waveforms': rms_waveforms,
                      'mean_time_values': mean_time_values
                     }
    return formatted_data


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


def parse_temperature_steps(output_path, time_values, temperatures, pid_log, tz_correction):
    '''Run through the PID log and parse temperature steps
    The PID log has as the first column the timestamp a PID setting STARTS
    The second column is the power or temperature setting point
    '''
    times = pan.read_csv(pid_log, delimiter='\t', header=None)
    times = times.values[:, 0]
    times = times + tz_correction  # adjust for any timezone issues
    # Each index of times is now the starting time of a temperature step. Include an appropriate offset for mean computation BUT only a softer one for time boundaries
    # time_list is a list of tuples.
    time_list = []
    start_offset = 1*60
    end_offset = 45
    if times.size > 1:
        for index in range(times.size - 1):
            cut = np.logical_and(time_values > times[index]+start_offset, time_values < times[index+1]-end_offset)
            mean_temperature = np.mean(temperatures[cut])
            start_time = times[index]+start_offset
            stop_time = times[index+1]-end_offset
            time_list.append((start_time, stop_time, mean_temperature))
        # Handle the last step
        # How long was the previous step?
        d_t = time_list[0][1] - time_list[0][0]
        start_time = time_list[-1][1] + start_offset
        end_time = start_time + d_t - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mean_temperature = np.mean(temperatures[cut])
        time_list.append((start_time, end_time, mean_temperature))
        d_t = time_values-time_values[0]
    else:
        # Only 1 temperature step defined
        start_time = times[0] + start_offset
        end_time = time_values[-1] - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mean_temperature = np.mean(temperatures[cut])
        time_list.append((start_time, end_time, mean_temperature))
        d_t = time_values-time_values[0]
    ivp.test_steps(d_t, temperatures, time_list, time_values[0], 'Time', 'T', output_path + '/' + 'test_temperature_steps.png')
    return time_list


def find_temperature_steps(output_path, time_values, temperatures, thermometer='EP'):
    ''' Given an array of time and temperatures identify temperature steps and return the start, stop and mean Temp
    Will return these as a list of tuples
    '''
    # At this point we should only process data further based on unique IV time blocks so here we must find
    # the list of time tuples (start_time, end_time, tTemp) that define a particular IV-curve temperature set

    # First construct d_temperature/d_t and set time values to start at 0
    d_temperature = np.gradient(temperatures, time_values)
    d_t = time_values-time_values[0]
    # Construct a diagnostic plot
    cut = np.logical_and(d_t > 2000, d_t < 4000)
    ivp.test_plot(d_t[cut], temperatures[cut], 'Time', 'T', output_path + '/' + 'test_Tbyt.png')
    ivp.test_plot(d_t[cut], d_temperature[cut], 'Time', 'd_temperature/d_t', output_path + '/' + 'test_d_temperaturebyd_t.png')
    # Now set some parameters for stablized temperatures
    # Define temperature steps to be larger than temperature_step Kelvins
    # Define the rolling windows to contain buffer_length entries
    temp_sensor = 'EP_Cal' if thermometer == 'EP' else 'NT'
    if temp_sensor == 'EP_Cal':
        temperature_step = 5e-5
        buffer_length = 10
    elif temp_sensor == 'NT':
        # Usually is noisy
        temperature_step = 7e-4
        buffer_length = 400
    print('Attempting to find temperature steps now...')
    time_list = get_stable_temperature_steps(time_values, temperatures, buffer_length, temperature_step)
    ivp.test_steps(d_t, temperatures, time_list, time_values[0], 'Time', 'T', output_path + '/' + 'test_temperature_steps.png')
    return time_list


def get_pyiv_data(input_path, output_path, new_format=True, number_of_windows=1, thermometer='EP'):
    '''Function to gather iv data in correct format
    Returns time values, temperatures, mean waveforms, rms waveforms and the list of times for temperature jumps
    '''
    iv_data = get_iv_data_from_file(input_path, new_format=new_format, thermometer=thermometer)
    formatted_data = format_iv_data(iv_data, output_path, new_format=new_format, number_of_windows=number_of_windows, thermometer=thermometer)
    return formatted_data


def fit_sc_branch(xdata, ydata, sigma_y, plane):
    '''Walk and fit the superconducting branch
    In the vOut vs iBias plane x = iBias, y = vOut --> dy/dx ~ resistance
    In the iTES vs vTES plane x = vTES, y = iTES --> dy/dx ~ 1/resistance
    '''
    # First generate a sort_key since dy/dx will require us to be sorted
    sort_key = np.argsort(xdata)
    (event_left, event_right) = walk_sc(xdata[sort_key], ydata[sort_key], plane=plane)
    print('SC fit gives event_left={} and event_right={}'.format(event_left, event_right))
    print('Diagnostics: The input into curve_fit is as follows:')
    print('\txdata size: {}, ydata size: {}, xdata NaN: {}, ydata NaN: {}'.format(xdata[sort_key][event_left:event_right].size, ydata[sort_key][event_left:event_right].size, nsum(np.isnan(xdata[sort_key][event_left:event_right])), nsum(np.isnan(ydata[sort_key][event_left:event_right]))))
    xvalues = xdata[sort_key][event_left:event_right]
    yvalues = ydata[sort_key][event_left:event_right]
    ysigma = sigma_y[sort_key][event_left:event_right]
    print('The values of x, y, and sigmaY are: {} and {} and {}'.format(xvalues, yvalues, ysigma))
    result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, sigma=ysigma, absolute_sigma=True, method='trf')
    # result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, p0=(38, 0), method='trf')
    perr = np.sqrt(np.diag(pcov))
    # In order to properly plot the superconducting branch fit try to find the boundaries of the SC region
    # One possibility is that the region has the smallest and largest y-value excursions. However this may not be the case
    # and certainly unless the data is sorted these indices are meaningless to use in a slice
    #index_y_min = np.argmin(y)
    #index_y_max = np.argmax(y)
    return result, perr #, (index_y_max, index_y_min)


def fit_normal_branches(xdata, ydata, sigma_y):
    '''Walk and fit the normal branches in the vOut vs iBias plane.'''
    # Generate a sort_key since dy/dx must be sorted
    sort_key = np.argsort(xdata)
    # Get the left side normal branch first
    left_ev = walk_normal(xdata[sort_key], ydata[sort_key], 'left')
    xvalues = xdata[sort_key][0:left_ev]
    yvalues = ydata[sort_key][0:left_ev]
    ysigmas = sigma_y[sort_key][0:left_ev]
    # cut = ysigmas > 0
    left_result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, sigma=ysigmas, absolute_sigma=True, p0=(2, 0), method='trf')
    left_perr = npsqrt(np.diag(pcov))
    # Now get the other branch
    right_ev = walk_normal(xdata[sort_key], ydata[sort_key], 'right')
    xvalues = xdata[sort_key][right_ev:]
    yvalues = ydata[sort_key][right_ev:]
    ysigmas = sigma_y[sort_key][right_ev:]
    # cut = ysigmas > 0
    right_result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, sigma=ysigmas, absolute_sigma=True, p0=(2, 0), method='trf')
    right_perr = np.sqrt(np.diag(pcov))
    return left_result, left_perr, right_result, right_perr


def correct_offsets(fit_params, iv_data, branch='normal'):
    ''' Based on the fit parameters for the normal and superconduting branch correct the offset'''
    # Adjust data based on intersection of SC and Normal data
    # V = Rn*I + Bn
    # V = Rs*I + Bs
    # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
    # This won't work if the lines are basically the same so let's detect if the sc and normal branch results roughly the same slope.
    # Recall that the slope of the fit is very big for a superconducting region.
    m_sc = fit_params.sc.result[0]
    m_right = fit_params.right.result[0]
    if np.abs((m_sc - m_right)/(m_sc)) < 0.5:
        print("Slopes are similar enough so try to impose a symmetrical shift")
        vmax = iv_data['vOut'].max()
        vmin = iv_data['vOut'].min()
        imax = iv_data['iBias'].max()
        imin = iv_data['iBias'].min()
        # TODO: FIX THIS
        current_intersection = (imax + imin)/2
        voltage_intersection = (vmax + vmin)/2
    else:
        #current_intersection = (fit_params.sc.result[1] - fit_params.left.result[1])/(fit_params.left.result[0] - fit_params.sc.result[0])
        #voltage_intersection = fit_params.sc.result[0]*current_intersection + fit_params.sc.result[1]
        vmax = iv_data['vOut'].max()
        vmin = iv_data['vOut'].min()
        imax = iv_data['iBias'].max()
        imin = iv_data['iBias'].min()
        # The below only works IF the data really truly is symmetric
        #current_intersection = (imax + imin)/2
        #voltage_intersection = (vmax + vmin)/2
        # Try this by selecting a specific point instead
        if branch == 'normal':
            idx_min = np.argmin(iv_data['iBias'])
            idx_max = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            voltage_intersection = (iv_data['vOut'][idx_max] + iv_data['vOut'][idx_min])/2
        elif branch == 'sc':
            idx_min = np.argmin(iv_data['vOut'])
            idx_max = np.argmax(iv_data['vOut'])
            current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            voltage_intersection = (iv_data['vOut'][idx_max] + iv_data['vOut'][idx_min])/2
        elif branch == 'right':
            current_intersection = (fit_params.sc.result[1] - fit_params.right.result[1])/(fit_params.right.result[0] - fit_params.sc.result[0])
            voltage_intersection = fit_params.sc.result[0]*current_intersection + fit_params.sc.result[1]
        elif branch == 'left':
            current_intersection = (fit_params.sc.result[1] - fit_params.left.result[1])/(fit_params.left.result[0] - fit_params.sc.result[0])
            voltage_intersection = fit_params.sc.result[0]*current_intersection + fit_params.sc.result[1]
        elif branch == 'interceptbalance':
            idx_min = np.argmin(iv_data['iBias'])
            idx_max = np.argmax(iv_data['iBias'])
            # balance current
            #current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            current_intersection = 0
            # balance y-intercepts
            voltage_intersection = (fit_params.left.result[1] + fit_params.right.result[1])/2
        elif branch == 'normal_current_sc_offset':
            idx_min = np.argmin(iv_data['iBias'])
            idx_max = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            voltage_intersection = fit_params.sc.result[1]
        elif branch == 'sc_current_normal_voltage':
            # Correct offset in current based on symmetrizing the SC region and correct the voltage by symmetrizing the normals
            # NOTE: THIS IS BAD
            idx_min = np.argmin(iv_data['vOut'])
            idx_max = np.argmax(iv_data['vOut'])
            current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            idx_min = np.argmin(iv_data['iBias'])
            idx_max = np.argmax(iv_data['iBias'])
            voltage_intersection = (iv_data['vOut'][idx_max] + iv_data['vOut'][idx_min])/2
        elif branch == 'sc_voltage_normal_current':
            # Correct offset in current based on symmetrizing the normal region and correct the voltage by symmetrizing the SC
            # THIS IS BAD
            idx_min = np.argmin(iv_data['iBias'])
            idx_max = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            idx_min = np.argmin(iv_data['vOut'])
            idx_max = np.argmax(iv_data['vOut'])
            voltage_intersection = (iv_data['vOut'][idx_max] + iv_data['vOut'][idx_min])/2
        elif branch == 'None':
            current_intersection = 0
            voltage_intersection = 0
        elif branch == 'dual':
            # Do both left and right inercept matches but take mean of the offset pairs.
            # Get Right
            right_current_intersection = (fit_params.sc.result[1] - fit_params.right.result[1])/(fit_params.right.result[0] - fit_params.sc.result[0])
            right_voltage_intersection = fit_params.sc.result[0]*right_current_intersection + fit_params.sc.result[1]
            # Do left
            left_current_intersection = (fit_params.sc.result[1] - fit_params.left.result[1])/(fit_params.left.result[0] - fit_params.sc.result[0])
            left_voltage_intersection = fit_params.sc.result[0]*left_current_intersection + fit_params.sc.result[1]
            # Compute mean
            current_intersection = (right_current_intersection + left_current_intersection)/2
            voltage_intersection = (right_voltage_intersection + left_voltage_intersection)/2
        elif branch == 'normal_bias_symmetric_normal_offset_voltage':
            idx_min = np.argmin(iv_data['iBias'])
            idx_max = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            right_current_intersection = (fit_params.sc.result[1] - fit_params.right.result[1])/(fit_params.right.result[0] - fit_params.sc.result[0])
            right_voltage_intersection = fit_params.sc.result[0]*right_current_intersection + fit_params.sc.result[1]
            # Do left
            left_current_intersection = (fit_params.sc.result[1] - fit_params.left.result[1])/(fit_params.left.result[0] - fit_params.sc.result[0])
            left_voltage_intersection = fit_params.sc.result[0]*left_current_intersection + fit_params.sc.result[1]
            voltage_intersection = (right_voltage_intersection + left_voltage_intersection)/2
        elif branch == 'normal_bias_symmetric_only':
            idx_min = np.argmin(iv_data['iBias'])
            idx_max = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_max] + iv_data['iBias'][idx_min])/2
            voltage_intersection = 0
        elif branch == 'dual_intersect_voltage_only':
            right_current_intersection = (fit_params.sc.result[1] - fit_params.right.result[1])/(fit_params.right.result[0] - fit_params.sc.result[0])
            right_voltage_intersection = fit_params.sc.result[0]*right_current_intersection + fit_params.sc.result[1]
            # Do left
            left_current_intersection = (fit_params.sc.result[1] - fit_params.left.result[1])/(fit_params.left.result[0] - fit_params.sc.result[0])
            left_voltage_intersection = fit_params.sc.result[0]*left_current_intersection + fit_params.sc.result[1]
            voltage_intersection = (right_voltage_intersection + left_voltage_intersection)/2
            current_intersection = 0
    return current_intersection, voltage_intersection


def convert_fit_to_resistance(fit_parameters, squid, fit_type='iv', r_p=None, r_p_rms=None):
    '''Given a iv_results.FitParameters object convert to Resistance and Resistance error iv_resistance.TESResistance objects

    If a parasitic resistance is provided subtract it from the normal and superconducting branches and assign it
    to the parasitic property.

    If no parasitic resistance is provided assume that the superconducting region values are purely parasitic
    and assign the resulting value to both properties.

    '''
    squid_parameters = squid_info.SQUIDParameters(squid)
    r_sh = squid_parameters.Rsh
    m_ratio = squid_parameters.M
    r_fb = squid_parameters.Rfb

    resistance = iv_resistance.TESResistance()
    # The interpretation of the fit parameters depends on what plane we are in
    if fit_type == 'iv':
        # We fit something of the form vOut = a*iBias + b
        r_sc = r_sh * ((m_ratio*r_fb)/fit_parameters.sc.result[0] - 1)
        if r_p is None:
            r_p = r_sc
        else:
            r_sc = r_sc - r_p
        r_sc_rms = np.abs((-1*m_ratio*r_fb*r_sh)/pow2(fit_parameters.sc.result[0]) * fit_parameters.sc.error[0])
        if r_p_rms is None:
            r_p_rms = r_sc_rms
        else:
            r_sc_rms = npsqrt(pow2(r_sc_rms) + pow2(r_p_rms))
        if fit_parameters.left.result is None:
            r_left, r_left_rms = None, None
        else:
            r_left = (m_ratio*r_fb*r_sh)/fit_parameters.left.result[0] - r_sh - r_p
            r_left_rms = npsqrt(pow2(fit_parameters.left.error[0] * (-1*m_ratio*r_fb*r_sh)/pow2(fit_parameters.left.result[0])) + pow2(-1*r_p_rms))
        if fit_parameters.right.result is None:
            r_right, r_right_rms = None, None
        else:
            r_right = (m_ratio*r_fb*r_sh)/fit_parameters.right.result[0] - r_sh - r_p
            r_right_rms = npsqrt(pow2(fit_parameters.right.error[0] * (-1*m_ratio*r_fb*r_sh)/pow2(fit_parameters.right.result[0])) + pow2(-1*r_p_rms))
    elif fit_type == 'tes':
        # Here we fit something of the form iTES = a*vTES + b
        # Fundamentally iTES = vTES/rTES ...
        print('The type of fit_parameters is: {}'.format(type(fit_parameters)))
        print('The fit parameters is: {}'.format(fit_parameters))
        print('The dict object is: {}'.format(vars(fit_parameters)))
        r_sc = 1/fit_parameters.sc.result[0]
        if r_p is None:
            r_p = r_sc
        else:
            r_sc = r_sc - r_p
        r_sc_rms = np.abs((-1*fit_parameters.sc.error[0])/pow2(fit_parameters.sc.result[0]))
        if r_p_rms is None:
            r_p_rms = r_sc_rms
        else:
            r_sc_rms = npsqrt(pow2(r_sc_rms) + pow2(r_p_rms))
        if fit_parameters.left.result is None:
            r_left, r_left_rms = None, None
        else:
            r_left = 1/fit_parameters.left.result[0]
            r_left_rms = np.abs((-1*fit_parameters.left.error[0])/pow2(fit_parameters.left.result[0]))
        if fit_parameters.right.result is None:
            r_right, r_right_rms = None, None
        else:
            r_right = 1/fit_parameters.right.result[0]
            r_right_rms = np.abs((-1*fit_parameters.right.error[0])/pow2(fit_parameters.right.result[0]))
    resistance.parasitic.set_values(r_p, r_p_rms)
    resistance.left.set_values(r_left, r_left_rms)
    resistance.right.set_values(r_right, r_right_rms)
    resistance.sc.set_values(r_sc, r_sc_rms)
    return resistance


def fit_iv_regions(xdata, ydata, sigma_y, plane='iv'):
    '''Fit the iv data regions and extract fit parameters'''

    fit_params = iv_results.FitParameters()
    # We need to walk and fit the superconducting region first since there RTES = 0
    result, perr = fit_sc_branch(xdata, ydata, sigma_y, plane)
    # Now we can fit the rest
    left_result, left_perr, right_result, right_perr = fit_normal_branches(xdata, ydata, sigma_y)
    fit_params.sc.set_values(result, perr)
    fit_params.left.set_values(left_result, left_perr)
    fit_params.right.set_values(right_result, right_perr)
    # TODO: Make sure everything is right here with the equations and error prop.
    return fit_params


def get_parasitic_resistances(iv_dictionary, squid):
    '''Loop through IV data to obtain parasitic series resistance'''
    parasitic_dictionary = {}
    fit_params = iv_results.FitParameters()
    min_temperature = list(iv_dictionary.keys())[np.argmin([float(temperature) for temperature in iv_dictionary.keys()])]
    for temperature, iv_data in sorted(iv_dictionary.items()):
        print('Attempting to fit superconducting branch for temperature: {} mK'.format(temperature))
        result, perr = fit_sc_branch(iv_data['iBias'], iv_data['vOut'], iv_data['vOut_rms'], plane='iv')
        fit_params.sc.set_values(result, perr)
        resistance = convert_fit_to_resistance(fit_params, squid, fit_type='iv')
        parasitic_dictionary[temperature] = resistance.parasitic
    return parasitic_dictionary, min_temperature


def process_iv_curves(output_path, data_channel, squid, iv_curves):
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

    # First we should try to obtain a measure of the parasitic series resistance. This value will be subtracted from
    # subsequent fitted values of the TES resistance
    parasitic_dictionary, min_temperature = get_parasitic_resistances(iv_curves['iv'], squid)
    # Loop through the iv data now and obtain fit parameters and correct alignment
    fit_parameters_dictionary = {}
    for temperature, iv_data in sorted(iv_curves['iv'].items()):
        fit_parameters_dictionary[temperature] = fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigma_y=iv_data['vOut_rms'], plane='iv')
        # Make it pass through zero. Correct offset.
        # i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'interceptbalance')
        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'dual')
        # Manual offset adjustment
        # i_offset, v_offset = [0, 0]
        # i_offset, v_offset = (-1.1104794020729887e-05, 0.010244294053372446)
        # v_offset = fit_parameters_dictionary[temperature].sc.result[1]
        print('The maximum iBias={} and the minimum iBias={} with a total size={}'.format(iv_data['iBias'].max(), iv_data['iBias'].min(), iv_data['iBias'].size))
        print('For temperature {} the normal offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
        iv_data['vOut'] -= v_offset
        iv_data['iBias'] -= i_offset
#        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'sc')
#        #v_offset = fit_parameters_dictionary[temperature].sc.result[1]
#        print('For temperature {} the sc offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
#        iv_data['vOut'] -= v_offset
#        iv_data['iBias'] -= i_offset
        # Re-walk on shifted data
        fit_parameters_dictionary[temperature] = fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigma_y=iv_data['vOut_rms'], plane='iv')
#        # Next shift the voltages
#        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'dual_intersect_voltage_only')
#        # v_offset = fit_parameters_dictionary[temperature].sc.result[1]
#        print('The maximum iBias={} and the minimum iBias={} with a total size={}'.format(iv_data['iBias'].max(), iv_data['iBias'].min(), iv_data['iBias'].size))
#        print('For temperature {} the normal offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
#        iv_data['vOut'] -= v_offset
#        iv_data['iBias'] -= i_offset
#        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'sc')
#        # v_offset = fit_parameters_dictionary[temperature].sc.result[1]
#        print('For temperature {} the sc offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
#        iv_data['vOut'] -= v_offset
#        iv_data['iBias'] -= i_offset
#        # Re-walk on shifted data
#        fit_parameters_dictionary[temperature] = fit_iv_regions(x=iv_data['iBias'], y=iv_data['vOut'], sigma_y=iv_data['vOut_rms'], plane='iv')
    # Next loop through to generate plots
    for temperature, iv_data in sorted(iv_curves['iv'].items()):
        # Make I-V plot
        file_name = output_path + '/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
        plt_data = [iv_data['iBias'], iv_data['vOut'], iv_data['iBias_rms'], iv_data['vOut_rms']]
        axes_options = {'xlabel': 'Bias Current [uA]',
                        'ylabel': 'Output Voltage [mV]',
                        'title': 'Channel {} Output Voltage vs Bias Current for temperatures = {} mK'.format(data_channel, temperature)
                        }
        model_resistance = convert_fit_to_resistance(fit_parameters_dictionary[temperature], squid, fit_type='iv', r_p=parasitic_dictionary[min_temperature].value, r_p_rms=parasitic_dictionary[min_temperature].rms)
        ivp.iv_fitplot(plt_data, fit_parameters_dictionary[temperature], model_resistance, parasitic_dictionary[min_temperature], file_name, axes_options, xscale=1e6, yscale=1e3)
        # Let's make a ROOT style plot (yuck)
        # ivp.make_root_plot(output_path, data_channel, temperature, iv_data, fit_parameters_dictionary[temperature], parasitic_dictionary[min_temperature], xscale=1e6, yscale=1e3)
        iv_curves['iv'] = iv_curves['iv']
        iv_curves['fit_parameters'] = fit_parameters_dictionary
        iv_curves['parasitic'] = parasitic_dictionary
    return iv_curves


def get_power_temperature_curves(output_path, data_channel, iv_dictionary):
    '''Generate a power vs temperature curve for a TES'''
    # Need to select power in the biased region, i.e. where P(R) ~ constant
    # Try something at 0.5*Rn
    temperatures = np.empty(0)
    power = np.empty(0)
    power_rms = np.empty(0)
    iTES = np.empty(0)
    stat_mode = 'direct-serr-mean'
    for temperature, iv_data in iv_dictionary.items():
        # Create cut to select only data going in the Normal to SC mode
        # This happens in situations as follows:
        # if iBias > 0 and iBias is decreasing over time
        # if iBias < 0 and iBias is increasing
        # Basically whenever iBias is approaching 0
        di_bias = np.gradient(iv_data['iBias'], edge_order=2)
        c_normal_to_sc_pos = np.logical_and(iv_data['iBias'] > 0, di_bias < 0)
        c_normal_to_sc_neg = np.logical_and(iv_data['iBias'] <= 0, di_bias > 0)
        c_normal_to_sc = np.logical_or(c_normal_to_sc_pos, c_normal_to_sc_neg)
        # Also select data that is some fraction of the normal resistance, say 20-30%
        #TODO: Make this a parameter
        r_n = 580e-3
        r_0 = 0.5*r_n
        d_r = r_n/6
        cut = np.logical_and(iv_data['rTES'] > r_0 - d_r, iv_data['rTES'] < r_0 + d_r)
        cut = np.logical_and(cut, c_normal_to_sc)
        if nsum(cut) > 0:
            if stat_mode == 'mean':
                temperatures = np.append(temperatures, float(temperature)*1e-3)
                power = np.append(power, np.mean(iv_data['pTES'][cut]))
                iTES = np.append(iTES, np.mean(iv_data['iTES'][cut]))
                # power_rms = np.append(power_rms, np.mean(iv_data['pTES_rms'][cut]))
                power_rms = np.append(power_rms, np.std(iv_data['pTES_rms'][cut]))
            if stat_mode == 'median':
                temperatures = np.append(temperatures, float(temperature)*1e-3)
                power = np.append(power, np.median(iv_data['pTES'][cut]))
                iTES = np.append(iTES, np.mean(iv_data['iTES'][cut]))
                # power_rms = np.append(power_rms, np.mean(iv_data['pTES_rms'][cut]))
                power_rms = np.append(power_rms, mad(iv_data['pTES_rms'][cut]))
            if stat_mode == 'direct':
                temperatures = np.append(temperatures, np.ones(nsum(cut))*float(temperature)*1e-3)
                power = np.append(power, iv_data['pTES'][cut])
                iTES = np.append(iTES, iv_data['iTES'][cut])
                # power_rms = np.append(power_rms, np.mean(iv_data['pTES_rms'][cut]))
                power_rms = np.append(power_rms, iv_data['pTES_rms'][cut])
            if stat_mode == 'direct-fuzzy':
                temperatures = np.append(temperatures, np.random.normal(float(temperature)*1e-3, 0.1e-3, nsum(cut)))
                power = np.append(power, iv_data['pTES'][cut])
                iTES = np.append(iTES, iv_data['iTES'][cut])
                # power_rms = np.append(power_rms, np.mean(iv_data['pTES_rms'][cut]))
                power_rms = np.append(power_rms, iv_data['pTES_rms'][cut])
            if stat_mode == 'direct-median':
                temperatures = np.append(temperatures, float(temperature)*1e-3)
                power = np.append(power, np.median(iv_data['pTES'][cut]))
                iTES = np.append(iTES, np.median(iv_data['iTES'][cut]))
                # power_rms = np.append(power_rms, np.mean(iv_data['pTES_rms'][cut]))
                power_rms = np.append(power_rms, mad(iv_data['pTES'][cut]))
            if stat_mode == 'direct-mean':
                temperatures = np.append(temperatures, float(temperature)*1e-3)
                power = np.append(power, np.mean(iv_data['pTES'][cut]))
                iTES = np.append(iTES, np.mean(iv_data['iTES'][cut]))
                # power_rms = np.append(power_rms, np.mean(iv_data['pTES_rms'][cut]))
                power_rms = np.append(power_rms, np.std(iv_data['pTES'][cut]))
            if stat_mode == 'direct-serr-mean':
                temperatures = np.append(temperatures, float(temperature)*1e-3)
                power = np.append(power, np.mean(iv_data['pTES'][cut]))
                iTES = np.append(iTES, np.mean(iv_data['iTES'][cut]))
                # power_rms = np.append(power_rms, np.mean(iv_data['pTES_rms'][cut]))
                power_rms = np.append(power_rms, np.std(iv_data['pTES'][cut])/np.sqrt(iv_data['pTES'][cut].size))
        else:
            print('For T = {} mK there were no values used.'.format(temperature))
    # print('The main T vector is: {}'.format(temperatures))
    # print('The iTES vector is: {}'.format(iTES))
    #TODO: Make these input values?
    cut_temperature = np.logical_and(temperatures > 35e-3, temperatures < 53e-3)  # This should be the expected Tc
    cut_power = power < 1e-6
    cut_temperature = np.logical_and(cut_temperature, cut_power)
    # Attempt to fit it to a power function
    # [k, n, Ttes, Pp]
    lbounds = [100e-9, 1, 28e-3]
    ubounds = [10e-3, 6, 70e-3]
    # max_nfev=1e4 if using trf
    # (k, n, Ttes, Pp)
    fixedArgs = {'Pp': 0} # Holding P0 as 0 since there is some degeneracy with kTc^n and P0
    x0 = [1000e-9, 5, 55e-3]
    # fitargs = {'p0': x0, 'method': 'lm', 'maxfev': int(5e4)}
    use_sigmas = True
    if use_sigmas:
        # This fitarg will use the errors on y
        fitargs = {'p0': x0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True, 'sigma': power_rms[cut_temperature], 'method': 'trf', 'jac': '3-point', 'tr_solver': 'exact', 'x_scale': 'jac', 'xtol': 1e-15, 'ftol': 1e-15, 'gtol': None, 'loss': 'linear', 'max_nfev': 10000, 'verbose': 2}
    else:
        # This fitarg below will not use the errors on y
        fitargs = {'p0': x0, 'bounds': (lbounds, ubounds), 'method': 'trf', 'jac': '3-point', 'tr_solver': 'exact', 'x_scale': 'jac', 'xtol': 1e-15, 'ftol': 1e-15, 'gtol': None, 'loss': 'linear', 'max_nfev': 10000, 'verbose': 2}
    results, pcov = curve_fit(fitfuncs.tes_power_polynomial_fixed(fixedArgs), temperatures[cut_temperature], power[cut_temperature], **fitargs)
    print('The covariance matrix is: {}'.format(pcov))
    perr = np.sqrt(np.diag(pcov))

    fixedResults = [fixedArgs.get('k'), fixedArgs.get('n'), fixedArgs.get('Ttes'), fixedArgs.get('Pp')]
    results, perr = list(results), list(perr)
    results = [results.pop(0) if item is None else item for item in fixedResults]
    perr = [perr.pop(0) if item is None else 0 for item in fixedResults]

    fixedx0 = [fixedArgs.get('k'), fixedArgs.get('n'), fixedArgs.get('Ttes'), fixedArgs.get('Pp')]
    x0 = [fitargs['p0'].pop(0) if item is None else item for item in fixedx0]
    print('x0={}, results={}'.format(x0, results))
    fit_result = iv_results.FitParameters()
    fit_result.left.set_values(results, perr)
    # fit_result.right.set_values(x0, x0)
    # Next make a P-T plot
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e15
    ymax = power.max()*1.05*yscale
    params = {'marker': 'o', 'markersize': 7, 'markeredgecolor': 'black',
              'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': power_rms*yscale
              }
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Power [fW]',
                    'title': None, # 'Channel {} TES Power vs Temperature'.format(data_channel),
                    'xlim': (25, 60),
                    'ylim': (0, ymax)
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperatures, y=power, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = ivp.add_model_fits(axes=axes, x=temperatures, y=power, model=fit_result, model_function=fitfuncs.tes_power_polynomial, xscale=xscale, yscale=yscale)
    axes = ivp.pt_fit_textbox(axes=axes, model=fit_result)

    file_name = output_path + '/' + 'pTES_vs_T_ch_' + str(data_channel)
    #for label in axes.get_xticklabels() + axes.get_yticklabels():
    #    label.set_fontsize(32)
    ivp.save_plot(fig, axes, file_name, dpi=150)
    print('Results: k = {}, n = {}, Tb = {}, Pp = {}'.format(*results))
    print('Error Results: k = {}, n = {}, Tb = {}, Pp = {}'.format(*perr))
    # Compute G
    # P = k*(Ts^n - T^n)
    # G = n*k*T^(n-1)
    print('G(Ttes) = {} pW/K'.format(results[0]*results[1]*np.power(results[2], results[1]-1)*1e12))
    print('G(10 mK) = {} pW/K'.format(results[0]*results[1]*np.power(10e-3, results[1]-1)*1e12))
    return True


# new function
def get_resistance_temperature_curves_new(output_path, data_channel, iv_dictionary):
    '''Generate resistance vs temperature curves for a TES'''
    # Rtes = R(i,T) so we are really asking for R(i=constant, T).

    # Get only data with a fixed iTES
    fixed_name = 'iTES'
    fixed_value = 0.1e-6
    delta_value = 0.05e-6
    stat_flag = 'direct-serr-mean'
    target = 20
    r_normal = 0.570
    # TODO: See if we can make this..better
    # We need to handle the case of going from SC --> N and N --> SC separately so select cuts for these.
    norm_to_sc = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    sc_to_norm = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    for temperature, iv_data in iv_dictionary.items():
        fixed_cut = np.logical_and(iv_data[fixed_name] > fixed_value - delta_value, iv_data[fixed_name] < fixed_value + delta_value)
        cut_quality = iv_data['rTES_rms'] < 0.5*r_normal
        fixed_cut = np.logical_and(fixed_cut, cut_quality)
        # We need to select the lowest sensible rTES since there are degeneracies in the rTES vs iTES plane
        # Steps:
        #   1.  Examine counts in the biased region RL < R < 0.9 Rn. If there are more than N
        #       events in this region we select them.
        #   2.  If there are less than N events in the proposed biased region, look in the SC region and N regions
        #       This is relying on the fact that, for a given iTES, the only way degeneracies in Rn and Rsc type values
        #       can occur is due to the SC->N and N->SC type directions. Rely on the directionality cut to get rid of them.
        #   3.  Since step 1 is likely to exclude SC->N styles we should pair this ultimately with the direction cuts
        #       first and examine the data for each type of cut.
        ######
        # We need to use the gradient to determine if we go up or down
        dbias = np.gradient(iv_data['iBias'], edge_order=2)
        cut1 = np.logical_and(iv_data['iBias'] > 0, dbias < 0)   # Positive iBias -slope (High to Low, N-->Sc)
        cut2 = np.logical_and(iv_data['iBias'] <= 0, dbias > 0)  # Negative iBias +slope (-High to -Low, N-->Sc)
        cut_norm_to_sc = np.logical_or(cut1, cut2)
        cut_fixed_norm_to_sc = np.logical_and(fixed_cut, cut_norm_to_sc)
        cut_fixed_sc_to_norm = np.logical_and(fixed_cut, ~cut_norm_to_sc)
        # Now implement cut on resistance values for each case. I guess we should use the
        # region with most events?
        frac_rn = 0.8
        rsc_thresh = 100e-3
        cut_R_normal = iv_data['rTES'] > frac_rn*r_normal
        cut_R_biased = np.logical_and(iv_data['rTES'] > rsc_thresh, iv_data['rTES'] < frac_rn*r_normal)
        cut_R_sc = iv_data['rTES'] < rsc_thresh
        if nsum(np.logical_and(fixed_cut, cut_R_normal)) > nsum(np.logical_and(fixed_cut, cut_R_biased)) and nsum(np.logical_and(fixed_cut, cut_R_normal)) > nsum(np.logical_and(fixed_cut, cut_R_sc)):
            cut_fixed_norm_to_sc = np.logical_and(cut_fixed_norm_to_sc, cut_R_normal)
            cut_fixed_sc_to_norm = np.logical_and(cut_fixed_sc_to_norm, cut_R_normal)
        if nsum(np.logical_and(fixed_cut, cut_R_biased)) > nsum(np.logical_and(fixed_cut, cut_R_normal)) and nsum(np.logical_and(fixed_cut, cut_R_normal)) > nsum(np.logical_and(fixed_cut, cut_R_sc)):
            cut_fixed_norm_to_sc = np.logical_and(cut_fixed_norm_to_sc, cut_R_biased)
            cut_fixed_sc_to_norm = np.logical_and(cut_fixed_sc_to_norm, cut_R_biased)
        if nsum(np.logical_and(fixed_cut, cut_R_sc)) > nsum(np.logical_and(fixed_cut, cut_R_biased)) and nsum(np.logical_and(fixed_cut, cut_R_sc)) > nsum(np.logical_and(fixed_cut, cut_R_normal)):
            cut_fixed_norm_to_sc = np.logical_and(cut_fixed_norm_to_sc, cut_R_sc)
            cut_fixed_sc_to_norm = np.logical_and(cut_fixed_sc_to_norm, cut_R_sc)
        ######
#        # Make a cut on rTES too to not select normal branch unless we are normal.
#        rcut = np.logical_and(fixed_cut, iv_data['rTES'] < rn_threshold)
#        print('For T={} mK, sum of rcut is: {}, and sum of ~rcut is: {} and the ratio rcut/len(rcut): {}'.format(temperature, nsum(rcut), nsum(~rcut), nsum(rcut)/rcut.size))
#        if nsum(rcut) >= 40:
#            fixed_cut = rcut
#        else:
#            fixed_cut = np.logical_and(fixed_cut, iv_data['rTES'] > rn_threshold)
#        print('For T={} mK, the sum of the cut is: {}'.format(temperature, nsum(fixed_cut)))
#        # We need to use the gradient to determine if we go up or down
#        dbias = np.gradient(iv_data['iBias'], edge_order=2)
#        cut1 = np.logical_and(iv_data['iBias'] > 0, dbias < 0)   # Positive iBias -slope (High to Low, N-->Sc)
#        cut2 = np.logical_and(iv_data['iBias'] <= 0, dbias > 0)  # Negative iBias +slope (-High to -Low, N-->Sc)
#        cut_norm_to_sc = np.logical_or(cut1, cut2)
#        cut_fixed_norm_to_sc = np.logical_and(fixed_cut, cut_norm_to_sc)
#        cut_fixed_sc_to_norm = np.logical_and(fixed_cut, ~cut_norm_to_sc)
        print('\tTotal sum for N-->SC: {}'.format(nsum(cut_fixed_norm_to_sc)))
        print('\tTotal sum for SC-->N: {}'.format(nsum(cut_fixed_sc_to_norm)))
        if nsum(cut_fixed_norm_to_sc) > 0:
            if stat_flag == 'median':
                norm_to_sc['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
                norm_to_sc['R'] = np.append(norm_to_sc['R'], np.median(iv_data['rTES'][cut_fixed_norm_to_sc]))
                norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], np.median(iv_data['rTES_rms'][cut_fixed_norm_to_sc]))
            if stat_flag == 'mean':
                norm_to_sc['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
                norm_to_sc['R'] = np.append(norm_to_sc['R'], np.mean(iv_data['rTES'][cut_fixed_norm_to_sc]))
                norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], np.mean(iv_data['rTES_rms'][cut_fixed_norm_to_sc]))
            if stat_flag == 'direct-mean':
                norm_to_sc['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
                norm_to_sc['R'] = np.append(norm_to_sc['R'], np.mean(iv_data['rTES'][cut_fixed_norm_to_sc]))
                norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], np.std(iv_data['rTES'][cut_fixed_norm_to_sc]))
            if stat_flag == 'direct-median':
                norm_to_sc['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
                norm_to_sc['R'] = np.append(norm_to_sc['R'], np.median(iv_data['rTES'][cut_fixed_norm_to_sc]))
                norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], mad(iv_data['rTES'][cut_fixed_norm_to_sc]))
            if stat_flag == 'direct':
                # Take raw data points
                norm_to_sc['T'] = np.append(norm_to_sc['T'], np.random.normal(float(temperature)*1e-3, 0.001e-3, nsum(cut_fixed_norm_to_sc)))
                norm_to_sc['R'] = np.append(norm_to_sc['R'], iv_data['rTES'][cut_fixed_norm_to_sc])
                norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], iv_data['rTES_rms'][cut_fixed_norm_to_sc])
            if stat_flag == 'direct-serr-mean':
                # Take raw data points
                norm_to_sc['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
                norm_to_sc['R'] = np.append(norm_to_sc['R'], np.mean(iv_data['rTES'][cut_fixed_norm_to_sc]))
                N = iv_data['rTES_rms'][cut_fixed_norm_to_sc].size
                norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], np.std(iv_data['rTES_rms'][cut_fixed_norm_to_sc])/np.sqrt(N))
        if nsum(cut_fixed_sc_to_norm) > 0:
            sc_to_norm['T'] = np.append(sc_to_norm['T'], float(temperature)*1e-3)
            if stat_flag == 'median':
                sc_to_norm['R'] = np.append(sc_to_norm['R'], np.median(iv_data['rTES'][cut_fixed_sc_to_norm]))
                sc_to_norm['rmsR'] = np.append(sc_to_norm['rmsR'], np.median(iv_data['rTES_rms'][cut_fixed_sc_to_norm]))
            if stat_flag == 'mean':
                sc_to_norm['R'] = np.append(sc_to_norm['R'], np.mean(iv_data['rTES'][cut_fixed_sc_to_norm]))
                sc_to_norm['rmsR'] = np.append(sc_to_norm['rmsR'], np.mean(iv_data['rTES_rms'][cut_fixed_sc_to_norm]))
            if stat_flag == 'direct':
                sc_to_norm['R'] = np.append(sc_to_norm['R'], np.mean(iv_data['rTES'][cut_fixed_sc_to_norm]))
                sc_to_norm['rmsR'] = np.append(sc_to_norm['rmsR'], np.std(iv_data['rTES'][cut_fixed_sc_to_norm]))
            if stat_flag == 'direct-median':
                sc_to_norm['R'] = np.append(sc_to_norm['R'], np.median(iv_data['rTES'][cut_fixed_sc_to_norm]))
                sc_to_norm['rmsR'] = np.append(sc_to_norm['rmsR'], mad(iv_data['rTES'][cut_fixed_sc_to_norm]))
            if stat_flag == 'direct-serr-mean':
                sc_to_norm['R'] = np.append(sc_to_norm['R'], np.mean(iv_data['rTES'][cut_fixed_sc_to_norm]))
                N = iv_data['rTES'][cut_fixed_sc_to_norm].size
                sc_to_norm['rmsR'] = np.append(sc_to_norm['rmsR'], np.std(iv_data['rTES'][cut_fixed_sc_to_norm])/np.sqrt(N))

    # Now we have arrays of R and T for a fixed iTES so try to fit each domain
    # SC --> N first
    # Model function is a modified tanh(Rn, Rp, Tc, Tw)
    model_func = fitfuncs.tanh_tc
    fit_result = iv_results.FitParameters()
    # Try to do a smart Tc0 estimate:
    sort_key = np.argsort(norm_to_sc['T'])
    print('The size of norm_to_sc[R] is: {}, and norm_to_sc[T] is: {} and sort_key is {}'.format(norm_to_sc['R'].size, norm_to_sc['T'].size, sort_key.size))
    T0 = norm_to_sc['T'][sort_key][np.gradient(norm_to_sc['R'][sort_key], norm_to_sc['T'][sort_key], edge_order=2).argmax()]*1.01
    x_0 = [0.7, 0, T0, 1e-3]
    lbounds = (0, 0, 0, 0)
    ubounds = (np.inf, np.inf, np.inf, np.inf)

    print('For SC to N fit initial guess is {}, and the number of data points are: {}'.format(x_0, sc_to_norm['T'].size))
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True, 'sigma': sc_to_norm['rmsR'], 'method': 'trf', 'jac': '3-point', 'xtol': 1e-15, 'ftol': 1e-8, 'loss': 'linear', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, sc_to_norm['T'], sc_to_norm['R'], **fitargs)
    print('The cov matrix is: {}'.format(pcov))
    perr = np.sqrt(np.diag(pcov))
    print('Ascending (SC -> N): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.left.set_values(result, perr)

    # Attempt to fit the N-->Sc region now
    print('For N to SC fit initial guess is {}, and the number of data points are: {}'.format(x_0, norm_to_sc['T'].size))
    #fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True, 'sigma': norm_to_sc['rmsR'], 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'soft_l1', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'linear', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, norm_to_sc['T'], norm_to_sc['R'], **fitargs)
    perr = np.sqrt(np.diag(pcov))
    print('Descending (N -> SC): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.right.set_values(result, perr)
    # Make output plot
    make_resistance_vs_temperature_plots(output_path, data_channel, fixed_name, fixed_value, norm_to_sc, sc_to_norm, model_func, fit_result)
    return True



# old function
def get_resistance_temperature_curves(output_path, data_channel, iv_dictionary):
    '''Generate a resistance vs temperature curve for a TES'''
    # Rtes = R(i,T) really so select a fixed i and across multiple temperatures obtain values for R and then plot
    temperatures = np.empty(0)
    resistance = np.empty(0)
    fixed_value = 'iTES'
    for temperature, iv_data in iv_dictionary.items():
        cut = np.logical_and(iv_data[fixed_value] > 0e-6, iv_data[fixed_value] < 0.3e-6)
        if nsum(cut) > 0:
            temperatures = np.append(temperatures, float(temperature)*1e-3)  # T in K
            resistance = np.append(resistance, np.mean(iv_data['rTES'][cut]))
    # Next make an R-T plot
    # R vs T
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e3
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' +']',
                    'title': 'Channel {} TES Resistance vs Temperature'.format(data_channel)
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperatures, y=resistance, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    #axes.set_ylim((-1,1))
    #axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, sc_bounds=sc_bounds, xscale=xscale, yscale=yscale)
    #axes = ivp.iv_fit_textbox(axes=axes, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)

    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)

    # Make R vs T only for times we are going from higher iBias to lower iBias values
    # One way, assuming noise does not cause overlaps, is to only select points where iBias[i] > iBias[i+1]
    # If I take the diff array I get the following: di_bias[i] = iBias[i] - iBias[i-1]. If di_bias[i] < 0 then iBias is descending
    # So let's use that then.
    ####################################################################################
    # Rtes = R(i,T) really so select a fixed i and across multiple temperatures obtain values for R and then plot
    temperatures = np.empty(0)
    resistance = np.empty(0)
    resistance_right_rms = np.empty(0)
    temperature_desc = np.empty(0)
    resistance_desc = np.empty(0)
    resistance_right_rms_desc = np.empty(0)
    fit_result = iv_results.FitParameters()
    i_select = 0.1e-6
    selector = 'iTES'
    avg_flag = 'median'
    for temperature, iv_data in iv_dictionary.items():
        cut = np.logical_and(iv_data[selector] > i_select - 0.1e-6, iv_data[selector] < i_select + 0.15e-6)
        print('the sum of cut is: {}'.format(nsum(cut)))
        # Cuts to select physical case where we go from Normal --> SC modes
        di_bias = np.gradient(iv_data['iBias'], edge_order=2)
        cut1 = np.logical_and(iv_data['iBias'] > 0, di_bias < 0)
        cut2 = np.logical_and(iv_data['iBias'] <= 0, di_bias > 0)
        dcut = np.logical_or(cut1, cut2)
        cut_desc = np.logical_and(cut, dcut)
        cut_asc = np.logical_and(cut, ~dcut)
        if nsum(cut_asc) > 0:
            temperatures = np.append(temperatures, float(temperature)*1e-3)  # T in K
            if avg_flag == 'median':
                resistance = np.append(resistance, np.median(iv_data['rTES'][cut_asc]))
                resistance_right_rms = np.append(resistance_right_rms, np.median(iv_data['rTES_rms'][cut_asc]))
            if avg_flag == 'mean':
                resistance = np.append(resistance, np.mean(iv_data['rTES'][cut_asc]))
                resistance_right_rms = np.append(resistance_right_rms, np.mean(iv_data['rTES_rms'][cut_asc]))
            # resistance_right_rms = np.append(resistance_right_rms, np.std(iv_data['rTES'][cut_asc]))
        if nsum(cut_desc) > 0:
            temperature_desc = np.append(temperature_desc, float(temperature)*1e-3)  # T in K
            if avg_flag == 'median':
                resistance_desc = np.append(resistance_desc, np.median(iv_data['rTES'][cut_desc]))
                resistance_right_rms_desc = np.append(resistance_right_rms_desc, np.median(iv_data['rTES_rms'][cut_desc]))
            if avg_flag == 'mean':
                resistance_desc = np.append(resistance_desc, np.mean(iv_data['rTES'][cut_desc]))
                resistance_right_rms_desc = np.append(resistance_right_rms_desc, np.mean(iv_data['rTES_rms'][cut_desc]))
            # resistance_right_rms_desc = np.append(resistance_right_rms_desc, np.std(iv_data['rTES'][cut_desc]))
    # Next make an R-T plot
    # Add a T cut?
    # Remove half
#    temperatures = temperatures[T.size//2:-1]
#    resistance = resistance[R.size//2:-1]
#    resistance_right_rms = resistance_right_rms[resistance_right_rms.size//2:-1]
#    temperature_desc = temperature_desc[temperature_desc.size//2:-1]
#    resistance_desc = resistance_desc[resistance_desc.size//2:-1]
#    resistance_right_rms_desc = resistance_right_rms_desc[resistance_right_rms_desc.size//2:-1]
    cut_temperature = temperatures > 8e-3
    cut_temperature_desc = temperature_desc > 8e-3
    temperatures = temperatures[cut_temperature]
    resistance = resistance[cut_temperature]
    resistance_right_rms = resistance_right_rms[cut_temperature]
    temperature_desc = temperature_desc[cut_temperature_desc]
    resistance_desc = resistance_desc[cut_temperature_desc]
    resistance_right_rms_desc = resistance_right_rms_desc[cut_temperature_desc]
    # Try a fit?
    # [Rn, r_p, Tc, Tw]
    # In new fit we have [C, D, B, A] --> A = 1/Tw, B = -Tc/Tw
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e3
    sort_key = np.argsort(temperatures)
    nice_current = np.round(i_select*1e6, 3)
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' +']',
                    'title': 'Channel {}'.format(data_channel) +  ' TES Resistance vs Temperature for TES Current = {}'.format(nice_current)  + r'$\mu$' + 'A'}
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': resistance_right_rms[sort_key]*yscale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperatures[sort_key], y=resistance[sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_descending_iBias_nofit_' + str(np.round(i_select*1e6, 3)) + 'uA'
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)
    sort_key = np.argsort(temperatures)
    x_0 = [1, 0, temperatures[sort_key][np.gradient(resistance[sort_key], temperatures[sort_key], edge_order=2).argmax()]*1.1, 1e-3]
    # x_0 = [1, 0, 50e-3, 1e-3]
    # x_0 = [1, 0, -temperatures[sort_key][np.gradient(resistance[sort_key], temperatures[sort_key], edge_order=2).argmax()]/1e-3,  1/1e-3]
    print('For SN to N fit initial guess is {}'.format(x_0))
    # result, pcov = curve_fit(fitfuncs.tanh_tc, T, R, sigma=resistance_right_rms, absolute_sigma=True, p0=x_0, method='trf')

    result, pcov = curve_fit(fitfuncs.tanh_tc, temperatures, resistance, sigma=resistance_right_rms, absolute_sigma=True, method='trf', max_nfev=5e4)
    perr = np.sqrt(np.diag(pcov))
    print('Ascending (SC -> N): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.left.set_values(result, perr)
    # Try a fit?
    sort_key = np.argsort(temperature_desc)
    x_0 = [1, 0, temperature_desc[sort_key][np.gradient(resistance_desc[sort_key], temperature_desc[sort_key], edge_order=2).argmax()]*1.1, 1e-3]
    # x_0 = [1, 0, 50e-3, 1e-3]
    # x_0 = [1, 0, -temperature_desc[sort_key][np.gradient(resistance_desc[sort_key], temperature_desc[sort_key], edge_order=2).argmax()]/1e-3, 1/1e-3]
    print('For descending fit (N->S) initial guess is {}'.format(x_0))

    result_desc, pcov_desc = curve_fit(fitfuncs.tanh_tc, temperature_desc, resistance_desc, sigma=resistance_right_rms_desc, p0=x_0, absolute_sigma=True, method='lm', maxfev=int(5e4))
    perr_desc = np.sqrt(np.diag(pcov_desc))
    print('Descending (N -> SC): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result_desc]))
    print('Descending Errors (N -> SC): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in perr_desc]))
    fit_result.right.set_values(result_desc, perr_desc)
    # R vs T
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e3
    sort_key = np.argsort(temperatures)
    nice_current = np.round(i_select*1e6, 3)
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'title': 'Channel {}'.format(data_channel) + ' TES Resistance vs Temperature for TES Current = {}'.format(nice_current) + r'$\mu$' + 'A'
                    }
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': resistance_right_rms[sort_key]*yscale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperatures[sort_key], y=resistance[sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    sort_key = np.argsort(temperature_desc)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': resistance_right_rms_desc[sort_key]*yscale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperature_desc[sort_key], y=resistance_desc[sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # axes.set_ylim((-1,1))
    axes = ivp.add_model_fits(axes=axes, x=temperatures, y=resistance, model=fit_result, model_function=fitfuncs.tanh_tc, xscale=xscale, yscale=yscale)
    axes = ivp.rt_fit_textbox(axes=axes, model=fit_result)
    axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + selector + '_' + str(np.round(i_select*1e6, 3)) + 'uA'
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)
    # Make a nicer plot
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e3
    sort_key = np.argsort(temperature_desc)
    normal_to_sc_fit_result = iv_results.FitParameters()
    normal_to_sc_fit_result.right.set_values(result_desc, perr_desc)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': resistance_right_rms_desc[sort_key]*yscale}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'title': 'Channel {}'.format(data_channel) + ' TES Resistance vs Temperature for TES Current = {}'.format(np.round(i_select*1e6, 3)) + r'$\mu$' + 'A'
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperature_desc[sort_key], y=resistance_desc[sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # Let us pad the T values so they are smoooooooth
    model_temperatures = np.linspace(temperature_desc.min(), 70e-3, 100000)
    axes = ivp.add_model_fits(axes=axes, x=model_temperatures, y=resistance, model=normal_to_sc_fit_result, model_function=fitfuncs.tanh_tc, xscale=xscale, yscale=yscale)
    axes = ivp.rt_fit_textbox(axes=axes, model=normal_to_sc_fit_result)
    #axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + selector + '_' + str(np.round(i_select*1e6, 3)) + 'uA_normal_to_sc_only'
    axes.set_xlim((10, 70))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)
    ###############################################################
    # We can try to plot alpha vs R as well why not
    # alpha = To/Ro * dR/d_temperature --> dln(R)/dln(temperatures)
    #alpha = np.gradient(np.log(R), np.log(temperatures), edge_order=2)
    model_temperatures = np.linspace(temperatures.min(), temperatures.max(), 100)
    model_resistance = fitfuncs.tanh_tc(model_temperatures, *fit_result.right.result)
    model_sort_key = np.argsort(model_temperatures)
    # model_alpha = (model_temperatures[model_sort_key]/model_resistance[model_sort_key])*np.gradient(model_resistance[model_sort_key], model_temperatures[model_sort_key], edge_order=1)
    # model_alpha = np.gradient(np.log(model_resistance[model_sort_key]), np.log(model_temperatures[model_sort_key]), edge_order=1)
    model_alpha = (model_temperatures[model_sort_key]/model_resistance[model_sort_key])*np.gradient(model_resistance[model_sort_key], model_temperatures[model_sort_key], edge_order=2)
    print('The max alpha is: {}'.format(np.max(model_alpha)))
    sort_key = np.argsort(temperatures)
    alpha = (temperatures[sort_key]/resistance[sort_key])*np.gradient(resistance[sort_key], temperatures[sort_key], edge_order=1)/1e3
    # alpha = np.gradient(np.log(resistance[sort_key]), np.log(temperatures[sort_key]), edge_order=1)
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': '-', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'ylabel': r'$\alpha$',
                    'title': 'Channel {} TES '.format(data_channel) + r'$\alpha$' + ' vs Resistance'
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=model_resistance[model_sort_key], y=model_alpha, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=resistance[sort_key], y=alpha, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # axes.set_ylim((-1,1))
    # axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, sc_bounds=sc_bounds, xscale=xscale, yscale=yscale)
    # axes = ivp.iv_fit_textbox(axes=axes, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)
    file_name = output_path + '/' + 'alpha_vs_rTES_ch_' + str(data_channel)
    # axes.set_ylim((0,150))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)
    # alpha vs T
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': r'$\alpha$',
                    'title': 'Channel {} TES '.format(data_channel) + r'$\alpha$' +' vs Temperature'
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperatures[sort_key], y=alpha, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # axes.set_ylim((-1,1))
    # axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, sc_bounds=sc_bounds, xscale=xscale, yscale=yscale)
    # axes = ivp.iv_fit_textbox(axes=axes, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)
    file_name = output_path + '/' + 'alpha_vs_T_ch_' + str(data_channel)
    axes.set_ylim((0, 150))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)

    # We can get R-T curves for multiple current selections as well :)
    # Proceed to do 0-1, 1-2, 2-3, up to 9-10
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e3
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': '-', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'title': 'Channel {} TES Resistance vs Temperature'.format(data_channel)
                    }
    for i in range(10):
        temperatures = np.empty(0)
        resistance = np.empty(0)
        for temperature, iv_data in iv_dictionary.items():
            cut = np.logical_and(iv_data['iTES'] > i*1e-6, iv_data['iTES'] < (i+1)*1e-6)  # select 'constant' I0
            # Select normal --> sc transition directions
            di_bias = np.gradient(iv_data['iBias'], edge_order=2)
            cut1 = np.logical_and(iv_data['iBias'] > 0, di_bias < 0)
            cut2 = np.logical_and(iv_data['iBias'] <= 0, di_bias > 0)
            dcut = np.logical_or(cut1, cut2)
            cut = np.logical_and(cut, dcut)
            if nsum(cut) > 0:
                temperatures = np.append(temperatures, float(temperature)*1e-3)
                resistance = np.append(resistance, np.mean(iv_data['rTES'][cut]))
        sort_key = np.argsort(temperatures)
        axes = ivp.generic_fitplot_with_errors(axes=axes, x=temperatures[sort_key], y=resistance[sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'rTES_vs_T_multi_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, file_name)
    return True


def process_tes_curves(iv_curves):
    '''Take TES data and find r_p and Rn values.'''
    fit_dictionary = {}
    for temperature, iv_data in iv_curves['iv'].items():
        print('Processing TES for temperatures = {} mK'.format(temperature))
        fit_params = fit_iv_regions(xdata=iv_data['vTES'], ydata=iv_data['iTES'], sigma_y=iv_data['iTES_rms'], plane='tes')
        fit_dictionary[temperature] = fit_params
    iv_curves['tes_fit_parameters'] = fit_dictionary
    return iv_curves


def get_tes_values(iv_curves, squid):
    '''From I-V data values compute the TES values for iTES and vTES, ultimately yielding rTES'''
    squid_parameters = squid_info.SQUIDParameters(squid)
    r_bias = squid_parameters.Rbias
    r_sh = squid_parameters.Rsh
    m_ratio = squid_parameters.M
    r_fb = squid_parameters.Rfb
    # Test: Select the parasitic resistance from the lowest temperature fit to use for everything
    min_temperature = list(iv_curves['parasitic'].keys())[np.argmin([float(temperature) for temperature in iv_curves['parasitic'].keys()])]
    r_p, r_p_rms = iv_curves['parasitic'][min_temperature].value, iv_curves['parasitic'][min_temperature].rms
    for temperature, iv_data in iv_curves['iv'].items():
        print('Getting TES values for T = {}'.format(temperature))
        iv_data['iTES'] = get_i_tes(iv_data['vOut'], r_fb, m_ratio)
        iv_data['iTES_rms'] = get_i_tes_rms(iv_data['vOut_rms'], r_fb, m_ratio)
        iv_data['rTES'] = get_r_tes_new(iv_data['iBias'], iv_data['iTES'], r_sh, r_p)
        iv_data['vTES'] = get_v_tes_new(iv_data['iTES'], iv_data['rTES'])
        # iv_data['vTES'] = get_v_tes(iv_data['iBias'], iv_data['vOut'], r_fb, m_ratio, r_sh, r_p)
        # iv_data['vTES_rms'] = get_v_tes_rms(iv_data['iBias_rms'], iv_data['vOut'], iv_data['vOut_rms'], r_fb, m_ratio, r_sh, r_p, r_p_rms)
        # iv_data['rTES'] = get_r_tes(iv_data['iTES'], iv_data['vTES'])
        iv_data['rTES_rms'] = get_r_tes_rms_new(iv_data['iBias'], iv_data['iBias_rms'], iv_data['iTES'], iv_data['iTES_rms'], r_p_rms, r_sh, iv_data['rTES'])
        # iv_data['rTES_rms'] = get_r_tes_rms(iv_data['iTES'], iv_data['iTES_rms'], iv_data['vTES'], iv_data['vTES_rms'])
        iv_data['vTES_rms'] = get_v_tes_rms_new(iv_data['iTES'], iv_data['iTES_rms'], iv_data['rTES'], iv_data['rTES_rms'], iv_data['vTES'])
        iv_data['pTES'] = get_p_tes(iv_data['iTES'], iv_data['vTES'])
        iv_data['pTES_rms'] = get_p_tes_rms(iv_data['iTES'], iv_data['iTES_rms'], iv_data['vTES'], iv_data['vTES_rms'])
    return iv_curves


def compute_extra_quantities(iv_curves):
    '''Function to compute other helpful debug quantities'''
    for temperature in iv_curves['iv'].keys():
        # Extra stuff
        dvd_t = np.gradient(iv_curves['iv'][temperature]['vOut'], iv_curves['iv'][temperature]['timestamps'], edge_order=2)
        did_t = np.gradient(iv_curves['iv'][temperature]['iBias'], iv_curves['iv'][temperature]['timestamps'], edge_order=2)
        dvdi = dvd_t/did_t
        ratio = iv_curves['iv'][temperature]['vOut']/iv_curves['iv'][temperature]['iBias']
        # How about a detrended iBias?
        detrend_i_bias = detrend(iv_curves['iv'][temperature]['iBias'], type='linear')
        # How about a "fake" RTES?
        # i_offset = -1.1275759312455495e-05
        # v_offset = 0.008104607442168656
        i_offset = 0
        v_offset = 0
        # ertes = 21e-3*((iv_curves['iv'][temperature]['iBias'] - i_offset)*(-1.28459*10000)/(iv_curves['iv'][temperature]['vOut'] - v_offset) - 1) - 12e-3
        # index_vector = np.array([i for i in range(dvdi.size)])
        iv_curves['iv'][temperature]['dvd_t'] = dvd_t
        iv_curves['iv'][temperature]['did_t'] = did_t
        iv_curves['iv'][temperature]['dvdi'] = dvdi
        iv_curves['iv'][temperature]['Ratio'] = ratio
        # iv_curves['iv'][temperature]['fakeRtes'] = ertes
        iv_curves['iv'][temperature]['iBiasDetrend'] = detrend_i_bias
    return iv_curves


def chop_data_by_temperature_steps(formatted_data, timelist, bias_channel, data_channel, squid):
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
    for values in timelist:
        start_time, stop_time, mean_temperature = values
        print('The value and type of mean_time_values is: {} and {}'.format(formatted_data['mean_time_values'], type(formatted_data['mean_time_values'])))
        print('The value and type of stop_time is: {} and {}'.format(stop_time, type(stop_time)))
        cut = np.logical_and(formatted_data['mean_time_values'] >= start_time + time_buffer, formatted_data['mean_time_values'] <= stop_time)
        cut = np.logical_and(cut, formatted_data['rms_waveforms'][data_channel] > 0)
        timestamps = formatted_data['mean_time_values'][cut]
        i_bias = formatted_data['mean_waveforms'][bias_channel][cut]/r_bias
        i_bias_rms = formatted_data['rms_waveforms'][bias_channel][cut]/r_bias
        v_out = formatted_data['mean_waveforms'][data_channel][cut]
        v_out_rms = formatted_data['rms_waveforms'][data_channel][cut]
        # Let us toss out T values wherein the digitizer rails
        if np.any(v_out_rms < 1e-15):
            print('Invalid digitizer response for T: {} mK'.format(np.round(mean_temperature*1e3, 3)))
            continue
        if stop_time - start_time > expected_duration:
            print('Temperature step is too long for T: {} mK. End: {}, Start: {}, Duration: {}'.format(np.round(mean_temperature*1e3, 3), stop_time, start_time, stop_time - start_time))
            continue
        else:
            temperature = str(np.round(mean_temperature*1e3, 3))
            # Let us also toss out temperatures if they contain bad data or jumps
            if cut_temperature_min < float(temperature) < cut_temperature_max:
                continue
            # Proceed to correct for SQUID Jumps
            # We should SORT everything by increasing time....
            sort_key = np.argsort(timestamps)
            timestamps = timestamps[sort_key]
            i_bias = i_bias[sort_key]
            i_bias_rms = i_bias_rms[sort_key]
            v_out = v_out[sort_key]
            v_out_rms = v_out_rms[sort_key]
            time_since_start = timestamps - timestamps[0]
            # We can technically get iTES at this point too since it is proportional to vOut but since it is let's not.
            print('Creating dictionary entry for T: {} mK'.format(temperature))
            # Make gradient to save as well
            # Try to do this: dV/d_t and di/d_t and then (dV/d_t)/(di/d_t) --> (dV/di)
            index_vector = np.array([i for i in range(timestamps.size)])
            iv_dictionary[temperature] = {'vBias': i_bias*r_bias, 'vBias_rms': i_bias_rms*r_bias, 'iBias': i_bias, 'iBias_rms': i_bias_rms, 'vOut': v_out, 'vOut_rms': v_out_rms, 'timestamps': timestamps, 'index': index_vector, 'TimeSinceStart': time_since_start}
    return iv_dictionary


def get_iv_data(argin):
    '''Function that returns a formatted iv dictionary from waveform root file'''
    formatted_data = get_pyiv_data(argin.inputPath, argin.outputPath, new_format=argin.newFormat, number_of_windows=argin.numberOfWindows, thermometer=argin.thermometer)
    timelist = get_temperature_steps(argin.outputPath, formatted_data['time_values'], formatted_data['temperatures'], pid_log=argin.pidLog, thermometer=argin.thermometer, tz_correction=argin.tzOffset)
    iv_dictionary = chop_data_by_temperature_steps(formatted_data, timelist, argin.biasChannel, argin.dataChannel, argin.squid)
    return iv_dictionary


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


def format_and_make_output_path(path, output_path):
    '''This function will format the output path into an absolute path and make
    the directory if needed.
    '''
    output_path = output_path if output_path else dirname(path) + '/' + basename(path).replace('.root', '')
    if not isabs(output_path):
        output_path = dirname(path) + '/' + output_path
    mkdpaths(output_path)
    return output_path


def iv_main(argin):
    '''The main IV processing function that will call other functions in order
    to process the entire IV dataset
    '''
    argin.outputPath = format_and_make_output_path(argin.inputPath, argin.outputPath)
    print('We will run with the following options:')
    print('The squid run is {}'.format(argin.run))
    print('The SQUID is: {}'.format(argin.squid))
    print('The output path is: {}'.format(argin.outputPath))
    iv_curves = {}
    # First step is to get basic IV data into a dictionary format. Either read raw files or load from a saved root file
    if argin.readROOT is False and argin.readTESROOT is False:
        iv_curves['iv'] = get_iv_data(argin)
        # Next try to correct squid jumps
        # iv_curves['iv'] = correct_squid_jumps(argin.outputPath, iv_curves['iv'])
        iv_curves = compute_extra_quantities(iv_curves)
        # Next save the iv_curves
        save_iv_to_root(argin.outputPath, iv_curves['iv'])
    if argin.readROOT is True and argin.readTESROOT is False:
        # If we saved the root file and want to load it do so here
        iv_curves['iv'] = read_from_ivroot(argin.outputPath + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 'timestamps'])
    # Next we can process the IV curves to get Rn and r_p values. Once we have r_p we can obtain vTES and go onward
    if argin.readTESROOT is False:
        iv_curves = process_iv_curves(argin.outputPath, argin.dataChannel, argin.squid, iv_curves)
        save_iv_to_root(argin.outputPath, iv_curves['iv'])
        iv_curves = get_tes_values(iv_curves, argin.squid)
        save_iv_to_root(argin.outputPath, iv_curves['iv'])
        print('Obtained TES values')
    if argin.readTESROOT is True:
        iv_curves['iv'] = read_from_ivroot(argin.outputPath + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 'timestamps', 'iTES', 'iTES_rms', 'vTES', 'vTES_rms', 'rTES', 'rTES_rms', 'pTES', 'pTES_rms'])
        # Note: We would need to also save or re-generate the fit_parameters dictionary?
    # This step onwards assumes iv_dictionary contains TES values
    iv_curves = process_tes_curves(iv_curves)
    # Make TES Plots
    if argin.plotTES is True:
        make_tes_plots(output_path=argin.outputPath,  data_channel=argin.dataChannel, squid=argin.squid, iv_dictionary=iv_curves['iv'], fit_dictionary=iv_curves['tes_fit_parameters'], individual=True)
    # Next let's do some special processing...R vs T, P vs T type of thing
    get_power_temperature_curves(argin.outputPath, argin.dataChannel, iv_curves['iv'])
    get_resistance_temperature_curves_new(argin.outputPath, argin.dataChannel, iv_curves['iv'])
    return True


if __name__ == '__main__':
    ARGS = input_parser()
    argin = InputArguments()
    argin.set_from_args(ARGS)
    iv_main(argin)
    print('done')
