import os
import argparse
from os.path import isabs
from os.path import dirname
from os.path import basename

import numpy as np
from numpy import log as ln
from numpy import square as pow2
from numpy import sqrt as npsqrt
from numpy import sum as nsum

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
from scipy.signal import detrend

import matplotlib as mp
from matplotlib import pyplot as plt

import pandas as pan
import iv_results
import iv_resistance
import squid_info

import IVPlots as ivp
import pyTESFitFunctions as fitfuncs
from ring_buffer import RingBuffer

import ROOT as rt

from readROOT import readROOT
from writeROOT import writeROOT

EPS = np.finfo(float).eps

#mp.use('agg')

class ArrayIsUnsortedException(Exception):
    '''Exception to raise if an array is unsorted'''
    pass

class InvalidChannelNumberException(Exception):
    '''Exception to raise if a channel number is invalid'''
    pass

class InvalidObjectTypeException(Exception):
    '''Exception to raise if the object passed into a function
    is not the correct type
    '''
    pass

class RequiredValueNotSetException(Exception):
    '''Exception to raise if a required value is missing'''
    pass


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
    keys = tfile.Getime_listOfKeys()
    key_list = [key.GetName() for key in keys]
    del tfile
    return key_list


def process_waveform(waveform, time_values, sample_length, number_of_windows=1, process_type='mean'):
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
    print('Processing waveform with {} winodws...'.format(number_of_windows))
    # Pre-allocation size
    number_of_entries = len(waveform) * number_of_windows
    # mean_waveform = np.empty(number_of_entries)
    # rms_waveform = np.empty(number_of_entries)
    # new_time_values = np.empty(number_of_entries)
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
    return processed_waveform


def correct_squid_jumps(output_path, iv_dictionary):
    '''Attempt to correct squid jumps for various IV data collections'''
    for temperature, iv_data in iv_dictionary.items():
        print('Attempting to correct for SQUID jumps for temperature {}'.format(temperature))
        iv_data = find_and_fix_squid_jumps(output_path, temperature, iv_data)
    return iv_dictionary


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
    print('The first y value is: {} and the location of the maxes y value is: {}'.format(ydata[0], np.argmax(ydata)))
    while difference_of_means < 3 and event < dyd_t.size - 1:
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
        step_away = 15
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
        result, pcov = curve_fit(fitfuncs.lin_sq, timestamps[event+1:event+distance_ahead], ydata[event+1:event+distance_ahead])
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


#@obsolete
#def process_waveform(dWaveform, procType='mean'):
#    '''Take an input waveform dictionary and process it by collapse to 1 point
#    An incoming waveform is a dictionary with keys = event number and values = waveform of duration 1s
#    This function will collapse the waveform to 1 value based on procType and return it as a numpy array
#    We will also return a bonus array of the rms for each point too
#    '''
#    # We can have events missing so better to make the vectors equal to the maxes key
#    npWaveform = np.empty(len(dWaveform.keys()))
#    npWaveformRMS = np.empty(len(dWaveform.keys()))
#    if procType == 'mean':
#        for ev, waveform in dWaveform.items():
#            npWaveform[ev] = np.mean(waveform)
#            npWaveformRMS[ev] = np.std(waveform)
#    elif procType == 'median':
#        for ev, waveform in dWaveform.items():
#            npWaveform[ev] = np.median(waveform)
#            q75, q25 = np.percentile(waveform, [75, 25])
#            npWaveformRMS[ev] = q75 - q25
#    else:
#        raise Exception('Please enter mean or median for process')
#    return npWaveform, npWaveformRMS


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
    for idx in range(buffer.size_maxes):
        buffer.append(data[idx + offset])
    return buffer


def find_end(past_buffer, current_buffer, future_buffer, temperature_step, temperatures, event):
    '''find an ending step boundary'''
    # Compute dMu_ij pairs
    step_event = None
    while event < temperatures.size - 1:
        dmu_pc = np.abs(past_buffer.get_mean() - current_buffer.get_mean())
        dmu_cf = np.abs(current_buffer.get_mean() - future_buffer.get_mean())
        # dMu_pf = np.abs(past_buffer.get_mean() - future_buffer.get_mean())
        # print(past_buffer.get_mean(), current_buffer.get_mean(), future_buffer.get_mean())
        # print(dmu_pc, dmu_cf, dMu_pf)
        # Test if past and future are similar...if these are then we assume current is similar and so advance
        if dmu_pc < 0.5*temperature_step and not dmu_cf < 0.5*temperature_step:
            # This is the return case...current similar to past but not future
            # Make the step_event be the first event of the future
            step_event = event - future_buffer.get_size()
            print('The step event is {}'.format(step_event))
            return_list = [event, step_event, past_buffer, current_buffer, future_buffer]
            break
        else:
            # past and future are within 1/2 temperature_step so increment all things by 1 event
            # other conditions too but for now increment by 1 event
            future_buffer_data = future_buffer.get_all()[-1]
            current_buffer_data = current_buffer.get_all()[-1]
            past_buffer.append(current_buffer_data)
            current_buffer.append(future_buffer_data)
            future_buffer.append(temperatures[event])
            event += 1
        return_list = [event - 1, event - 1, past_buffer, current_buffer, future_buffer]
    return return_list


def find_start(past_buffer, current_buffer, future_buffer, temperature_step, temperatures, event):
    '''find a starting step boundary'''
    # Compute dMu_ij pairs
    step_event = None
    while event < temperatures.size - 1:
        dmu_pc = np.abs(past_buffer.get_mean() - current_buffer.get_mean())
        dmu_cf = np.abs(current_buffer.get_mean() - future_buffer.get_mean())
        # dMu_pf = np.abs(past_buffer.get_mean() - future_buffer.get_mean())
        # print(past_buffer.get_mean(), current_buffer.get_mean(), future_buffer.get_mean())
        # print(dmu_pc, dmu_cf, dMu_pf)
        # Test if past and future are similar...if these are then we assume current is similar and so advance
        if dmu_cf < 0.5*temperature_step and not dmu_pc < 0.5*temperature_step:
            # This is the return case...current similar to future but not past
            # Make the step_event be the first event of the current
            step_event = event - future_buffer.get_size() - current_buffer.get_size()
            # print('The step event is {}'.format(step_event))
            return_list = [event, step_event, past_buffer, current_buffer, future_buffer]
            break
        else:
            # past and future are within 1/2 temperature_step so increment all things by 1 event
            # other conditions too but for now increment by 1 event
            # print('The 3 dMu values for start search are: {}'.format([dmu_pc, dmu_cf, dMu_pf]))
            future_buffer_data = future_buffer.get_all()[-1]
            current_buffer_data = current_buffer.get_all()[-1]
            past_buffer.append(current_buffer_data)
            current_buffer.append(future_buffer_data)
            future_buffer.append(temperatures[event])
            event += 1
        return_list = [event - 1, event - 1, past_buffer, current_buffer, future_buffer]
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
    past_buffer = RingBuffer(buffer_length, dtype=float)
    current_buffer = RingBuffer(buffer_length, dtype=float)
    future_buffer = RingBuffer(buffer_length, dtype=float)
    # Fill the buffers initially
    event = 0
    time_list = []
    start_time = timestamps[0]
    d_t = 0
    while event < temperatures.size - 1 and event + buffer_length < temperatures.size - 1:
        # We start with assuming the first window starts at t = timestamps[0]
        if time_list == []:
            start_time = timestamps[0]
            # Start past_buffer at ev = 0
            past_buffer = fill_ringbuffer(past_buffer, event, temperatures)
            event += buffer_length + 1
            current_buffer = fill_ringbuffer(current_buffer, event, temperatures)
            event += buffer_length + 1
            future_buffer = fill_ringbuffer(future_buffer, event, temperatures)
            event += buffer_length + 1
            # Now proceed to find an end
        else:
            # we have now found an end so now need to find a new start
            # We need new windows first
            # When we find an end point past ~ current !~ future
            # So adjust past <-- current, current <-- future, future <-- temperatures
            current_buffer_data = np.flip(current_buffer.get_all(), 0)
            future_buffer_data = np.flip(future_buffer.get_all(), 0)
            for idx in current_buffer_data:
                past_buffer.append(idx)
            for idx in future_buffer_data:
                current_buffer.append(idx)
            future_buffer = fill_ringbuffer(future_buffer, event, temperatures)
            event += buffer_length + 1
            event, step_event, past_buffer, current_buffer, future_buffer = find_start(past_buffer, current_buffer, future_buffer, temperature_step, temperatures, event)
            start_time = timestamps[step_event]
        # Now we have a start_time so we need to find a end_time.
        # When we find a start point, past !~ current ~ future
        # we can keep sliding forward until we reach past ~ current !~ future
        event, step_event, past_buffer, current_buffer, future_buffer = find_end(past_buffer, current_buffer, future_buffer, temperature_step, temperatures, event)
        end_time = timestamps[step_event]
        # Check validity of this temperature step: It must last longer than some amount of time
        # Also if the temperature is flagged as bad, ignore it.
        bad_temperatures = [(10e-3, 10.8e-3)]
        if end_time - start_time > d_t:
            cut = np.logical_and(timestamps >= start_time + d_t, timestamps <= end_time)
            mean_temperature = np.mean(temperatures[cut])
            ctemp = False
            for bad_temp_range in bad_temperatures:
                cbad = np.logical_and(mean_temperature >= bad_temp_range[0], mean_temperature <= bad_temp_range[1])
                if cbad is True:
                    print('Temperature {} is flagged as a bad temperature and will not be included onward'.format(mean_temperature))
                ctemp = np.logical_or(ctemp, cbad)
            if ctemp is False:
                tri = (start_time, end_time, mean_temperature)
                time_list.append(tri)
    return time_list


def walk_normal(xdata, ydata, side, buffer_size=200):
    '''Function to walk the normal branches and find the line fit
    To do this we will start at the min or maxes input current and compute a walking derivative
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
    while difference_of_means < 1e-2 and event < dydx.size - 1:
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


def walk_sc(xdata, ydata, buffer_size=5, plane='iv'):
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
    c_normal_to_sc_pos = np.logical_and(xdata > 0, di_bias < 0)
    c_normal_to_sc_neg = np.logical_and(xdata <= 0, di_bias > 0)
    c_normal_to_sc = np.logical_or(c_normal_to_sc_pos, c_normal_to_sc_neg)

    # Also select data that is some fraction of the normal resistance, say 20%
    # First let us compute the gradient (i.e. dy/dx)
    dydx = np.gradient(ydata, xdata, edge_order=2)

    # Set data that is in the SC to N transition to NaN in here
    xdata[~c_normal_to_sc] = np.nan
    ydata[~c_normal_to_sc] = np.nan
    dydx[~c_normal_to_sc] = np.nan

    # In the sc region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time.
    # If the new average differs from the previous by some amount mark that as the end.

    # First we should find whereabouts of (0,0)
    # This should roughly correspond to x = 0 since if we input nothing we should get out nothing. In reality there are parasitics of course
    if plane == 'tes':
        index_min_x = np.nanargmin(np.abs(xdata))
        # Occasionally we may have a shifted curve that is not near 0 for some reason (SQUID jump)
        # So find the min and maxes iTES and then find the central point
    elif plane == 'iv':
        ioffset = 0
        index_min_x = np.nanargmin(np.abs(xdata + ioffset))
        # NOTE: The above will fail for small SC regions where vOut normal > vOut sc!!!!
    # First go from index_min_x and increase
    # Create ring buffer of to store signal
    # TODO: FIX THIS TO HANDLE SQUID JUMPS
    # Start by walking buffer_size events to the right from the minimum abs. voltage
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


def make_tes_multiplot(output_path, data_channel, iv_dictionary, fit_parameters):
    '''Make a plot of all temperatures at once
    rTES vs iBias

    '''
    # Convert fit parameters to R values
    #R = convert_fit_to_resistance(fit_parameters, fit_type='tes')
    # Current vs Voltage
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    for temperature, data in iv_dictionary.items():
        params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
        axes_options = {'xlabel': 'Bias Current [uA]',
                        'ylabel': r'TES Resistance [m \Omega]',
                        'title': 'Channel {} TES Resistance vs Bias Current'.format(data_channel)
                        }
        axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['iBias'], y=data['rTES'], params=params, axes_options=axes_options, xScale=xScale, yScale=yScale)
        #axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, xScale=xScale, yScale=yScale)
        #axes = ivp.iv_fit_textbox(axes=axes, R=R, model=fit_parameters)
    axes.set_ylim((0*yScale, 1*yScale))
    axes.set_xlim((-20, 20))
    fName = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)

    # Overlay multiple IV plots
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e6
    tNames = []
    for temperature, data in iv_dictionary.items():
        if temperature not in ['9.908']:
            tNames.append(temperature)
            params = {'marker': 'o', 'markersize': 2, 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
            axes_options = {'xlabel': r'TES Voltage [$\mu$V]',
                            'ylabel': r'TES Current [$\mu$A]',
                            'title': 'Channel {} TES Current vs Voltage'.format(data_channel)
                            }
            axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['iTES'], params=params, axes_options=axes_options, xScale=xScale, yScale=yScale)
    # Add legend?
    axes.legend(['T = {} mK'.format(temperature) for temperature in tNames], markerscale=3)
    axes.set_ylim((0, 3))
    axes.set_xlim((-0.5, 2))
    fName = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName, dpi=150)

    # Overlay multiple IV plots
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    tNames = []
    for temperature, data in iv_dictionary.items():
        if temperature not in ['9.908']:
            tNames.append(temperature)
            params = {'marker': 'o', 'markersize': 2, 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
            axes_options = {'xlabel': r'TES Voltage [$\mu$V]',
                            'ylabel': r'TES Resistance [m$\Omega$]',
                            'title': 'Channel {} TES Resistance vs Voltage'.format(data_channel)
                            }
            axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['rTES'], params=params, axes_options=axes_options, xScale=xScale, yScale=yScale)
    # Add legend?
    axes.legend(['T = {} mK'.format(temperature) for temperature in tNames], markerscale=3)
    axes.set_ylim((0, 600))
    axes.set_xlim((0, 2))
    fName = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName, dpi=200)
    return True


def tes_plots(output_path, data_channel, temperature, data, fit_parameters):
    '''Helper function to generate standard TES plots
    iTES vs vTES
    rTES vs iTES
    rTES vs vTES
    rTES vs iBias
    pTES vs rTES
    pTES vs vTES
    '''

    # Convert fit parameters to R values
    R = convert_fit_to_resistance(fit_parameters, fit_type='tes')
    # Current vs Voltage
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e6
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': data['iTES_rms']*yScale}
    axes_options = {'xlabel': 'TES Voltage [uV]', 'ylabel': 'TES Current [uA]',
                    'title': 'Channel {} TES Current vs TES Voltage for T = {} mK'.format(data_channel, temperature)}

    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['iTES'], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, xScale=xScale, yScale=yScale)
    axes = ivp.iv_fit_textbox(axes=axes, R=R, model=fit_parameters)

    fName = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)

    # Resistance vs Current
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    ylim = (0, 1*yScale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': data['iTES_rms']*xScale, 'yerr': data['rTES_rms']*yScale}
    axes_options = {'xlabel': 'TES Current [uA]',
                    'ylabel': 'TES Resistance [mOhm]',
                    'title': 'Channel {} TES Resistance vs TES Current for T = {} mK'.format(data_channel, temperature),
                    'ylim': ylim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['iTES'], y=data['rTES'], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    fName = output_path + '/' + 'rTES_vs_iTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, fName)

    # Resistance vs Voltage
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    ylim = (0, 1*yScale)
    params = {
        'marker': 'o',
        'markersize': 2,
        'markeredgecolor': 'black',
        'markerfacecolor': 'black',
        'markeredgewid_th': 0,
        'linestyle': 'None',
        'xerr': data['vTES_rms']*xScale,
        'yerr': data['rTES_rms']*yScale
    }
    axes_options = {'xlabel': 'TES Voltage [uV]',
                    'ylabel': 'TES Resistance [mOhm]',
                    'title': 'Channel {} TES Resistance vs TES Voltage for T = {} mK'.format(data_channel, temperature),
                    'ylim': ylim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['rTES'], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    fName = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, fName)

    # Resistance vs Bias Current
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e3
    ylim = (0, 1*yScale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': data['iBias_rms']*xScale, 'yerr': data['rTES_rms']*yScale}
    axes_options = {'xlabel': 'Bias Current [uA]',
                    'ylabel': 'TES Resistance [mOhm]',
                    'title': 'Channel {} TES Resistance vs Bias Current for T = {} mK'.format(data_channel, temperature),
                    'ylim': ylim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['iBias'], y=data['rTES'], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    fName = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, fName)

    # Power vs rTES
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e12
    xlim = (0, 1*xScale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': data['rTES_rms']*xScale, 'yerr': data['pTES_rms']*yScale}
    axes_options = {'xlabel': 'TES Resistance [mOhm]',
                    'ylabel': 'TES Power [pW]',
                    'title': 'Channel {} TES Power vs TES Resistance for T = {} mK'.format(data_channel, temperature),
                    'xlim': xlim
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['rTES'], y=data['pTES'], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    fName = output_path + '/' + 'pTES_vs_rTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, fName)

    # Power vs vTES
    # Note this ideally is a parabola
    cut = np.logical_and(data['rTES'] > 500e-3, data['rTES'] < 2*500e-3)
    if nsum(cut) < 3:
        cut = np.ones(data['pTES'].size, dtype=bool)
    v = data['vTES'][cut]
    p = data['pTES'][cut]
    prms = data['pTES_rms'][cut]
    result, pcov = curve_fit(fitfuncs.quad_sq, v, p, sigma=prms, absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))
    pFit = fitfuncs.quad_sq(data['vTES'], result[0], result[1], result[2])
    fitResult = iv_results.FitParameters()
    fitResult.left.set_values(result, perr)
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e6
    yScale = 1e12
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': data['vTES_rms']*xScale, 'yerr': data['pTES_rms']*yScale}
    axes_options = {'xlabel': 'TES Voltage [uV]',
                    'ylabel': 'TES Power [pW]',
                    'title': 'Channel {} TES Power vs TES Resistance for T = {} mK'.format(data_channel, temperature)
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['pTES'], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['pTES'], model=fitResult, model_function=fitfuncs.quad_sq, xScale=xScale, yScale=yScale)
    axes = ivp.pr_fit_textbox(axes=axes, model=fitResult)

    fName = output_path + '/' + 'pTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivp.save_plot(fig, axes, fName)
    return True


def make_tes_plots(output_path, data_channel, iv_dictionary, fit_dictionary, individual=False):
    '''Loop through data to generate TES specific plots'''

    if individual == True:
        for temperature, data in iv_dictionary.items():
            tes_plots(output_path, data_channel, temperature, data, fit_dictionary[temperature])
    # Make a for all temperatures here
    make_tes_multiplot(output_path=output_path, data_channel=data_channel, iv_dictionary=iv_dictionary, fit_parameters=fit_dictionary)
    return True


def get_iTES(vOut, Rfb, M):
    '''Computes the TES current and TES current RMS in Amps'''
    iTES = vOut/Rfb/M
    return iTES


def get_rTES(iTES, vTES):
    '''Computes the TES resistance in Ohms'''
    rTES = vTES/iTES
    return rTES


def get_rTES_rms(iTES, iTES_rms, vTES, vTES_rms):
    '''Comptues the RMS on the TES resistance in Ohms'''
    # Fundamentally this is R = a*iBias/vOut - b
    # a = Rsh*Rfb*M
    # b = Rsh
    # dR/di_bias = a/vOut
    # dR/dVout = -a*iBias/vOut^2
    # rTES_rms = npsqrt( (dR/di_bias * iBiasRms)^2 + (dR/dVout * vOutRms)^2 )
    dR = npsqrt(pow2(vTES_rms/iTES) + pow2(-1*vTES*iTES_rms/pow2(iTES)))
    return dR


def get_pTES(iTES, vTES):
    '''Compute the TES power dissipation (Joule)'''
    pTES = iTES*vTES
    return pTES


def get_pTES_rms(iTES, iTES_rms, vTES, vTES_rms):
    '''Computes the RMS on the TES (Joule) power dissipation'''
    dP = npsqrt(pow2(iTES*vTES_rms) + pow2(vTES*iTES_rms))
    return dP


def get_vTES(iBias, vOut, Rfb, M, Rsh, Rp):
    '''computes the TES voltage in Volts
    vTES = vSh - vPara
    vTES = Rsh*(iSh) - Rp*iTES
    vTES = Rsh*(iBias - iTES) - Rp*iTES = Rsh*iBias - iTES*(Rp+Rsh)
    '''
    vTES = iBias*Rsh - (vOut/(M*Rfb))*Rsh - (vOut/(M*Rfb))*Rp
    # Simple model
    #vTES = Rsh*(iBias - vOut/M/Rfb)
    return vTES


def get_vTES_rms(iBias_rms, vOut, vOut_rms, Rfb, M, Rsh, Rp, Rp_err):
    '''compute the RMS on the TES voltage in Volts'''
    # Fundamentally this is V = Rsh*iBias - Rp/(MRfb)*vOut - Rsh/(MRfb)*vOut
    # errV**2 = (dV/di_bias * erriBias)**2 + (dV/dvOut * errvOut)**2 + (dV/dRp * errRp)**2
    # a = Rsh
    # b = Rsh/Rf/M
    # dV/di_bias = a
    # dV/dVout = -b
    # So this does as dV = npsqrt((dV/dIbias * iBiasRMS)^2 + (dV/dVout * vOutRMS)^2)
    dV = npsqrt(pow2(Rsh*iBias_rms) + pow2(((Rp+Rsh)/(M*Rfb))*vOut_rms) + pow2((vOut/(M*Rfb))*Rp_err))
    return dV


def get_TES_fits(tes_values):
    '''Function to get the TES fit parameters for the normal branches and superconducting branch'''
    #TODO: determine if we want to get slopes of V = m*I+ b or I = m*V + b and if we want to plot I vs V or V vs I
    # First umpack. These are sorted by vTES value
    iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms = tes_values

    # Get the superconducting branch first. This will tell us what the parasitic series resistance is
    # And will let us knwo the correction shift needed so iTES = 0 when vTES = 0
    # Get SC branch
    (evLeft, evRight) = walk_sc(vTES, iTES)
    # From here let us make a fit of the form y = m*x+b where y = vTES and x = iTES
    # Here m = Resistance and b = voltage offset
    # So when iTES = 0, vTES = b
    # NOTE: This follows Ohm's law but our plotting plane is inverted
    result, pcov = curve_fit(fitfuncs.lin_sq, iTES[evLeft:evRight], vTES[evLeft:evRight], sigma=vTES_rms[evLeft:evRight], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = fitfuncs.lin_sq(iTES, result[0], result[1])
    vF_sc = {'result': result, 'perr': perr, 'model': vFit*1e6}
    # Get the left side normal branch first
    lev = walk_normal(vTES, iTES, 'left')
    # Model is vTES = m*iTES + b
    result, pcov = curve_fit(fitfuncs.lin_sq, iTES[0:lev], vTES[0:lev], sigma=vTES_rms[0:lev], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = fitfuncs.lin_sq(iTES, result[0], result[1])
    vF_left = {'result': result, 'perr': perr, 'model': vFit*1e6}

    rev = walk_normal(vTES, iTES, 'right')
    result, pcov = curve_fit(fitfuncs.lin_sq, iTES[rev:], vTES[rev:], sigma=vTES_rms[rev:], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = fitfuncs.lin_sq(iTES, result[0], result[1])
    vF_right = {'result': result, 'perr': perr, 'model': vFit*1e6}

    # Adjust data based on intersection of SC and Normal data
    # V = Rn*I + Bn
    # V = Rs*I + Bs
    # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
    current_intersection = (vF_sc['result'][1] - vF_left['result'][1])/(vF_left['result'][0] - vF_sc['result'][0])
    voltage_intersection = vF_sc['result'][0]*current_intersection + vF_sc['result'][1]
    print('The current and voltage intersections are {} uA and {} uV'.format(current_intersection*1e6, voltage_intersection*1e6))
    iTES = iTES - current_intersection
    vTES = vTES - voltage_intersection

    # Redo walks
    (evLeft, evRight) = walk_sc(vTES, iTES)
    # From here let us make a fit of the form y = m*x+b where y = vTES and x = iTES
    # Here m = Resistance and b = voltage offset
    # So when iTES = 0, vTES = b
    # NOTE: This follows Ohm's law but our plotting plane is inverted
    result, pcov = curve_fit(fitfuncs.lin_sq, iTES[evLeft:evRight], vTES[evLeft:evRight], sigma=vTES_rms[evLeft:evRight], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = fitfuncs.lin_sq(iTES, result[0], result[1])
    vF_sc = {'result': result, 'perr': perr, 'model': vFit*1e6}
    # Get the left side normal branch first
    lev = walk_normal(vTES, iTES, 'left')
    # Model is vTES = m*iTES + b
    result, pcov = curve_fit(fitfuncs.lin_sq, iTES[0:lev], vTES[0:lev], sigma=vTES_rms[0:lev], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = fitfuncs.lin_sq(iTES, result[0], result[1])
    vF_left = {'result': result, 'perr': perr, 'model': vFit*1e6}

    rev = walk_normal(vTES, iTES, 'right')
    result, pcov = curve_fit(fitfuncs.lin_sq, iTES[rev:], vTES[rev:], sigma=vTES_rms[rev:], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    vFit = fitfuncs.lin_sq(iTES, result[0], result[1])
    vF_right = {'result': result, 'perr': perr, 'model': vFit*1e6}
    # Finally also recompute rTES
    rTES = vTES/iTES
    return vF_left, vF_sc, vF_right, [iTES, iTES_rms, vTES, vTES_rms, rTES, rTES_rms]


def dump2text(R, T, fileName):
    '''Quick function to dump R and T values to a text file'''
    print('The shape of R and T are: {0} and {1}'.format(R.shape, T.shape))
    np.savetxt(fileName, np.stack((R, T), axis=1), fmt='%12.10f')
    return True


def get_iv_data_from_file(input_path, new_format=False, thermometer='EP'):
    '''Load IV data from specified directory'''
    if thermometer == 'EP':
        thermometer_name = 'EPCal_K'
    else:
        thermometer_name = 'NT'
    if new_format is False:
        tree = 'data_tree'
        branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWid_th_s', 'Waveform', thermometer_name]
        method = 'chain'
        rData = readROOT(input_path, tree, branches, method)
        # Make life easier:
        rData = rData['data']
    else:
        chlist = 'ChList'
        channels = readROOT(input_path, None, None, method='single', tobject=chlist)
        channels = channels['data'][chlist]
        branches = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWid_th_s', thermometer_name] + ['Waveform' + '{:03d}'.format(int(i)) for i in channels]
        print('Branches to be read are: {}'.format(branches))
        tree = 'data_tree'
        method = 'chain'
        rData = readROOT(input_path, tree, branches, method)
        rData = rData['data']
        rData['Channel'] = channels
    return rData


def format_iv_data(iv_data, output_path, new_format=False, number_of_windows=1, thermometer='EP'):
    '''Format the IV data into easy to use forms'''

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
    # t[N] = Timestamp_s + Timestamp_mus*1e-6 + N*SamplingWid_th_s
    # Note that there will be NEvents different values of Timestamp_s and Timestamp_mus, each of these containing 1/SamplingWid_th_s samples
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
    # How we process depends upon the format
    if new_format == False:
        waveForms = {ch: {} for ch in np.unique(iv_data['Channel'])}
        nChan = np.unique(iv_data['Channel']).size
        for ev, ch in enumerate(iv_data['Channel'][cut]):
            waveForms[ch][ev//nChan] = iv_data['Waveform'][ev]
        # We now have a nested dictionary of waveforms.
        # waveForms[ch] consists of a dictionary of Nevents keys (actual events)
        # So waveForms[ch][ev] will be a numpy array with size = NumberOfSamples.
        # The timestamp of waveForms[ch][ev][sample] is time_values[ev] + sample*SamplingWid_th_s
        for channel in waveForms.keys():
            mean_waveforms[channel], rms_waveforms[channel], mean_time_values = process_waveform(waveForms[channel], time_values, iv_data['SamplingWid_th_s'][0], number_of_windows=number_of_windows)
    else:
        for channel in iv_data['Channel']:
            mean_waveforms[channel], rms_waveforms[channel], mean_time_values = process_waveform(iv_data['Waveform' + '{:03d}'.format(int(channel))], time_values, iv_data['SamplingWid_th_s'][0], number_of_windows=number_of_windows)
    print('The number of things in the mean time values are: {} and the number of waveforms are: {}'.format(np.size(mean_time_values), len(mean_waveforms[5])))
    formatted_data = {'time_values': time_values,
                      'temperatures': temperatures,
                      'mean_waveforms': mean_waveforms,
                      'rms_waveforms': rms_waveforms,
                      'mean_time_values': mean_time_values
                     }
    return formatted_data


def get_temperature_steps(output_path, time_values, temperatures, pid_log, thermometer='EP'):
    '''Returns a list of tuples that corresponds to temperature steps
    Depending on the value of pid_log we will either parse an existing pid log file or if it is None
    attempt to find the temperature steps
    '''
    if pid_log is None:
        timelist = find_temperature_steps(output_path, time_values, temperatures, thermometer)
    else:
        timelist = parse_temperature_steps(output_path, time_values, temperatures, pid_log)
    return timelist


def parse_temperature_steps(output_path, time_values, temperatures, pid_log):
    '''Run through the PID log and parse temperature steps
    The PID log has as the first column the timestamp a PID setting STARTS
    The second column is the power or temperature setting point
    '''
    times = pan.read_csv(pid_log, delimiter='\t', header=None)
    times = times.values[:, 0]
    # Each index of times is now the starting time of a temperature step. Include an appropriate offset for mean computation BUT only a softer one for time boundaries
    # timeList is a list of tuples.
    timeList = []
    start_offset = 40*60
    end_offset = 60
    if times.size > 1:
        for index in range(times.size - 1):
            cut = np.logical_and(time_values > times[index]+start_offset, time_values < times[index+1]-end_offset)
            mT = np.mean(temperatures[cut])
            start_time = times[index]+start_offset
            stop_time = times[index+1]-end_offset
            timeList.append((start_time, stop_time, mT))
        # Handle the last step
        # How long was the previous step?
        d_t = timeList[-1][1] - timeList[-1][0]
        start_time = timeList[-1][1] + start_offset
        end_time = start_time + d_t - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mT = np.mean(temperatures[cut])
        timeList.append((start_time, end_time, mT))
        d_t = time_values-time_values[0]
    else:
        # Only 1 temperature step defined
        start_time = times[0] + start_offset
        end_time = time_values[-1] - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mT = np.mean(temperatures[cut])
        timeList.append((start_time, end_time, mT))
        d_t = time_values-time_values[0]
    ivp.test_steps(d_t, temperatures, timeList, time_values[0], 'Time', 'T', output_path + '/' + 'test_temperature_steps.png')
    return timeList


def find_temperature_steps(output_path, time_values, temperatures, thermometer='EP'):
    ''' Given an array of time and temperatures identify temperature steps and return the start, stop and mean Temp
    Will return these as a list of tuples
    '''
    # At this point we should only process data further based on unique IV time blocks so here we must find
    # the list of time tuples (start_time, end_time, tTemp) that define a particular IV-curve temperature set

    # First construct dT/d_t and set time values to start at 0
    dT = np.gradient(temperatures, time_values)
    d_t = time_values-time_values[0]
    # Construct a diagnostic plot
    cut = np.logical_and(d_t > 2000, d_t < 4000)
    ivp.test_plot(d_t[cut], temperatures[cut], 'Time', 'T', output_path + '/' + 'test_Tbyt.png')
    ivp.test_plot(d_t[cut], dT[cut], 'Time', 'dT/d_t', output_path + '/' + 'test_dTbyd_t.png')
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
    timeList = get_stable_temperature_steps(time_values, temperatures, buffer_length, temperature_step)
    ivp.test_steps(d_t, temperatures, timeList, time_values[0], 'Time', 'T', output_path + '/' + 'test_temperature_steps.png')
    return timeList


def get_pyIV_data(input_path, output_path, new_format=True, number_of_windows=1, thermometer='EP'):
    '''Function to gather iv data in correct format
    Returns time values, temperatures, mean waveforms, rms waveforms and the list of times for temperature jumps
    '''
    iv_data = get_iv_data_from_file(input_path, new_format=new_format, thermometer=thermometer)
    formatted_data = format_iv_data(iv_data, output_path, new_format=new_format, number_of_windows=number_of_windows, thermometer=thermometer)
    return formatted_data


def fit_sc_branch(xdata, ydata, sigmaY, plane):
    '''Walk and fit the superconducting branch
    In the vOut vs iBias plane x = iBias, y = vOut --> dy/dx ~ resistance
    In the iTES vs vTES plane x = vTES, y = iTES --> dy/dx ~ 1/resistance
    '''
    # First generate a sortKey since dy/dx will require us to be sorted
    sortKey = np.argsort(xdata)
    (evLeft, evRight) = walk_sc(xdata[sortKey], ydata[sortKey], plane=plane)
    print('SC fit gives evLeft={} and evRight={}'.format(evLeft, evRight))
    print('Diagnostics: The input into curve_fit is as follows:')
    print('xdata size: {}, ydata size: {}, xdata NaN: {}, ydata NaN: {}'.format(xdata[sortKey][evLeft:evRight].size, ydata[sortKey][evLeft:evRight].size, nsum(np.isnan(xdata[sortKey][evLeft:evRight])), nsum(np.isnan(ydata[sortKey][evLeft:evRight]))))
    result, pcov = curve_fit(fitfuncs.lin_sq, xdata[sortKey][evLeft:evRight], ydata[sortKey][evLeft:evRight], sigma=sigmaY[sortKey][evLeft:evRight], absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))
    # In order to properly plot the superconducting branch fit try to find the boundaries of the SC region
    # One possibility is that the region has the smallest and largest y-value excursions. However this may not be the case
    # and certainly unless the data is sorted these indices are meaningless to use in a slice
    #index_y_min = np.argmin(y)
    #index_y_maxes = np.argmax(y)
    return result, perr #, (index_y_maxes, index_y_min)


def fit_normal_branches(xdata, ydata, sigmaY):
    '''Walk and fit the normal branches in the vOut vs iBias plane.'''
    # Generate a sortKey since dy/dx must be sorted
    sortKey = np.argsort(xdata)
    # Get the left side normal branch first
    left_ev = walk_normal(xdata[sortKey], ydata[sortKey], 'left')
    left_result, pcov = curve_fit(fitfuncs.lin_sq, xdata[sortKey][0:left_ev], ydata[sortKey][0:left_ev], sigma=sigmaY[sortKey][0:left_ev], absolute_sigma=True, method='trf')
    left_perr = npsqrt(np.diag(pcov))
    # Now get the other branch
    right_ev = walk_normal(xdata[sortKey], ydata[sortKey], 'right')
    right_result, pcov = curve_fit(fitfuncs.lin_sq, xdata[sortKey][right_ev:], ydata[sortKey][right_ev:], sigma=sigmaY[sortKey][right_ev:], absolute_sigma=True, method='trf')
    right_perr = np.sqrt(np.diag(pcov))
    return left_result, left_perr, right_result, right_perr


def correct_offsets(fitParams, iv_data, branch='normal'):
    ''' Based on the fit parameters for the normal and superconduting branch correct the offset'''
    # Adjust data based on intersection of SC and Normal data
    # V = Rn*I + Bn
    # V = Rs*I + Bs
    # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
    # This won't work if the lines are basically the same so let's detect if the sc and normal branch results roughly the same slope.
    # Recall that the slope of the fit is very big for a superconducting region.
    m_sc = fitParams.sc.result[0]
    m_right = fitParams.right.result[0]
    if np.abs((m_sc - m_right)/(m_sc)) < 0.5:
        print("Slopes are similar enough so try to impose a symmetrical shift")
        vmaxes = iv_data['vOut'].maxes()
        vmin = iv_data['vOut'].min()
        imaxes = iv_data['iBias'].maxes()
        imin = iv_data['iBias'].min()
        # TODO: FIX THIS
        current_intersection = (imaxes + imin)/2
        voltage_intersection = (vmaxes + vmin)/2
    else:
        #current_intersection = (fitParams.sc.result[1] - fitParams.left.result[1])/(fitParams.left.result[0] - fitParams.sc.result[0])
        #voltage_intersection = fitParams.sc.result[0]*current_intersection + fitParams.sc.result[1]
        vmaxes = iv_data['vOut'].maxes()
        vmin = iv_data['vOut'].min()
        imaxes = iv_data['iBias'].maxes()
        imin = iv_data['iBias'].min()
        # The below only works IF the data really truly is symmetric
        #current_intersection = (imaxes + imin)/2
        #voltage_intersection = (vmaxes + vmin)/2
        # Try this by selecting a specific point instead
        if branch == 'normal':
            idx_min = np.argmin(iv_data['iBias'])
            idx_maxes = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            voltage_intersection = (iv_data['vOut'][idx_maxes] + iv_data['vOut'][idx_min])/2
        elif branch == 'sc':
            idx_min = np.argmin(iv_data['vOut'])
            idx_maxes = np.argmax(iv_data['vOut'])
            current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            voltage_intersection = (iv_data['vOut'][idx_maxes] + iv_data['vOut'][idx_min])/2
        elif branch == 'right':
            current_intersection = (fitParams.sc.result[1] - fitParams.right.result[1])/(fitParams.right.result[0] - fitParams.sc.result[0])
            voltage_intersection = fitParams.sc.result[0]*current_intersection + fitParams.sc.result[1]
        elif branch == 'left':
            current_intersection = (fitParams.sc.result[1] - fitParams.left.result[1])/(fitParams.left.result[0] - fitParams.sc.result[0])
            voltage_intersection = fitParams.sc.result[0]*current_intersection + fitParams.sc.result[1]
        elif branch == 'interceptbalance':
            idx_min = np.argmin(iv_data['iBias'])
            idx_maxes = np.argmax(iv_data['iBias'])
            # balance current
            #current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            current_intersection = 0
            # balance y-intercepts
            voltage_intersection = (fitParams.left.result[1] + fitParams.right.result[1])/2
        elif branch == 'normal_current_sc_offset':
            idx_min = np.argmin(iv_data['iBias'])
            idx_maxes = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            voltage_intersection = fitParams.sc.result[1]
        elif branch == 'sc_current_normal_voltage':
            # Correct offset in current based on symmetrizing the SC region and correct the voltage by symmetrizing the normals
            # NOTE: THIS IS BAD
            idx_min = np.argmin(iv_data['vOut'])
            idx_maxes = np.argmax(iv_data['vOut'])
            current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            idx_min = np.argmin(iv_data['iBias'])
            idx_maxes = np.argmax(iv_data['iBias'])
            voltage_intersection = (iv_data['vOut'][idx_maxes] + iv_data['vOut'][idx_min])/2
        elif branch == 'sc_voltage_normal_current':
            # Correct offset in current based on symmetrizing the normal region and correct the voltage by symmetrizing the SC
            # THIS IS BAD
            idx_min = np.argmin(iv_data['iBias'])
            idx_maxes = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            idx_min = np.argmin(iv_data['vOut'])
            idx_maxes = np.argmax(iv_data['vOut'])
            voltage_intersection = (iv_data['vOut'][idx_maxes] + iv_data['vOut'][idx_min])/2
        elif branch == 'None':
            current_intersection = 0
            voltage_intersection = 0
        elif branch == 'dual':
            # Do both left and right inercept matches but take mean of the offset pairs.
            # Get Right
            right_current_intersection = (fitParams.sc.result[1] - fitParams.right.result[1])/(fitParams.right.result[0] - fitParams.sc.result[0])
            right_voltage_intersection = fitParams.sc.result[0]*right_current_intersection + fitParams.sc.result[1]
            # Do left
            left_current_intersection = (fitParams.sc.result[1] - fitParams.left.result[1])/(fitParams.left.result[0] - fitParams.sc.result[0])
            left_voltage_intersection = fitParams.sc.result[0]*left_current_intersection + fitParams.sc.result[1]
            # Compute mean
            current_intersection = (right_current_intersection + left_current_intersection)/2
            voltage_intersection = (right_voltage_intersection + left_voltage_intersection)/2
        elif branch == 'normal_bias_symmetric_normal_offset_voltage':
            idx_min = np.argmin(iv_data['iBias'])
            idx_maxes = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            right_current_intersection = (fitParams.sc.result[1] - fitParams.right.result[1])/(fitParams.right.result[0] - fitParams.sc.result[0])
            right_voltage_intersection = fitParams.sc.result[0]*right_current_intersection + fitParams.sc.result[1]
            # Do left
            left_current_intersection = (fitParams.sc.result[1] - fitParams.left.result[1])/(fitParams.left.result[0] - fitParams.sc.result[0])
            left_voltage_intersection = fitParams.sc.result[0]*left_current_intersection + fitParams.sc.result[1]
            voltage_intersection = (right_voltage_intersection + left_voltage_intersection)/2
        elif branch == 'normal_bias_symmetric_only':
            idx_min = np.argmin(iv_data['iBias'])
            idx_maxes = np.argmax(iv_data['iBias'])
            current_intersection = (iv_data['iBias'][idx_maxes] + iv_data['iBias'][idx_min])/2
            voltage_intersection = 0
        elif branch == 'dual_intersect_voltage_only':
            right_current_intersection = (fitParams.sc.result[1] - fitParams.right.result[1])/(fitParams.right.result[0] - fitParams.sc.result[0])
            right_voltage_intersection = fitParams.sc.result[0]*right_current_intersection + fitParams.sc.result[1]
            # Do left
            left_current_intersection = (fitParams.sc.result[1] - fitParams.left.result[1])/(fitParams.left.result[0] - fitParams.sc.result[0])
            left_voltage_intersection = fitParams.sc.result[0]*left_current_intersection + fitParams.sc.result[1]
            voltage_intersection = (right_voltage_intersection + left_voltage_intersection)/2
            current_intersection = 0
    return current_intersection, voltage_intersection


def convert_fit_to_resistance(fit_parameters, fit_type='iv', Rp=None, Rp_rms=None):
    '''Given a iv_results.FitParameters object convert to Resistance and Resistance error iv_resistance.TESResistance objects

    If a parasitic resistance is provided subtract it from the normal and superconducting branches and assign it
    to the parasitic property.

    If no parasitic resistance is provided assume that the superconducting region values are purely parasitic
    and assign the resulting value to both properties.

    '''
    squid_parameters = squid_info.SQUIDParameters(2)
    Rsh = squid_parameters.Rsh
    M = squid_parameters.M
    Rfb = squid_parameters.Rfb

    R = iv_resistance.TESResistance()
    # The interpretation of the fit parameters depends on what plane we are in
    if fit_type == 'iv':
        # We fit something of the form vOut = a*iBias + b
        Rsc = Rsh * ((M*Rfb)/fit_parameters.sc.result[0] - 1)
        if Rp is None:
            Rp = Rsc
        else:
            Rsc = Rsc - Rp
        Rsc_rms = np.abs((-1*M*Rfb*Rsh)/pow2(fit_parameters.sc.result[0]) * fit_parameters.sc.error[0])
        if Rp_rms is None:
            Rp_rms = Rsc_rms
        else:
            Rsc_rms = npsqrt(pow2(Rsc_rms) + pow2(Rp_rms))
        if fit_parameters.left.result is None:
            Rl, Rl_rms = None, None
        else:
            Rl = (M*Rfb*Rsh)/fit_parameters.left.result[0] - Rsh - Rp
            Rl_rms = npsqrt(pow2(fit_parameters.left.error[0] * (-1*M*Rfb*Rsh)/pow2(fit_parameters.left.result[0])) + pow2(-1*Rp_rms))
        if fit_parameters.right.result is None:
            Rr, Rr_rms = None, None
        else:
            Rr = (M*Rfb*Rsh)/fit_parameters.right.result[0] - Rsh - Rp
            Rr_rms = npsqrt(pow2(fit_parameters.right.error[0] * (-1*M*Rfb*Rsh)/pow2(fit_parameters.right.result[0])) + pow2(-1*Rp_rms))
    elif fit_type == 'tes':
        # Here we fit something of the form iTES = a*vTES + b
        # Fundamentally iTES = vTES/rTES ...
        Rsc = 1/fit_parameters.sc.result[0]
        if Rp is None:
            Rp = Rsc
        else:
            Rsc = Rsc - Rp
        Rsc_rms = np.abs((-1*fit_parameters.sc.error[0])/pow2(fit_parameters.sc.result[0]))
        if Rp_rms is None:
            Rp_rms = Rsc_rms
        else:
            Rsc_rms = npsqrt(pow2(Rsc_rms) + pow2(Rp_rms))
        if fit_parameters.left.result is None:
            Rl, Rl_rms = None, None
        else:
            Rl = 1/fit_parameters.left.result[0]
            Rl_rms = np.abs((-1*fit_parameters.left.error[0])/pow2(fit_parameters.left.result[0]))
        if fit_parameters.right.result is None:
            Rr, Rr_rms = None, None
        else:
            Rr = 1/fit_parameters.right.result[0]
            Rr_rms = np.abs((-1*fit_parameters.right.error[0])/pow2(fit_parameters.right.result[0]))
    R.parasitic.set_values(Rp, Rp_rms)
    R.left.set_values(Rl, Rl_rms)
    R.right.set_values(Rr, Rr_rms)
    R.sc.set_values(Rsc, Rsc_rms)
    return R


def fit_iv_regions(xdata, ydata, sigmaY, fittype='iv', plane='iv'):
    '''Fit the iv data regions and extract fit parameters'''
    squid_parameters = squid_info.SQUIDParameters(2)
    Rsh = squid_parameters.Rsh
    M = squid_parameters.M
    Rfb = squid_parameters.Rfb

    fitParams = iv_results.FitParameters()
    # We need to walk and fit the superconducting region first since there RTES = 0
    result, perr = fit_sc_branch(xdata, ydata, sigmaY, plane)
    # Now we can fit the rest
    left_result, left_perr, right_result, right_perr = fit_normal_branches(xdata, ydata, sigmaY)
#    # The interpretation of the fit parameters depends on what plane we are in
#    if fittype == 'iv':
#        # We fit something of the form vOut = a*iBias + b
#        Rp = Rsh * ((M*Rfb/result[0]) - 1)
#        Rp_rms = np.abs( ((-1*M*Rfb*Rsh)/(pow2(result[0])))*perr[0] )
#        Rl = (M*Rfb*Rsh/left_result[0]) - Rsh - R.parasitic
#        Rl_rms = npsqrt(pow2(left_perr[0]*(-1*M*Rfb*Rsh)/pow2(left_result[0])) + pow2(-1*Rerr.parasitic))
#        Rr = (M*Rfb*Rsh/right_result[0]) - Rsh - R.parasitic
#        Rr_rms = npsqrt(pow2(right_perr[0]*(-1*M*Rfb*Rsh)/pow2(right_result[0])) + pow2(-1*Rerr.parasitic))
#    elif fittype == 'tes':
#        # Here we fit something of the form iTES = a*vTES + b
#        # Fundamentally iTES = vTES/rTES ...
#        Rp = 1/result[0]
#        Rp_rms = np.abs(-1*perr[0]/pow2(result[0]))
#        Rl = 1/left_result[0]
#        Rl_rms = np.abs(-1*left_perr[0]/pow2(left_result[0]))
#        Rr = 1/right_result[0]
#        Rr_rms = np.abs(-1*right_perr[0]/pow2(right_result[0]))
#    R.parasitic.set_values(Rp, Rp_rms)
#    R.left.set_values(Rl, Rl_rms)
#    R.right.set_values(Rr, Rr_rms)
    fitParams.sc.set_values(result, perr)
    fitParams.left.set_values(left_result, left_perr)
    fitParams.right.set_values(right_result, right_perr)
    #TODO: Make sure everything is right here with the equations and error prop.
    return fitParams


def get_parasitic_resistances(iv_dictionary):
    '''Loop through IV data to obtain parasitic series resistance'''
    parasitic_dictionary = {}
    fitParams = iv_results.FitParameters()
    minT = list(iv_dictionary.keys())[np.argmin([float(T) for T in iv_dictionary.keys()])]
    for temperature, iv_data in iv_dictionary.items():
        print('Attempting to fit superconducting branch for temperature: {} mK'.format(temperature))
        result, perr = fit_sc_branch(iv_data['iBias'], iv_data['vOut'], iv_data['vOut_rms'], plane='iv')
        fitParams.sc.set_values(result, perr)
        R = convert_fit_to_resistance(fitParams, fit_type='iv')
        parasitic_dictionary[temperature] = R.parasitic
    return parasitic_dictionary, minT


def process_iv_curves(output_path, data_channel, iv_dictionary):
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

    squid_parameters = squid_info.SQUIDParameters(2)
    Rsh = squid_parameters.Rsh
    M = squid_parameters.M
    Rfb = squid_parameters.Rfb
    fit_parameters_dictionary = {}

    # First we should try to obtain a measure of the parasitic series resistance. This value will be subtracted from
    # subsequent fitted values of the TES resistance
    parasitic_dictionary, minT = get_parasitic_resistances(iv_dictionary)
    Rp, Rp_rms = parasitic_dictionary[minT].value, parasitic_dictionary[minT].rms

    # Loop through the iv data now and obtain fit parameters and correct alignment
    for temperature, iv_data in iv_dictionary.items():
        fit_parameters_dictionary[temperature] = fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigmaY=iv_data['vOut_rms'], fittype='iv', plane='iv')
        # Make it pass through zero. Correct offset.
        #i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'interceptbalance')
        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'dual')
        # Manual offset adjustment
        #i_offset, v_offset = (-1.1104794020729887e-05, 0.010244294053372446)
        #v_offset = fit_parameters_dictionary[temperature].sc.result[1]
        print('The maxesimum iBias={} and the minimum iBias={} with a total size={}'.format(iv_data['iBias'].maxes(), iv_data['iBias'].min(), iv_data['iBias'].size))
        print('For temperature {} the normal offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
        iv_data['vOut'] -= v_offset
        iv_data['iBias'] -= i_offset
#        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'sc')
#        #v_offset = fit_parameters_dictionary[temperature].sc.result[1]
#        print('For temperature {} the sc offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
#        iv_data['vOut'] -= v_offset
#        iv_data['iBias'] -= i_offset
        # Re-walk on shifted data
        fit_parameters_dictionary[temperature] = fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigmaY=iv_data['vOut_rms'], fittype='iv', plane='iv')
#        # Next shift the voltages
#        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'dual_intersect_voltage_only')
#        #v_offset = fit_parameters_dictionary[temperature].sc.result[1]
#        print('The maxesimum iBias={} and the minimum iBias={} with a total size={}'.format(iv_data['iBias'].maxes(), iv_data['iBias'].min(), iv_data['iBias'].size))
#        print('For temperature {} the normal offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
#        iv_data['vOut'] -= v_offset
#        iv_data['iBias'] -= i_offset
##        i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'sc')
##        #v_offset = fit_parameters_dictionary[temperature].sc.result[1]
##        print('For temperature {} the sc offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
##        iv_data['vOut'] -= v_offset
##        iv_data['iBias'] -= i_offset
#        # Re-walk on shifted data
#        fit_parameters_dictionary[temperature] = fit_iv_regions(x=iv_data['iBias'], y=iv_data['vOut'], sigmaY=iv_data['vOut_rms'], fittype='iv', plane='iv')
    # Next loop through to generate plots
    for temperature, iv_data in iv_dictionary.items():
        # Make I-V plot
        fName = output_path + '/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
        plt_data = [iv_data['iBias'], iv_data['vOut'], iv_data['iBias_rms'], iv_data['vOut_rms']]
        axes_options = {'xlabel': 'Bias Current [uA]',
                        'ylabel': 'Output Voltage [mV]',
                        'title': 'Channel {} Output Voltage vs Bias Current for T = {} mK'.format(data_channel, temperature)
                        }
        modelR = convert_fit_to_resistance(fit_parameters_dictionary[temperature], fit_type='iv', Rp=parasitic_dictionary[minT].value, Rp_rms=parasitic_dictionary[minT].rms)
        ivp.iv_fitplot(plt_data, fit_parameters_dictionary[temperature], modelR, parasitic_dictionary[minT], fName, axes_options, xScale=1e6, yScale=1e3)
        # Let's make a ROOT style plot (yuck)
        ivp.make_root_plot(output_path, data_channel, temperature, iv_data, fit_parameters_dictionary[temperature], parasitic_dictionary[minT], xScale=1e6, yScale=1e3)
    return iv_dictionary, fit_parameters_dictionary, parasitic_dictionary


def get_PT_curves(output_path, data_channel, iv_dictionary):
    '''Generate a power vs temperature curve for a TES'''
    # Need to select power in the biased region, i.e. where P(R) ~ constant
    # Try something at 0.5*Rn
    T = np.empty(0)
    P = np.empty(0)
    P_rms = np.empty(0)
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
        # Also select data that is some fraction of the normal resistance, say 20%
        R0 = 200e-3
        dR = 50e-3
        cut = np.logical_and(iv_data['rTES'] > R0 - dR, iv_data['rTES'] < R0 + dR)
        cut = np.logical_and(cut, c_normal_to_sc)
        if nsum(cut) > 0:
            T = np.append(T, float(temperature)*1e-3)
            P = np.append(P, np.mean(iv_data['pTES'][cut]))
            P_rms = np.append(P_rms, np.std(iv_data['pTES'][cut]))
    print('The main T vector is: {}'.format(T))
    # Remove the first half?
#    T = T[T.size//2:-1]
#    P = P[P.size//2:-1]
#    P_rms = P_rms[P_rms.size//2:-1]
#    T = T[0:T.size//2]
#    P = P[0:P.size//2]
#    P_rms = P_rms[0:P_rms.size//2]
    print('The half T vector is: {}'.format(T))
    # Make a plot without any fits to see what we have to work with...
    cutT = T < 32e-3
    cutP = P < 1000e-15
    cutT = np.logical_and(cutT, cutP)
    # Next make a P-T plot
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e15
    params = {'marker': 'o', 'markersize': 6, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': P_rms[cutT]*yScale}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Power [fW]',
                    'title': 'Channel {} TES Power vs Temperature'.format(data_channel),
                    'ylim': (0, None)
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T[cutT], y=P[cutT], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    fName = output_path + '/' + 'pTES_vs_T_nfuture_buffer_datait_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)
    # Attempt to fit it to a power function
    lBounds = [1e-15, 0, 10e-3]
    uBounds = [1, 7, 100e-3]
    x0 = [5e-06, 5, 35e-3]
    print('The input value of T is {} and for P it is: {} and for Prms it is: {}'.format(T[cutT], P[cutT], P_rms[cutT]))
    results, pcov = curve_fit(fitfuncs.tes_power_polynomial, T[cutT], P[cutT], p0=x0, sigma=P_rms[cutT], bounds=(lBounds, uBounds), absolute_sigma=True, method='trf', maxes_nfev=1e4)
    #results, pcov = curve_fit(fitfuncs.tes_power_polynomial, T[cutT], P[cutT], sigma=P_rms[cutT], p0=x0, absolute_sigma=True, method='lm', maxesfev=int(2e4))
    print('The covariance matrix columns are: [k, n, T] and the matrix is: {}'.format(pcov))
    #results, pcov = curve_fit(fitfuncs.tes_power_polynomial, T[cutT], P[cutT], sigma=P_rms[cutT], absolute_sigma=True, method='trf')
    perr = np.sqrt(np.diag(pcov))
    #results = [results[0], 5, results[1]]
    #perr = [perr[0], 0, perr[1]]
    #x0 = [x0[0], 5, x0[1]]
    #results = minimize(nll_error, x0=np.array(x0), args=(P,P_rms,T,fitfuncs.tes_power_polynomial), method='Nelder-Mead', options={'fatol':1e-15})
    #perr = results.x/100
    #results = results.x
    #results, pcov = curve_fit(fitfuncs.tes_power_polynomial, T[cutT], P[cutT], method='dogbox')

    fitResult = iv_results.FitParameters()
    fitResult.left.set_values(results, perr)
    #fitResult.right.set_values([results[0], 5, results[2]], [perr[0], 0, perr[2]])
    #fitResult.right.set_values(x0, [0,0,0])
    # Next make a P-T plot
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e15
    params = {'marker': 'o', 'markersize': 6, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': P_rms*yScale}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Power [fW]',
                    'title': 'Channel {} TES Power vs Temperature'.format(data_channel),
                    'ylim': (0, 400)
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T, y=P, axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    axes = ivp.add_model_fits(axes=axes, x=T, y=P, model=fitResult, model_function=fitfuncs.tes_power_polynomial, xScale=xScale, yScale=yScale)
    axes = ivp.pt_fit_textbox(axes=axes, model=fitResult)

    fName = output_path + '/' + 'pTES_vs_T_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)
    print('Results: k = {}, n = {}, Tb = {}'.format(*results))
    # Compute G
    # P = k*(Ts^n - T^n)
    # G = n*k*T^(n-1)
    print('G(Ttes) = {} pW/K'.format(results[0]*results[1]*np.power(results[2], results[1]-1)*1e12))
    print('G(10 mK) = {} pW/K'.format(results[0]*results[1]*np.power(10e-3, results[1]-1)*1e12))
    return True


def get_RT_curves(output_path, data_channel, iv_dictionary):
    '''Generate a resistance vs temperature curve for a TES'''
    # Rtes = R(i,T) really so select a fixed i and across multiple temperatures obtain values for R and then plot
    T = np.empty(0)
    R = np.empty(0)
    fixed_value = 'iTES'
    for temperature, iv_data in iv_dictionary.items():
        cut = np.logical_and(iv_data[fixed_value] > 0.0e-6, iv_data[fixed_value] < 1e-6)
        if nsum(cut) > 0:
            T = np.append(T, float(temperature)*1e-3) # T in K
            R = np.append(R, np.mean(iv_data['rTES'][cut]))
    # Next make an R-T plot
    # R vs T
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' +']',
                    'title': 'Channel {} TES Resistance vs Temperature'.format(data_channel)
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T, y=R, axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    #axes.set_ylim((-1,1))
    #axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, sc_bounds=sc_bounds, xScale=xScale, yScale=yScale)
    #axes = ivp.iv_fit_textbox(axes=axes, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)

    fName = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)

    # Make R vs T only for times we are going from higher iBias to lower iBias values
    # One way, assuming noise does not cause overlaps, is to only select points where iBias[i] > iBias[i+1]
    # If I take the diff array I get the following: di_bias[i] = iBias[i] - iBias[i-1]. If di_bias[i] < 0 then iBias is descending
    # So let's use that then.
    ####################################################################################
    # Rtes = R(i,T) really so select a fixed i and across multiple temperatures obtain values for R and then plot
    T = np.empty(0)
    R = np.empty(0)
    Rrms = np.empty(0)
    T_desc = np.empty(0)
    R_desc = np.empty(0)
    Rrms_desc = np.empty(0)
    fitResult = iv_results.FitParameters()
    i_select = 0.7e-6
    selector = 'iTES'
    for temperature, iv_data in iv_dictionary.items():
        cut = np.logical_and(iv_data[selector] > i_select - 0.1e-6, iv_data[selector] < i_select + 0.0e-6)
        print('the sum of cut is: {}'.format(nsum(cut)))
        # Cuts to select physical case where we go from Normal --> SC modes
        di_bias = np.gradient(iv_data['iBias'], edge_order=2)
        cut1 = np.logical_and(iv_data['iBias'] > 0, di_bias < 0)
        cut2 = np.logical_and(iv_data['iBias'] <= 0, di_bias > 0)
        dcut = np.logical_or(cut1, cut2)
        cut_desc = np.logical_and(cut, dcut)
        cut_asc = np.logical_and(cut, ~dcut)
        if nsum(cut_asc) > 0:
            T = np.append(T, float(temperature)*1e-3) # T in K
            R = np.append(R, np.median(iv_data['rTES'][cut_asc]))
            Rrms = np.append(Rrms, np.mean(iv_data['rTES_rms'][cut_asc]))
            #Rrms = np.append(Rrms, np.std(iv_data['rTES'][cut_asc]))
        if nsum(cut_desc) > 0:
            T_desc = np.append(T_desc, float(temperature)*1e-3) # T in K
            R_desc = np.append(R_desc, np.median(iv_data['rTES'][cut_desc]))
            Rrms_desc = np.append(Rrms_desc, np.mean(iv_data['rTES_rms'][cut_desc]))
            #Rrms_desc = np.append(Rrms_desc, np.std(iv_data['rTES'][cut_desc]))
    # Next make an R-T plot
    # Add a T cut?
    # Remove half
#    T = T[T.size//2:-1]
#    R = R[R.size//2:-1]
#    Rrms = Rrms[Rrms.size//2:-1]
#    T_desc = T_desc[T_desc.size//2:-1]
#    R_desc = R_desc[R_desc.size//2:-1]
#    Rrms_desc = Rrms_desc[Rrms_desc.size//2:-1]
    Tcut = T > 8e-3
    Tcut_desc = T_desc > 8e-3
    T = T[Tcut]
    R = R[Tcut]
    Rrms = Rrms[Tcut]
    T_desc = T_desc[Tcut_desc]
    R_desc = R_desc[Tcut_desc]
    Rrms_desc = Rrms_desc[Tcut_desc]
    # Try a fit?
    # [Rn, Rp, Tc, Tw]
    # In new fit we have [C, D, B, A] --> A = 1/Tw, B = -Tc/Tw


    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    sortKey = np.argsort(T)
    nice_current = np.round(i_select*1e6, 3)
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' +']',
                    'title': 'Channel {}'.format(data_channel) +  ' TES Resistance vs Temperature for TES Current = {}'.format(nice_current)  + r'$\mu$' + 'A'}
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewid_th': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': Rrms[sortKey]*yScale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T[sortKey], y=R[sortKey], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    axes.legend(['SC to N', 'N to SC'])
    fName = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_descending_iBias_nofit_' + str(np.round(i_select*1e6, 3)) + 'uA'
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)
    sortKey = np.argsort(T)
    x0 = [1, 0, T[sortKey][np.gradient(R[sortKey], T[sortKey], edge_order=2).argmax()]*1.1, 1e-3]
    x0 = [1, 0, 20e-3, 1e-3]
    #x0 = [1, 0, -T[sortKey][np.gradient(R[sortKey], T[sortKey], edge_order=2).argmax()]/1e-3,  1/1e-3]
    print('For SN to N fit initial guess is {}'.format(x0))
    #result, pcov = curve_fit(fitfuncs.tanh_tc, T, R, sigma=Rrms, absolute_sigma=True, p0=x0, method='trf')

    result, pcov = curve_fit(fitfuncs.tanh_tc, T, R, sigma=Rrms, absolute_sigma=True, method='trf', maxes_nfev=5e4)
    perr = np.sqrt(np.diag(pcov))
    print('Ascending (SC -> N): Rn = {} mOhm, Rp = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fitResult.left.set_values(result, perr)
    # Try a fit?
    sortKey = np.argsort(T_desc)
    x0 = [1, 0, T_desc[sortKey][np.gradient(R_desc[sortKey], T_desc[sortKey], edge_order=2).argmax()]*1.1, 1e-3]
    x0 = [1, 0, 20e-3, 1e-3]
    #x0 = [1, 0, -T_desc[sortKey][np.gradient(R_desc[sortKey], T_desc[sortKey], edge_order=2).argmax()]/1e-3, 1/1e-3]
    print('For descending fit (N->S) initial guess is {}'.format(x0))

    result_desc, pcov_desc = curve_fit(fitfuncs.tanh_tc, T_desc, R_desc, sigma=Rrms_desc, p0=x0, absolute_sigma=True, method='lm', maxesfev=int(5e4))
    perr_desc = np.sqrt(np.diag(pcov_desc))
    print('Descending (N -> SC): Rn = {} mOhm, Rp = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result_desc]))
    print('Descending Errors (N -> SC): Rn = {} mOhm, Rp = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in perr_desc]))
    fitResult.right.set_values(result_desc, perr_desc)
    # R vs T
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    sortKey = np.argsort(T)
    nice_current = np.round(i_select*1e6, 3)
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'title': 'Channel {}'.format(data_channel) + ' TES Resistance vs Temperature for TES Current = {}'.format(nice_current) + r'$\mu$' + 'A'
                    }
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewid_th': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': Rrms[sortKey]*yScale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T[sortKey], y=R[sortKey], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    sortKey = np.argsort(T_desc)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewid_th': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': Rrms_desc[sortKey]*yScale}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T_desc[sortKey], y=R_desc[sortKey], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    # axes.set_ylim((-1,1))
    axes = ivp.add_model_fits(axes=axes, x=T, y=R, model=fitResult, model_function=fitfuncs.tanh_tc, xScale=xScale, yScale=yScale)
    axes = ivp.rt_fit_textbox(axes=axes, model=fitResult)
    axes.legend(['SC to N', 'N to SC'])
    fName = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + selector + '_' + str(np.round(i_select*1e6, 3)) + 'uA'
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)


    # Make a nicer plot
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    sortKey = np.argsort(T_desc)
    NtoS_fitResult = iv_results.FitParameters()
    NtoS_fitResult.right.set_values(result_desc, perr_desc)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewid_th': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': Rrms_desc[sortKey]*yScale}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' +']',
                    'title': 'Channel {}'.format(data_channel) + ' TES Resistance vs Temperature for TES Current = {}'.format(np.round(i_select*1e6, 3))  + r'$\mu$' + 'A'
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T_desc[sortKey], y=R_desc[sortKey], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    # Let us pad the T values so they are smoooooooth
    model_T = np.linspace(T_desc.min(), 36e-3, 100000)
    axes = ivp.add_model_fits(axes=axes, x=model_T, y=R, model=NtoS_fitResult, model_function=fitfuncs.tanh_tc, xScale=xScale, yScale=yScale)
    axes = ivp.rt_fit_textbox(axes=axes, model=NtoS_fitResult)
    #axes.legend(['SC to N', 'N to SC'])
    fName = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + selector + '_' + str(np.round(i_select*1e6, 3)) + 'uA_normal_to_sc_only'
    axes.set_xlim((10, 36))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)
    ###############################################################
    # We can try to plot alpha vs R as well why not
    # alpha = To/Ro * dR/dT --> dln(R)/dln(T)
    #alpha = np.gradient(np.log(R), np.log(T), edge_order=2)
    modelT = np.linspace(T.min(), T.maxes(), 100)
    modelR = fitfuncs.tanh_tc(modelT, *fitResult.right.result)
    model_sortKey = np.argsort(modelT)
    #model_alpha = (modelT[model_sortKey]/modelR[model_sortKey])*np.gradient(modelR[model_sortKey], modelT[model_sortKey], edge_order=1)
    #model_alpha = np.gradient(np.log(modelR[model_sortKey]), np.log(modelT[model_sortKey]), edge_order=1)
    model_alpha = (modelT[model_sortKey]/modelR[model_sortKey])*np.gradient(modelR[model_sortKey], modelT[model_sortKey], edge_order=2)
    print('The maxes alpha is: {}'.format(np.max(model_alpha)))
    sortKey = np.argsort(T)
    alpha = (T[sortKey]/R[sortKey])*np.gradient(R[sortKey], T[sortKey], edge_order=1)/1e3
    #alpha = np.gradient(np.log(R[sortKey]), np.log(T[sortKey]), edge_order=1)
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markeredgewid_th': 0, 'linestyle': '-', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'TES Resistance [m' + 'r$\Omega$' + ']',
                    'ylabel': r'$\alpha$',
                    'title': 'Channel {} TES '.format(data_channel) + r'$\alpha$' +' vs Resistance'
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=modelR[model_sortKey], y=model_alpha, axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=R[sortKey], y=alpha, axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    # axes.set_ylim((-1,1))
    # axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, sc_bounds=sc_bounds, xScale=xScale, yScale=yScale)
    # axes = ivp.iv_fit_textbox(axes=axes, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)
    fName = output_path + '/' + 'alpha_vs_rTES_ch_' + str(data_channel)
    # axes.set_ylim((0,150))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)
    # alpha vs T
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': r'$\alpha$',
                    'title': 'Channel {} TES '.format(data_channel) + r'$\alpha$' +' vs Temperature'
                    }
    axes = ivp.generic_fitplot_with_errors(axes=axes, x=T[sortKey], y=alpha, axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    # axes.set_ylim((-1,1))
    # axes = ivp.add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, sc_bounds=sc_bounds, xScale=xScale, yScale=yScale)
    # axes = ivp.iv_fit_textbox(axes=axes, R=data['R'], Rerr=data['Rerr'], model=fit_parameters)
    fName = output_path + '/' + 'alpha_vs_T_ch_' + str(data_channel)
    axes.set_ylim((0, 150))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)

    # We can get R-T curves for multiple current selections as well :)
    # Proceed to do 0-1, 1-2, 2-3, up to 9-10
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xScale = 1e3
    yScale = 1e3
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black', 'markeredgewid_th': 0, 'linestyle': '-', 'xerr': None, 'yerr': None}
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES Resistance [m' + r'$\Omega$' + ']',
                    'title': 'Channel {} TES Resistance vs Temperature'.format(data_channel)
                    }
    for i in range(10):
        T = np.empty(0)
        R = np.empty(0)
        for temperature, iv_data in iv_dictionary.items():
            cut = np.logical_and(iv_data['iTES'] > i*1e-6, iv_data['iTES'] < (i+1)*1e-6) # select 'constant' I0
            # Select normal --> sc transition directions
            di_bias = np.gradient(iv_data['iBias'], edge_order=2)
            cut1 = np.logical_and(iv_data['iBias'] > 0, di_bias < 0)
            cut2 = np.logical_and(iv_data['iBias'] <= 0, di_bias > 0)
            dcut = np.logical_or(cut1, cut2)
            cut = np.logical_and(cut, dcut)
            if nsum(cut) > 0:
                T = np.append(T, float(temperature)*1e-3)
                R = np.append(R, np.mean(iv_data['rTES'][cut]))
        sortKey = np.argsort(T)
        axes = ivp.generic_fitplot_with_errors(axes=axes, x=T[sortKey], y=R[sortKey], axes_options=axes_options, params=params, xScale=xScale, yScale=yScale)
    fName = output_path + '/' + 'rTES_vs_T_multi_ch_' + str(data_channel)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    ivp.save_plot(fig, axes, fName)

    return True


def process_tes_curves(output_path, data_channel, iv_dictionary):
    '''Take TES data and find Rp and Rn values.'''
    squid_parameters = squid_info.SQUIDParameters(2)
    Rsh = squid_parameters.Rsh
    M = squid_parameters.M
    Rfb = squid_parameters.Rfb
    fit_dictionary = {}
    for temperature, iv_data in iv_dictionary.items():
        print('Processing TES for T = {} mK'.format(temperature))
        fitParams = fit_iv_regions(xdata=iv_data['vTES'], ydata=iv_data['iTES'], sigmaY=iv_data['iTES_rms'], fittype='tes', plane='tes')
        # Next we should center the IV data...it should pass through (0,0)
        # Adjust data based on intersection of SC and Normal data
        # V = Rn*I + Bn
        # V = Rs*I + Bs
        # Rn*I + Bn = Rs*I + Bs --> I = (Bs - Bn)/(Rn - Rs)
        #current_intersection, voltage_intersection = correct_offsets(fitParams)
        #iv_data['iBias'] = iv_data['iBias'] - current_intersection
        #iv_data['vOut'] = iv_data['vOut'] - voltage_intersection
        # Re-walk on shifted data
        #R, Rerr, fitParams, sc_bounds = fit_iv_regions(iv_data, keys)
        #iv_data['R'] = R
        #iv_data['Rerr'] = Rerr
        # Obtain R and P values
        fit_dictionary[temperature] = fitParams
    return iv_dictionary, fit_dictionary


def get_TES_values(output_path, data_channel, iv_dictionary, parasitic_dictionary):
    '''From I-V data values compute the TES values for iTES and vTES, ultimately yielding rTES'''
    squid_parameters = squid_info.SQUIDParameters(2)
    Rsh = squid_parameters.Rsh
    M = squid_parameters.M
    Rfb = squid_parameters.Rfb
    # Test: Select the parasitic resistance from the lowest temperature fit to use for everything
    minT = list(parasitic_dictionary.keys())[np.argmin([float(T) for T in parasitic_dictionary.keys()])]
    Rp, Rp_rms = parasitic_dictionary[minT].value, parasitic_dictionary[minT].rms
    for temperature, iv_data in iv_dictionary.items():
        iv_data['iTES'] = get_iTES(iv_data['vOut'], Rfb, M)
        iv_data['iTES_rms'] = get_iTES(iv_data['vOut_rms'], Rfb, M)

        iv_data['vTES'] = get_vTES(iv_data['iBias'], iv_data['vOut'], Rfb, M, Rsh, Rp)
        iv_data['vTES_rms'] = get_vTES_rms(iv_data['iBias_rms'], iv_data['vOut'], iv_data['vOut_rms'], Rfb, M, Rsh, Rp, Rp_rms)

        iv_data['rTES'] = get_rTES(iv_data['iTES'], iv_data['vTES'])
        iv_data['rTES_rms'] = get_rTES_rms(iv_data['iTES'], iv_data['iTES_rms'], iv_data['vTES'], iv_data['vTES_rms'])

        iv_data['pTES'] = get_pTES(iv_data['iTES'], iv_data['vTES'])
        iv_data['pTES_rms'] = get_pTES_rms(iv_data['iTES'], iv_data['iTES_rms'], iv_data['vTES'], iv_data['vTES_rms'])
    return iv_dictionary




def compute_extra_quantities(iv_dictionary):
    '''Function to compute other helpful debug quantities'''
    for T in iv_dictionary.keys():
        # Extra stuff
        dvd_t = np.gradient(iv_dictionary[T]['vOut'], iv_dictionary[T]['timestamps'], edge_order=2)
        did_t = np.gradient(iv_dictionary[T]['iBias'], iv_dictionary[T]['timestamps'], edge_order=2)
        dvdi = dvd_t/did_t
        Ratio = iv_dictionary[T]['vOut']/iv_dictionary[T]['iBias']
        # How about a detrended iBias?
        detrend_iBias = detrend(iv_dictionary[T]['iBias'], type='linear')
        # How about a "fake" RTES?
        #i_offset = -1.1275759312455495e-05
        #v_offset = 0.008104607442168656
        i_offset = 0
        v_offset = 0
        ertes = 21e-3*((iv_dictionary[T]['iBias']-i_offset)*(-1.28459*10000)/(iv_dictionary[T]['vOut'] - v_offset) - 1) - 12e-3
        #index_vector = np.array([i for i in range(dvdi.size)])
        iv_dictionary[T]['dvd_t'] = dvd_t
        iv_dictionary[T]['did_t'] = did_t
        iv_dictionary[T]['dvdi'] = dvdi
        iv_dictionary[T]['Ratio'] = Ratio
        iv_dictionary[T]['fakeRtes'] = ertes

        iv_dictionary[T]['iBiasDetrend'] = detrend_iBias
    return iv_dictionary


def chop_data_by_temperature_steps(output_path, formatted_data, timelist, bias_channel, data_channel):
    '''Chop up waveform data based on temperature steps'''
    squid_parameters = squid_info.SQUIDParameters(2)
    Rbias = squid_parameters.Rbias
    time_buffer = 0
    iv_dictionary = {}
    expected_duration = 6000 #TODO: make this an input argument or auto-determined somehow
    for values in timelist:
        start_time, stop_time, mean_temperature = values
        cut = np.logical_and(formatted_data['mean_time_values'] >= start_time + time_buffer, formatted_data['mean_time_values'] <= stop_time)
        timestamps = formatted_data['mean_time_values'][cut]
        iBias = formatted_data['mean_waveforms'][bias_channel][cut]/Rbias
        iBias_rms = formatted_data['rms_waveforms'][bias_channel][cut]/Rbias
        vOut = formatted_data['mean_waveforms'][data_channel][cut]
        vOut_rms = formatted_data['rms_waveforms'][data_channel][cut]
        # Let us toss out T values wherein the digitizer rails
        if np.any(vOut_rms < 1e-9):
            print('Invalid digitizer response for T: {} mK'.format(np.round(mean_temperature*1e3, 3)))
            continue
        if stop_time - start_time > expected_duration:
            print('Temperature step is too long for T: {} mK'.format(np.round(mean_temperature*1e3, 3)))
            continue
        else:
            T = str(np.round(mean_temperature*1e3, 3))
            # Proceed to correct for SQUID Jumps
            # We should SORT everything by increasing time....
            sortKey = np.argsort(timestamps)
            timestamps = timestamps[sortKey]
            iBias = iBias[sortKey]
            iBias_rms = iBias_rms[sortKey]
            vOut = vOut[sortKey]
            vOut_rms = vOut_rms[sortKey]
            normTime = timestamps - timestamps[0]
            # We can technically get iTES at this point too since it is proportional to vOut but since it is let's not.
            print('Creating dictionary entry for T: {} mK'.format(T))
            # Make gradient to save as well
            # Try to do this: dV/d_t and di/d_t and then (dV/d_t)/(di/d_t) --> (dV/di)
            index_vector = np.array([i for i in range(timestamps.size)])
            iv_dictionary[T] = {'iBias': iBias, 'iBias_rms': iBias_rms, 'vOut': vOut, 'vOut_rms': vOut_rms, 'timestamps': timestamps, 'index': index_vector, 'TimeSinceStart': normTime}
    return iv_dictionary


def get_iv_data(input_path, output_path, squid_run, bias_channel, data_channel, pid_log=None, new_format=False, number_of_windows=1, thermometer='EP'):
    '''Function that returns a formatted iv dictionary from waveform root file'''
    formatted_data = get_pyIV_data(input_path, output_path, new_format=new_format, number_of_windows=number_of_windows, thermometer=thermometer)
    timelist = get_temperature_steps(output_path, formatted_data['time_values'], formatted_data['temperatures'], pid_log=pid_log, thermometer=thermometer)
    iv_dictionary = chop_data_by_temperature_steps(output_path, formatted_data, timelist, bias_channel, data_channel)
    return iv_dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', default='/Users/bwelliver/cuore/bolord/squid/', help='Specify full file path of input data')
    parser.add_argument('-o', '--outputPath', help='Path to put the plot directory. Defaults to input directory')
    parser.add_argument('-r', '--run', type=int, help='Specify the SQUID run number to use')
    parser.add_argument('-b', '--biasChannel', type=int, default=5, help='Specify the digitizer channel that corresponds to the bias channel. Defaults to 5')
    parser.add_argument('-d', '--dataChannel', type=int, default=7, help='Specify the digitizer channel that corresponds to the output channel. Defaults to 7')
    parser.add_argument('-s', '--makeRoot', action='store_true', help='Specify whether to write data to a root file')
    parser.add_argument('-L', '--readROOT', action='store_true', help='Read IV data from processed root file. Stored in outputPath /root/iv_data.root')
    parser.add_argument('-l', '--readTESROOT', action='store_true', help='Read IV and TES data from processed root file. Stored in outputPath /root/iv_data.root')
    parser.add_argument('-n', '--newFormat', action='store_true', help='Specify whether or not to use the new ROOT format for file reading')
    parser.add_argument('-p', '--pidLog', default=None, help='Specify an optional PID log file to denote the step timestamps. If not supplied program will try to find them from Temperature data')
    parser.add_argument('-w', '--numberOfWindows', default=1, type=int, help='Specify the number of windows to divide one waveform sample up into for averaging. Default is 1 window per waveform.')
    parser.add_argument('-T', '--thermometer', default='EP', help='Specify the name of the thermometer to use. Can be either EP for EPCal (default) or NT for the noise thermometer')
    args = parser.parse_args()

    path = args.inputFile
    run = args.run
    output_path = args.outputPath if args.outputPath else dirname(path) + '/' + basename(path).replace('.root', '')
    if not isabs(output_path):
        output_path = dirname(path) + '/' + output_path
    mkdpaths(output_path)
    print('We will run with the following options:')
    print('The squid run is {}'.format(run))
    print('The output path is: {}'.format(output_path))
    #sns.set()
    # First step is to get basic IV data into a dictionary format. Either read raw files or load from a saved root file
    if args.readROOT is False and args.readTESROOT is False:
        iv_dictionary = get_iv_data(input_path=args.inputFile, output_path=output_path, squid_run=args.run, bias_channel=args.biasChannel, data_channel=args.dataChannel, pid_log=args.pidLog, new_format=args.newFormat, number_of_windows=args.numberOfWindows, thermometer=args.thermometer)
        # Next try to correct squid jumps
        #iv_dictionary = correct_squid_jumps(output_path, iv_dictionary)
        iv_dictionary = compute_extra_quantities(iv_dictionary)
        # Next save the iv_curves
        save_iv_to_root(output_path, iv_dictionary)
    if args.readROOT is True and args.readTESROOT is False:
        # If we saved the root file and want to load it do so here
        iv_dictionary = read_from_ivroot(output_path + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 'timestamps'])

    # Next we can process the IV curves to get Rn and Rp values. Once we have Rp we can obtain vTES and go onward
    if args.readTESROOT is False:
        iv_dictionary, fit_parameters_dictionary, parasitic_dictionary = process_iv_curves(output_path, args.dataChannel, iv_dictionary)
        save_iv_to_root(output_path, iv_dictionary)
        iv_dictionary = get_TES_values(output_path, args.dataChannel, iv_dictionary, parasitic_dictionary)
        save_iv_to_root(output_path, iv_dictionary)
        print('Obtained TES values')
    if args.readTESROOT is True:
        iv_dictionary = read_from_ivroot(output_path + '/root/iv_data.root', branches=['iBias', 'iBias_rms', 'vOut', 'vOut_rms', 'timestamps', 'iTES', 'iTES_rms', 'vTES', 'vTES_rms', 'rTES', 'rTES_rms', 'pTES', 'pTES_rms'])
        # Note: We would need to also save or re-generate the fit_parameters dictionary?

    # This step onwards assumes iv_dictionary contains TES values
    iv_dictionary, fit_dictionary = process_tes_curves(output_path, args.dataChannel, iv_dictionary)
    # Make TES Plots
    make_tes_plots(output_path=output_path, data_channel=args.dataChannel, iv_dictionary=iv_dictionary, fit_dictionary=fit_dictionary, individual=False)

    # Next let's do some special processing...R vs T, P vs T type of thing
    get_PT_curves(output_path, args.dataChannel, iv_dictionary)
    get_RT_curves(output_path, args.dataChannel, iv_dictionary)

    print('done')
