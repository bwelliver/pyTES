'''IV Processing Module'''
import time

import numpy as np
from numpy import square as pow2
from numpy import sqrt as npsqrt
from numpy import sum as nsum
from numba import jit, prange

from scipy.optimize import curve_fit

import tes_fit_functions as fitfuncs

import iv_results
import iv_resistance
import squid_info
import pytes_errors as pyTESErrors
from ring_buffer import RingBuffer


@jit(nopython=True)
def is_sorted(arr):
    """Check if a flat array is sorted."""
    for idx in range(arr.size-1):
        if arr[idx+1] < arr[idx]:
            return False
    return True


def convert_dict_to_ndarray(data_dictionary):
    '''A simple function to convert an IV dictionary whose keys are event indices and values
    are sample arrays into a 2D array
    '''
    n_events = len(data_dictionary)
    sz_array = data_dictionary[0].size
    ndarray = np.empty((n_events, sz_array))
    for event, sample in data_dictionary.items():
        ndarray[event] = sample
    return ndarray, sz_array


def vout_ibias_plane_resistance(fit_parameters, squid, r_p, r_p_rms):
    '''Function to convert fit parameters to resistance values
    for data in the Vout - iBias plane
    '''
    squid_parameters = squid_info.SQUIDParameters(squid)
    r_sh = squid_parameters.Rsh
    m_ratio = squid_parameters.M
    r_fb = squid_parameters.Rfb
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
    resistance = iv_resistance.TESResistance()
    resistance.parasitic.set_values(r_p, r_p_rms)
    resistance.left.set_values(r_left, r_left_rms)
    resistance.right.set_values(r_right, r_right_rms)
    resistance.sc.set_values(r_sc, r_sc_rms)
    return resistance


def ites_vtes_plane_resistance(fit_parameters, r_p, r_p_rms):
    '''Function to convert fit parameters to resistance values
    for data in the iTES - vTES plane
    '''
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
    resistance = iv_resistance.TESResistance()
    resistance.parasitic.set_values(r_p, r_p_rms)
    resistance.left.set_values(r_left, r_left_rms)
    resistance.right.set_values(r_right, r_right_rms)
    resistance.sc.set_values(r_sc, r_sc_rms)
    return resistance


def convert_fit_to_resistance(fit_parameters, squid, fit_type='iv', r_p=None, r_p_rms=None):
    '''Given a iv_results.FitParameters object convert to Resistance and Resistance error iv_resistance.TESResistance objects

    If a parasitic resistance is provided subtract it from the normal and superconducting branches and assign it
    to the parasitic property.

    If no parasitic resistance is provided assume that the superconducting region values are purely parasitic
    and assign the resulting value to both properties.

    '''
    # The interpretation of the fit parameters depends on what plane we are in
    if fit_type == 'iv':
        # We fit something of the form vOut = a*iBias + b
        resistance = vout_ibias_plane_resistance(fit_parameters, squid, r_p, r_p_rms)
    elif fit_type == 'tes':
        # Here we fit something of the form iTES = a*vTES + b
        # Fundamentally iTES = vTES/rTES ...
        resistance = ites_vtes_plane_resistance(fit_parameters, r_p, r_p_rms)
    return resistance


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
            (Deprecated): A dictionary whose keys represent the event number. Values are numpy arrays with a length = NumberOfSamples
            The timestamp of waveform[event][sample] is time_values[event] + sample*sample_length
            A nEvent x nSample ndarray, where the first index represents the event number.
            The timestamp for waveform[event][sample] is time_values[event] + sample*sample_length as in (dict)waveform case
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
    print('The shape of the waveform is {} and the len of time is {}'.format(waveform.shape, time_values.size))
    # Step 1: How many actual entries will we wind up with?
    number_of_entries = len(waveform) * number_of_windows
    processed_waveform = {'mean_waveform': np.empty(number_of_entries), 'rms_waveform': np.empty(number_of_entries), 'new_time': np.empty(number_of_entries)}
    if process_type in ('mean', 'serr_mean'):
        if isinstance(waveform, dict):
            processed_waveform = {'mean_waveform': [], 'rms_waveform': [], 'new_time': []}
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
        else:
            # TODO: Is there a better numpy equivalent to enumerate?
            for event, samples in enumerate(waveform):
                event_metadata = {'base_index': samples.size//number_of_windows, 'event_time': time_values[event]}
                sub_times = [event_metadata['event_time'] + sample_length/(2*number_of_windows) + idx/number_of_windows for idx in range(number_of_windows)]
                mean_samples, std_samples = waveform_processor(samples, number_of_windows, process_type=process_type)
                #mean_samples, std_samples = average_groups(samples, number_of_windows)
                #processed_waveform['mean_waveform'].extend(mean_samples)
                #processed_waveform['rms_waveform'].extend(std_samples)
                #processed_waveform['new_time'].extend(sub_times)
                start_index = event*number_of_windows
                end_index = start_index + number_of_windows
                processed_waveform['new_time'][start_index:end_index] = sub_times
                processed_waveform['mean_waveform'][start_index:end_index] = mean_samples
                processed_waveform['rms_waveform'][start_index:end_index] = std_samples
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


def iv_windower(iv_dictionary, number_of_windows, mode='iv'):
    '''Returns a dictionary of windowed IV data'''
    iv_curves = {}
    if mode == 'iv':
        process_keys = ['iBias', 'vOut']
    if mode == 'tes':
        process_keys = ['iBias', 'vOut', 'iTES', 'vTES', 'rTES', 'pTES']
    if number_of_windows > 0:
        for temperature, iv_data in sorted(iv_dictionary.items()):
            iv_curves[temperature] = {}
            for key in process_keys:
                st = time.time()
                print('Size check: The size of sampling width is: {}'.format(iv_data['sampling_width'].size))
                processed_waveforms = process_waveform(iv_data[key], iv_data['timestamps'], iv_data['sampling_width'][0], number_of_windows=number_of_windows)
                print('Process function took {} s to run'.format(time.time() - st))
                print('The shape of the output processed waveform is: {}'.format(processed_waveforms['mean_waveform'].shape))
                value, value_err, time_values = processed_waveforms.values()
                iv_curves[temperature][key] = value
                iv_curves[temperature][key + '_rms'] = value_err
                iv_curves[temperature]['timestamps'] = time_values
#        for temperature, iv_data in sorted(iv_dictionary.items()):
#            st = time.time()
#            processed_waveforms = process_waveform(iv_data['iBias'], iv_data['timestamps'], iv_data['sampling_width'][0], number_of_windows=number_of_windows)
#            print('Process function took: {} s to run'.format(time.time() - st))
#            iBias, iBias_err, time_values = processed_waveforms.values()
#            st = time.time()
#            processed_waveforms = process_waveform(iv_data['vOut'], iv_data['timestamps'], iv_data['sampling_width'][0], number_of_windows=number_of_windows)
#            print('Process function took: {} s to run'.format(time.time() - st))
#            vOut, vOut_err, time_values = processed_waveforms.values()
#            # Here iBias is a simple 1D array...
#            iv_curves[temperature] = {'iBias': iBias, 'iBias_rms': iBias_err, 'vOut': vOut, 'vOut_rms': vOut_err, 'timestamps': time_values}
    else:
        for temperature, iv_data in sorted(iv_dictionary.items()):
            iv_curves[temperature] = {}
            for key in process_keys:
                end_idx = int(0.05*iv_data[key].size)
                iv_curves[temperature][key] = iv_data[key]
                iv_curves[temperature][key + '_rms'] = np.std(iv_data[key].flatten()[0:end_idx])/np.sqrt(end_idx)
            iv_curves[temperature]['timestamps'] = iv_data['timestamps']
#            # Take an estimate for the distribution width using 5% of the total data
#            end_idx = int(0.05*iv_data['iBias'].size)
#            # Here iv_data[iBias] is an nEvent x nSamples ndarray
#            iv_curves[temperature] = {
#                    'iBias': iv_data['iBias'], 'iBias_rms':  np.std(iv_data['iBias'].flatten()[0:end_idx])/np.sqrt(end_idx),
#                    'vOut': iv_data['vOut'], 'vOut_rms': np.std(iv_data['vOut'].flatten()[0:end_idx])/np.sqrt(end_idx), 'timestamps': iv_data['timestamps']
#                    }
    return iv_curves


def fit_iv_regions(xdata, ydata, sigma_y, number_samples, sampling_width, number_of_windows, slew_rate, plane='iv'):
    '''Fit the iv data regions and extract fit parameters'''

    fit_params = iv_results.FitParameters()
    # We need to walk and fit the superconducting region first since there RTES = 0
    result, perr = fit_sc_branch(xdata, ydata, sigma_y, number_samples, sampling_width, number_of_windows, slew_rate, plane)
    # Now we can fit the rest
    left_result, left_perr, right_result, right_perr = fit_normal_branches(xdata, ydata, sigma_y, number_samples, sampling_width, number_of_windows, slew_rate)
    fit_params.sc.set_values(result, perr)
    fit_params.left.set_values(left_result, left_perr)
    fit_params.right.set_values(right_result, right_perr)
    # TODO: Make sure everything is right here with the equations and error prop.
    return fit_params


def get_parasitic_resistance(iv_dictionary, squid, number_samples, sampling_width, number_of_windows, slew_rate):
    """Obtain estimate of parasitic series resistance from minimum temperature SC region.
    
    Here the data being examined is windowed.
    """
    min_temperature = list(iv_dictionary.keys())[np.argmin([float(temperature) for temperature in iv_dictionary.keys()])]
    fit_params = iv_results.FitParameters()
    print('Attempting to fit superconducting branch for temperature: {} mK'.format(min_temperature))
    result, perr = fit_sc_branch(iv_dictionary[min_temperature]['iBias'], iv_dictionary[min_temperature]['vOut'], iv_dictionary[min_temperature]['vOut_rms'], number_samples, sampling_width, number_of_windows, slew_rate, plane='iv')
    fit_params.sc.set_values(result, perr)
    resistance = convert_fit_to_resistance(fit_params, squid, fit_type='iv')
    return resistance.parasitic


def get_parasitic_resistances(iv_dictionary, squid, number_samples, sampling_width, number_of_windows, slew_rate):
    '''Loop through windowed IV data to obtain parasitic series resistance'''
    parasitic_dictionary = {}
    fit_params = iv_results.FitParameters()
    min_temperature = list(iv_dictionary.keys())[np.argmin([float(temperature) for temperature in iv_dictionary.keys()])]
    for temperature, iv_data in sorted(iv_dictionary.items()):
        print('Attempting to fit superconducting branch for temperature: {} mK'.format(temperature))
        result, perr = fit_sc_branch(iv_data['iBias'], iv_data['vOut'], iv_data['vOut_rms'], number_samples, sampling_width, number_of_windows, slew_rate, plane='iv')
        fit_params.sc.set_values(result, perr)
        resistance = convert_fit_to_resistance(fit_params, squid, fit_type='iv')
        parasitic_dictionary[temperature] = resistance.parasitic
    return parasitic_dictionary, min_temperature


def fit_sc_branch(xdata, ydata, sigma_y, number_samples, sampling_width, number_of_windows, slew_rate, plane):
    """Determine location of the SC branch and fit a line to it.
    
    In the 'iv' plane we have the vOut vs iBias and so dy/dx ~ resistance.
    In the 'tes' plane we have iTES vs vTES and so dy/dx ~ 1/resistance.
    """
    # The philosophy here will be to select 1 of the directionalities (SC->N or N->SC) and assume that where iBias ~ 0
    # we are around where we should be for the SC region
    # We can break this up into the following steps:
    # 1. Create directionality cut and select the relevant data
    # 2. Sort the data by x since we will be in the y vs x plane now and both y and x are time-ordered.
    # 3. Locate approximately (x=0, y(x=0))
    # 4. Determine appropriate cut to select a region for fitting
    # 5. Fit this region and extract relevant information.
    
    # If we did not need to sort by xdata we could just use flatiterators perhaps.
    if xdata.ndim == 2:
        number_samples = xdata.shape[1]
        xdata = xdata.flatten()
        ydata = ydata.flatten()
        sigma_y = sigma_y.flatten()
    # Get directionality cut
    di_bias = np.gradient(xdata, edge_order=2)
    c_normal_to_sc_pos = np.logical_and(xdata > 0, di_bias < 0)
    c_normal_to_sc_neg = np.logical_and(xdata <= 0, di_bias > 0)
    c_normal_to_sc = np.logical_or(c_normal_to_sc_pos, c_normal_to_sc_neg)
    # Sort by x
    sortkey = np.argsort(xdata)
    xdata = xdata[sortkey]
    ydata = ydata[sortkey]
    sigma_y = sigma_y[sortkey]
    c_normal_to_sc = c_normal_to_sc[sortkey]
    # Apply directionality cut
    xdata = xdata[~c_normal_to_sc]
    ydata = ydata[~c_normal_to_sc]
    sigma_y = sigma_y[~c_normal_to_sc]
    sc_cut = walk_sc(xdata, ydata, number_samples, sampling_width, number_of_windows, slew_rate, plane=plane)
    # Finally cut further to the sc region
    print('The size of sc_cut is {} and the amount that is true: {}'.format(sc_cut.size, sc_cut.sum()))
    xdata = xdata[sc_cut]
    ydata = ydata[sc_cut]
    sigma_y = sigma_y[sc_cut]
    # print('Diagnostics: The input into curve fit is as follows:')
    # print('xdata size: {}, ydata size: {}, xdata NaN: {}, ydata NaN: {}'.format(xdata.size, ydata.size, np.sum(np.isnan(xdata)), np.sum(np.isnan(ydata))))
    m0 = (ydata[-1] - ydata[0])/(xdata[-1] - xdata[0])
    p0 = (m0, 0)
    result, pcov = curve_fit(fitfuncs.lin_sq, xdata, ydata, sigma=sigma_y, absolute_sigma=True, p0=p0, method='trf')
    print('The sc fit result is: {}'.format(result))
    # result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, p0=(38, 0), method='trf')
    perr = np.sqrt(np.diag(pcov))
        
    return result, perr


def walk_sc(xdata, ydata, number_samples, sampling_width, number_of_windows, slew_rate, delta_current=None, plane='iv'):
    """Function to walk the superconducting region of the IV curve and get the left and right edges

    In order to be correct your x and y data values must be sorted by x
    """
    
    # Input is a directional (SC->N or N->SC) cut array. Our task is to find a cut on this that we will pass
    # off to a fitter.
    # We will walk along the curve from the 0 point in either direction and flip a bool array from False to True if the new point is acceptable.
    # To do this we can keep a RingBuffer that slides along the curve and checks the change in its average compared to some threshold. As long
    # as it does not exceed this threshold the new point is acceptible. Otherwise it is not.
    
    # Ensure we have the proper sorting of the data
    if not is_sorted(xdata):
        raise pyTESErrors.ArrayIsUnsortedException('Input argument x is unsorted')
    # Next we need to figure out how big the buffer should be. A larger buffer will be less suceptible to small scale fluctuations
    # but if it is too big it could select data outside the SC region. A few uA is probably ok.
    
    # deltaT == deltaI/slew_rate --> gives an idea of how long in time the requested current size would take for this data
    # tWindow = nSamples*dt/nWindows --> this is the total time per 'waveform' over the number of windows we chop it into --> lenght of time of each window
    # deltaT/tWindow = number of windows inside deltaT (aka number of points)
    # TODO: Adjust buffer size to reflect the amount of xData points (different based on the plane we live in)
    # Check buffer size
    if delta_current is None:
        if plane == 'iv':
            delta_current = 7
        elif plane == 'tes':
            delta_current = 7
        else:
            delta_current = 7
    buffer_size = int((delta_current / slew_rate) / ((number_samples * sampling_width) / number_of_windows))
    print('For a delta current of {} uA with a ramp slew rate of {} uA/s, the buffer requires {} windowed points'.format(delta_current, slew_rate, buffer_size))
    
    # First let us compute the gradient (i.e. dy/dx)
    dydx = np.gradient(ydata, xdata, edge_order=2)

    # Find whereabouts of (0,0)
    # This should roughly correspond to x = 0 since if we input nothing we should get out nothing. In reality there are parasitics of course
    if plane == 'tes':
        # Ideally we should look for the point that is closest to (0, 0)!
        distance = np.zeros(xdata.size)
        px, py = (0, 0)
        for idx in range(xdata.size):
            dx = xdata[idx] - px
            dy = ydata[idx] - py
            distance[idx] = np.sqrt(dx**2 + dy**2)
        index_min_x = np.nanargmin(distance)
        print('The point closest to ({}, {}) is at index {} with distance {} and is ({}, {})'.format(
            px,
            py,
            index_min_x,
            distance[index_min_x],
            xdata[index_min_x],
            ydata[index_min_x]))
        # Occasionally we may have a shifted curve that is not near 0 for some reason (SQUID jump)
        # So find the min and max iTES and then find the central point
    elif plane == 'iv':
        # Find the point closest to 0 iBias.
        ioffset = 0
        index_min_x = np.nanargmin(np.abs(xdata + ioffset))
        # NOTE: The above will fail for small SC regions where vOut normal > vOut sc!!!!
    # Start by walking buffer_size events to the right from the minimum abs. voltage
    cut = get_sc_endpoints(buffer_size, index_min_x, dydx)
    return cut


@jit(nopython=True)
def get_sc_endpoints(buffer_size, index_min_x, dydx):
    """Select the region of the SC branch to pass to the fitter."""
    # Look for rightmost endpoint, keeping in mind it could be our initial point
    delta_mean_threshold = 1e-2
    cut = np.zeros(dydx.shape, dtype=np.bool_)
    sz_check = np.zeros(2)
    sz_check[0] = dydx.size - index_min_x - 1
    sz_check[1] = 0
    if buffer_size + index_min_x >= dydx.size:
        # Buffer size and offset would go past end of data
        right_buffer_size = np.nanmax(sz_check)
    else:
        right_buffer_size = buffer_size
    right_buffer_size = np.int32(right_buffer_size)
    slope_buffer = RingBuffer(right_buffer_size, np.int32(0), np.float32)
    # Now fill the buffer and cut
    for event in range(right_buffer_size):
        slope_buffer.append(dydx[index_min_x + event])
        cut[index_min_x + event] = True
    # The buffer is full with initial values. Now walk along it
    ev_right = index_min_x + right_buffer_size
    difference_of_means = 0
    while difference_of_means < delta_mean_threshold and ev_right < dydx.size - 1:
        current_mean = slope_buffer.get_nanmean()
        slope_buffer.append(dydx[ev_right])
        new_mean = slope_buffer.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        cut[ev_right] = True
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
    left_buffer_size = np.int32(left_buffer_size)
    slope_buffer = RingBuffer(left_buffer_size, 0, np.float32)
    # Do initial appending
    for event in range(left_buffer_size):
        slope_buffer.append(dydx[index_min_x - event])
        cut[index_min_x - event] = True
    # Walk to the left
    ev_left = index_min_x - left_buffer_size
    difference_of_means = 0
    # print('The value of ev_left to start is: {}'.format(ev_left))
    while difference_of_means < delta_mean_threshold and ev_left >= 0:
        current_mean = slope_buffer.get_nanmean()
        slope_buffer.append(dydx[ev_left])
        new_mean = slope_buffer.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        cut[ev_left] = True
        ev_left -= 1
    return cut

def fit_sc_branch_old(xdata, ydata, sigma_y, number_samples, sampling_width, number_of_windows, slew_rate, plane):
    '''Walk and fit the superconducting branch
    In the vOut vs iBias plane x = iBias, y = vOut --> dy/dx ~ resistance
    In the iTES vs vTES plane x = vTES, y = iTES --> dy/dx ~ 1/resistance
    '''
    # First generate a sort_key since dy/dx will require us to be sorted
    # Flatten if necessary:
    print('The shape of xdata is {}'.format(xdata.shape))
    if len(xdata.shape) == 2:
        print('Flattening is necessary')
        number_samples = xdata.shape[1]
        xdata = xdata.flatten()
        ydata = ydata.flatten()
        sigma_y = sigma_y.flatten()
    if len(sigma_y.shape) == 1:
        sigma_y = np.zeros(xdata.size) + sigma_y
    sort_key = np.argsort(xdata)
    (event_left, event_right) = walk_sc_old(xdata[sort_key], ydata[sort_key], number_samples, sampling_width, number_of_windows, slew_rate, plane=plane)
    print('SC fit gives event_left={} and event_right={}'.format(event_left, event_right))
    print('Diagnostics: The input into curve_fit is as follows:')
    print('\txdata size: {}, ydata size: {}, xdata NaN: {}, ydata NaN: {}'.format(
        xdata[sort_key][event_left:event_right].size,
        ydata[sort_key][event_left:event_right].size,
        nsum(np.isnan(xdata[sort_key][event_left:event_right])),
        nsum(np.isnan(ydata[sort_key][event_left:event_right]))))
    xvalues = xdata[sort_key][event_left:event_right]
    yvalues = ydata[sort_key][event_left:event_right]
    ysigma = sigma_y[sort_key][event_left:event_right]
    # print('The values of x, y, and sigmaY are: {} and {} and {}'.format(xvalues, yvalues, ysigma))
    m0 = (yvalues[-1] - yvalues[0])/(xvalues[-1] - xvalues[0])
    p0 = (m0, 0)
    result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, sigma=ysigma, absolute_sigma=True, p0=p0, method='trf')
    # result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, p0=(38, 0), method='trf')
    perr = np.sqrt(np.diag(pcov))
    print('The sc fit result is: {}'.format(result))
    # In order to properly plot the superconducting branch fit try to find the boundaries of the SC region
    # One possibility is that the region has the smallest and largest y-value excursions. However this may not be the case
    # and certainly unless the data is sorted these indices are meaningless to use in a slice
    # index_y_min = np.argmin(y)
    # index_y_max = np.argmax(y)
    return result, perr  # index_y_max, index_y_min)


def fit_normal_branches(xdata, ydata, sigma_y, number_samples, sampling_width, number_of_windows, slew_rate):
    '''Walk and fit the normal branches in the vOut vs iBias plane.'''
    # Flatten if necessary
    # Flatten if necessary:
    if len(xdata.shape) == 2:
        print('Flattening is necessary')
        number_samples = xdata.shape[1]
        xdata = xdata.flatten()
        ydata = ydata.flatten()
        sigma_y = sigma_y.flatten()
    if len(sigma_y.shape) == 1:
        sigma_y = np.zeros(xdata.size) + sigma_y
    if np.any(sigma_y == 0):
        print('zeros detected in sigma_y')
        print('The result of all check is: {}'.format(np.all(sigma_y == 0)))
        print('Sigma y is: {}'.format(sigma_y))
        cut = sigma_y == 0
        xdata = xdata[cut]
        ydata = ydata[cut]
        sigma_y = sigma_y[cut]
    # Generate a sort_key since dy/dx must be sorted
    sort_key = np.argsort(xdata)
    # Get the left side normal branch first
    side = 'left'
    left_ev = walk_normal(xdata[sort_key], ydata[sort_key], side, number_samples, sampling_width, number_of_windows, slew_rate)
    xvalues = xdata[sort_key][0:left_ev]
    yvalues = ydata[sort_key][0:left_ev]
    ysigmas = sigma_y[sort_key][0:left_ev]
    # cut = ysigmas > 0
    # (m, b)
    m0 = (yvalues[-1] - yvalues[0])/(xvalues[-1] - xvalues[0])
    m0 = m0 if not np.isinf(np.abs(m0)) else 1
    p0 = (m0, 0)
    print('The left initial point is: {}'.format(p0))

    left_result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, sigma=ysigmas, absolute_sigma=True, p0=p0, method='trf')
    left_perr = npsqrt(np.diag(pcov))
    # Now get the other branch
    side = 'right'
    right_ev = walk_normal(xdata[sort_key], ydata[sort_key], side, number_samples, sampling_width, number_of_windows, slew_rate)
    xvalues = xdata[sort_key][right_ev:]
    yvalues = ydata[sort_key][right_ev:]
    ysigmas = sigma_y[sort_key][right_ev:]
    # cut = ysigmas > 0
    # (m, b)
    m0 = (yvalues[-1] - yvalues[0])/(xvalues[-1] - xvalues[0])
    m0 = m0 if not np.isinf(np.abs(m0)) else 1
    p0 = (m0, 0)
    print('The right initial point is: {}'.format(p0))
    right_result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, sigma=ysigmas, absolute_sigma=True, p0=p0, method='trf')
    right_perr = np.sqrt(np.diag(pcov))
    return left_result, left_perr, right_result, right_perr


@jit(nopython=True)
def get_normal_endpoints(buffer_size, dydx):
    '''Get the normal branch endpoints'''

    # In the normal region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time.
    # If the new average differs from the previous by some amount mark that as the boundary to the bias region
    dbuff = RingBuffer(buffer_size, 0, np.float32)
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
    return event


def walk_normal(xdata, ydata, side, number_samples, sampling_width, number_of_windows, slew_rate=8, delta_current=70):
    '''Function to walk the normal branches and find the line fit
    To do this we will start at the min or max input current and compute a walking derivative
    If the derivative starts to change then this indicates we entered the biased region and should stop
    NOTE: We assume data is sorted by voltage values
    '''

    print('The shape of xdata in walk_normal is: {}'.format(xdata.shape))
    # Ensure we have the proper sorting of the data
    if not np.all(xdata[:-1] <= xdata[1:]):
        raise pyTESErrors.ArrayIsUnsortedException('Input argument x is unsorted')
    # We should walk at least 100 uA
    buffer_size = int((delta_current / slew_rate) / ((number_samples * sampling_width) / number_of_windows))
    print('For a delta current of {} uA with a ramp slew rate of {} uA/s, the buffer requires {} windowed points'.format(delta_current, slew_rate, buffer_size))
    print('For reference the shape of the data is: {}'.format(xdata.shape))
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
    event = get_normal_endpoints(buffer_size, dydx)
    if side == 'right':
        # Flip event index back the right way
        event = dydx.size - 1 - event
    # print('The {} deviation occurs at ev = {} with current = {} and voltage = {} with dMean = {}'.format(side, ev, current[ev], voltage[ev], dMean))
    return event


def walk_sc_old(xdata, ydata, number_samples, sampling_width, number_of_windows, slew_rate, delta_current=None, plane='iv'):
    '''Function to walk the superconducting region of the IV curve and get the left and right edges
    Generally when ib = 0 we should be superconducting so we will start there and go up until the bias
    then return to 0 and go down until the bias
    In order to be correct your x and y data values must be sorted by x
    '''
    # Ensure we have the proper sorting of the data
    if np.all(xdata[:-1] <= xdata[1:]) is False:
        raise pyTESErrors.ArrayIsUnsortedException('Input argument x is unsorted')
    # TODO: Adjust buffer size to reflect the amount of xData points (different based on the plane we live in)
    # Check buffer size
    if delta_current is None:
        if plane == 'iv':
            delta_current = 15
        elif plane == 'tes':
            delta_current = 15
        else:
            delta_current = 10
    buffer_size = int((delta_current / slew_rate) / ((number_samples * sampling_width) / number_of_windows))
    print('For a delta current of {} uA with a rampe slew rate of {} uA/s, the buffer requires {} windowed points'.format(delta_current, slew_rate, buffer_size))
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
        xdata[~c_normal_to_sc] = np.nan
        ydata[~c_normal_to_sc] = np.nan
        dydx[~c_normal_to_sc] = np.nan
        print('Setting things to nan')

    # In the sc region the gradient should be constant
    # So we will walk along and compute the average of N elements at a time.
    # If the new average differs from the previous by some amount mark that as the end.

    # First we should find whereabouts of (0,0)
    # This should roughly correspond to x = 0 since if we input nothing we should get out nothing. In reality there are parasitics of course
    if plane == 'tes':
        # Ideally we should look for the point that is closest to (0, 0)!
        distance = np.zeros(xdata.size)
        px, py = (0, 0)
        for idx in range(xdata.size):
            dx = xdata[idx] - px
            dy = ydata[idx] - py
            distance[idx] = np.sqrt(dx**2 + dy**2)
        index_min_x = np.nanargmin(distance)
        print('The point closest to ({}, {}) is at index {} with distance {} and is ({}, {})'.format(
            px,
            py,
            index_min_x,
            distance[index_min_x],
            xdata[index_min_x],
            ydata[index_min_x]))
        # Occasionally we may have a shifted curve that is not near 0 for some reason (SQUID jump)
        # So find the min and max iTES and then find the central point
    elif plane == 'iv':
        # Find the point closest to 0 iBias.
        ioffset = 0
        index_min_x = np.nanargmin(np.abs(xdata + ioffset))
        # NOTE: The above will fail for small SC regions where vOut normal > vOut sc!!!!
    # Start by walking buffer_size events to the right from the minimum abs. voltage
    print('The size of dydx is: {}'.format(dydx.size))
    event_values = get_sc_endpoints_old(buffer_size, index_min_x, dydx)
    return event_values


@jit(nopython=True)
def get_sc_endpoints_old(buffer_size, index_min_x, dydx):
    '''A function to try and determine the endpoints for the SC region'''
    # Look for rightmost endpoint, keeping in mind it could be our initial point
    if buffer_size + index_min_x >= dydx.size:
        # Buffer size and offset would go past end of data
        right_buffer_size = np.nanmax([dydx.size - index_min_x - 1, 0])
    else:
        right_buffer_size = buffer_size
    right_buffer_size = np.int32(right_buffer_size)
    slope_buffer = RingBuffer(right_buffer_size, np.int32(0), np.float32)
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
    # print('We will create a ringbuffer with size: {}'.format(left_buffer_size))
    left_buffer_size = np.int32(left_buffer_size)
    slope_buffer = RingBuffer(left_buffer_size, 0, np.float32)
    # Do initial appending
    for event in range(left_buffer_size):
        slope_buffer.append(dydx[index_min_x - event])
    # Walk to the left
    ev_left = index_min_x - left_buffer_size
    difference_of_means = 0
    # print('The value of ev_left to start is: {}'.format(ev_left))
    while difference_of_means < 1e-2 and ev_left >= 0:
        current_mean = slope_buffer.get_nanmean()
        slope_buffer.append(dydx[ev_left])
        new_mean = slope_buffer.get_nanmean()
        difference_of_means = np.abs((current_mean - new_mean)/current_mean)
        ev_left -= 1
    ev_left = ev_left if ev_left >= 0 else ev_left + 1
    return (ev_left, ev_right)


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
