'''IV Processing Module'''

import numpy as np
from numpy import square as pow2
from numpy import sqrt as npsqrt
from numpy import sum as nsum
from numba import jit

from scipy.optimize import curve_fit

import tes_fit_functions as fitfuncs

import iv_results
import iv_resistance
import squid_info
import pytes_errors as pyTESErrors
from ring_buffer import RingBuffer


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


def fit_sc_branch(xdata, ydata, sigma_y, plane):
    '''Walk and fit the superconducting branch
    In the vOut vs iBias plane x = iBias, y = vOut --> dy/dx ~ resistance
    In the iTES vs vTES plane x = vTES, y = iTES --> dy/dx ~ 1/resistance
    '''
    # First generate a sort_key since dy/dx will require us to be sorted
    # Flatten if necessary:
    if len(xdata.shape) == 2:
        xdata = xdata.flatten()
        ydata = ydata.flatten()
        sigma_y = sigma_y.flatten()
    if len(sigma_y.shape) == 1:
        sigma_y = np.zeros(xdata.size) + sigma_y
    sort_key = np.argsort(xdata)
    (event_left, event_right) = walk_sc(xdata[sort_key], ydata[sort_key], plane=plane)
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
    result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, sigma=ysigma, absolute_sigma=True, method='trf')
    # result, pcov = curve_fit(fitfuncs.lin_sq, xvalues, yvalues, p0=(38, 0), method='trf')
    perr = np.sqrt(np.diag(pcov))
    # In order to properly plot the superconducting branch fit try to find the boundaries of the SC region
    # One possibility is that the region has the smallest and largest y-value excursions. However this may not be the case
    # and certainly unless the data is sorted these indices are meaningless to use in a slice
    # index_y_min = np.argmin(y)
    # index_y_max = np.argmax(y)
    return result, perr  # index_y_max, index_y_min)


def fit_normal_branches(xdata, ydata, sigma_y):
    '''Walk and fit the normal branches in the vOut vs iBias plane.'''
    # Flatten if necessary
    # Flatten if necessary:
    if len(xdata.shape) == 2:
        xdata = xdata.flatten()
        ydata = ydata.flatten()
        sigma_y = sigma_y.flatten()
    if len(sigma_y.shape) == 1:
        sigma_y = np.zeros(xdata.size) + sigma_y
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


def walk_normal(xdata, ydata, side, buffer_size=40*16):
    '''Function to walk the normal branches and find the line fit
    To do this we will start at the min or max input current and compute a walking derivative
    If the derivative starts to change then this indicates we entered the biased region and should stop
    NOTE: We assume data is sorted by voltage values
    '''
    # Ensure we have the proper sorting of the data
    if not np.all(xdata[:-1] <= xdata[1:]):
        raise pyTESErrors.ArrayIsUnsortedException('Input argument x is unsorted')
    # Check buffer is at least 5% of the data size
    if buffer_size < 0.05*xdata.size:
        buffer_size = int(0.05*xdata.size)
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
    # print('The {} deviation occurs at ev = {} with current = {} and voltage = {} with dMean = {}'.format(side, ev, current[ev], voltage[ev], dMean))
    return event

@jit(nopython=True)
def walk_sc(xdata, ydata, buffer_size=5*16, plane='iv'):
    '''Function to walk the superconducting region of the IV curve and get the left and right edges
    Generally when ib = 0 we should be superconducting so we will start there and go up until the bias
    then return to 0 and go down until the bias
    In order to be correct your x and y data values must be sorted by x
    '''
    # Ensure we have the proper sorting of the data
    if np.all(xdata[:-1] <= xdata[1:]) is False:
        raise pyTESErrors.ArrayIsUnsortedException('Input argument x is unsorted')
    # Check buffer size
    if buffer_size < 0.05*xdata.size:
        buffer_size = int(0.05*xdata.size)
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
    event_values = get_sc_endpoints(buffer_size, index_min_x, dydx)
    return event_values


@jit(nopython=True)
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
