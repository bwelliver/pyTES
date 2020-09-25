'''Module to handle processing aggregate TES data'''

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from numba import jit

from iv_processor import iv_windower
import iv_results
import iv_plots as ivplt
import tes_fit_functions as fitfuncs


def normal_to_sc_cut_constructor(timestamps, start_times, end_times):
    '''Constructor for master cut'''
    master_cut = np.zeros(timestamps.size, dtype=np.bool)
    idx = 0
    for t0, t1 in np.nditer((start_times, end_times)):
        idx = idx + 1
        if t0 == t1:
            # Single point fluctuation...can't really do much
            continue
        if idx == start_times.size:
            continue
        cut = np.logical_and(timestamps >= t0 - 0.5, timestamps <= t1 - 0.5)
        master_cut = np.logical_or(master_cut, cut)
    print('The shape of the master_cut is: {}'.format(master_cut.shape))
    return master_cut


def find_normal_to_sc_data(iv_dictionary, number_of_windows, iv_curves=None):
    '''A function to try and locate time boundaries to define N --> SC data'''
    # Step 1: Average using the number of windows\
    if iv_curves is None:
        iv_curves = iv_windower(iv_dictionary, number_of_windows, mode='tes')
    for temperature, iv_data in iv_curves.items():
        dbias = np.gradient(iv_data['iBias'].flatten(), edge_order=2)
        cut1 = np.logical_and(iv_data['iBias'].flatten() > 0, dbias < 0)   # Positive iBias -slope (High to Low, N-->Sc)
        cut2 = np.logical_and(iv_data['iBias'].flatten() <= 0, dbias >= 0)  # Negative iBias +slope (-High to -Low, N-->Sc)
        cut_norm_to_sc = np.logical_or(cut1, cut2)
        timestamps = iv_data['timestamps'].flatten()
        boundaries = np.where(cut_norm_to_sc[:-1] != cut_norm_to_sc[1:])[0]
        print(boundaries)
        # boundaries now contain the index where a change occurs. We want to ultimately get at list of (t0, t1)
        # pairs such that all data between each (t0, t1) pair is 'true' for this cut.
        # If the first value of cut is False, then the first boundary index + 1 is our first True value.
        # From here we go until the next boundary index, which is the last True. Then the next boundary index + 1 is our next
        # first True value and so on.
        # If however the first value of the cut is True, then index 0 is our first True and the first boundary is our last true
        # so the pair is (0, boundaries[0]). The next boundary index+1 then is the first True and subsequent one is last True.
        # First append the last index to properly span
        #if cut_norm_to_sc[0]:
        #boundaries = np.append(-1, boundaries)
        ###
        # Another approach is to use the boundaries and cut itself to select values directly
        # Here make use of the fact that the indices in boundaries represent the index at which a change occurs
        # so these are the final index of a particular value...thus boundaries+1 are the indices where a new value
        # starts.
        # To select the time value when we are first True:
        # So arr[boundaries+1] gives all the potential 'starting' values.
        # If we apply this to the cut as well, the filtered cut[boundaries+1] will tell us whether a value starts as a True or False
        # Thus: arr[boundaries+1][cut[boundaries+1]] selects the first True values.
        # To select the time value when we are last True:
        # Again note that boundaries corresponds to the last value before a change thus
        # arr[boundaries] gives all potential ending values. We want to select only end values that are True
        # thus we should simply filter the cut on these boundaries as well.
        # arr[boundaries][cut[boundaries]] selects the last True values.

        # Important caveats:
        #   Last value is True:
        #       If the last value is True, it won't be identified as having
        #       an ending point, thus we should always append the last index to boundaries
        #       after we select the True starting values (otherwise idx+1 will be out of bounds)
        #   First value is True:
        #       If the first value is True, it won't be identified as having a starting point.
        #       Thus we should always prepend the value -1 to the boundaries when selecting the t0 values
        nboundaries = np.append(-1, boundaries)
        t0 = timestamps[nboundaries+1][cut_norm_to_sc[nboundaries+1]]
        nboundaries = np.append(boundaries, timestamps.size - 1)
        t1 = timestamps[nboundaries][cut_norm_to_sc[nboundaries]]
        # Next iterate the bounds to construct a master cut
        iv_dictionary[temperature]['cut_norm_to_sc'] = normal_to_sc_cut_constructor(iv_dictionary[temperature]['timestamps'], t0, t1)
    return iv_dictionary


def debugger_RT(iv_data, fixed_name, fixed_value, delta_values, output_path, data_channel, temperature):
    """Debug the RT curves if something is weird."""
    timestamps0 = iv_data['timestamps'][0]
    timestamps = iv_data['timestamps'] - timestamps0
    sample_width = iv_data['sampling_width'][0]
    cut_norm_to_sc = iv_data['cut_norm_to_sc']
    cut_sc_to_norm = ~iv_data['cut_norm_to_sc']
    iBias = iv_data['iBias']
    iTES = iv_data['iTES']
    rTES = iv_data['rTES']
    vOut = iv_data['vOut']
    sample_times = np.tile([i*sample_width for i in range(iBias.shape[1])], [timestamps.size, 1])
    full_timestamps = sample_times + timestamps[:, None]
    ts = full_timestamps[cut_norm_to_sc].flatten()
    iBias = iBias[cut_norm_to_sc].flatten()
    rTES = rTES[cut_norm_to_sc].flatten()
    iTES = iTES[cut_norm_to_sc].flatten()
    fixed_cut = np.logical_and(iv_data[fixed_name][cut_norm_to_sc] > fixed_value - delta_values[0], iv_data[fixed_name][cut_norm_to_sc] < fixed_value + delta_values[1])
    fixed_cut = fixed_cut.flatten()
    fixed_cut = np.logical_and(fixed_cut, ts < 2e2)
    print('The shape of timestamps is: {} and the shape of iBias is: {}'.format(ts.shape, iBias.shape))
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
              'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': None
              }
    axes_options = {'xlabel': 'Time', 'ylabel': 'Bias Current [uA]',
                    'title': 'Channel {} Output Voltage vs t for temperatures = {} mK'.format(data_channel, temperature)}

    axes = ivplt.generic_fitplot_with_errors(axes=axes, x=ts[fixed_cut], y=rTES[fixed_cut], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)

    fixed_cut = np.logical_and(iv_data[fixed_name][cut_sc_to_norm] > fixed_value - delta_values[0], iv_data[fixed_name][cut_sc_to_norm] < fixed_value + delta_values[1])
    fixed_cut = fixed_cut.flatten()
    ts = full_timestamps[cut_sc_to_norm].flatten()
    iBias = iv_data['iBias'][cut_sc_to_norm].flatten()
    rTES = iv_data['rTES'][cut_sc_to_norm].flatten()
    iTES = iv_data['iTES'][cut_sc_to_norm].flatten()
    fixed_cut = np.logical_and(fixed_cut, ts < 2e2)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'red', 'markerfacecolor': 'red',
              'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': None
              }
    ivplt.generic_fitplot_with_errors(axes=axes, x=ts[fixed_cut], y=rTES[fixed_cut], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'vOut_vs_t_ch_' + str(data_channel) + '_' + temperature + 'mK'
    ivplt.save_plot(fig, axes, file_name)
    raise Exception('Debug halt')


def get_RT_values(iv_dictionary, fixed_name, fixed_value, delta_values, cut_name, data):
    """Loop over iv data and return R and T values for use in fitting."""
    for temperature, iv_data in iv_dictionary.items():
        # This is not good with the un-windowed data because of noise fluctuations
        # It is probably necessary to window the data and extract time boundaries for each case
        if cut_name == 'normal_to_sc':
            direction_cut = iv_data['cut_norm_to_sc']
        else:
            direction_cut = ~iv_data['cut_norm_to_sc']
        # Cuts get complicated. We will need to make a cut on a cut.
        # fixed_cut = np.logical_and(iv_data[fixed_name] > fixed_value - delta_values[0], iv_data[fixed_name] < fixed_value + delta_values[1])
        # fixed cut is (nEvents, nSamples)
        # cut_norm_to_sc is (nEvents, )
        # ultimately we will need to do data[cut_norm_to_sc][fixed_cut[cut_norm_to_sc]]
        # This means fixed_cut[cut_norm_to_sc] is cut_fixed_norm_to_sc now
        # Test plot for iBias vs time
        cut_fixed = np.logical_and(iv_data[fixed_name][direction_cut] > fixed_value - delta_values[0], iv_data[fixed_name][direction_cut] < fixed_value + delta_values[1])
        cut_fixed = np.logical_and(cut_fixed, iv_data['rTES'][direction_cut] > -100e-3)
        cut_fixed = cut_fixed.flatten()
        if cut_fixed.sum() > 0:
            data['T'] = np.append(data['T'], float(temperature)*1e-3)
            rTES = iv_data['rTES'][direction_cut].flatten()
            data['R'] = np.append(data['R'], np.mean(rTES[cut_fixed]))
            data['rmsR'] = np.append(data['rmsR'], np.std(rTES[cut_fixed])/np.sqrt(cut_fixed.sum()))
    return data


def get_corrected_RT_values(iv_dictionary, fixed_name, fixed_value, delta_values, cut_name, data, ptdata):
    """Loop over iv data and return R and T values for use in fitting."""
    for temperature, iv_data in iv_dictionary.items():
        # This is not good with the un-windowed data because of noise fluctuations
        # It is probably necessary to window the data and extract time boundaries for each case
        # Generate an estimate of Ttes from pTES and Tbath
        if cut_name == 'normal_to_sc':
            direction_cut = iv_data['cut_norm_to_sc']
        else:
            direction_cut = ~iv_data['cut_norm_to_sc']
        # Cuts get complicated. We will need to make a cut on a cut.
        # fixed_cut = np.logical_and(iv_data[fixed_name] > fixed_value - delta_values[0], iv_data[fixed_name] < fixed_value + delta_values[1])
        # fixed cut is (nEvents, nSamples)
        # cut_norm_to_sc is (nEvents, )
        # ultimately we will need to do data[cut_norm_to_sc][fixed_cut[cut_norm_to_sc]]
        # This means fixed_cut[cut_norm_to_sc] is cut_fixed_norm_to_sc now
        # Test plot for iBias vs time
        cut_fixed = np.logical_and(iv_data[fixed_name][direction_cut] > fixed_value - delta_values[0], iv_data[fixed_name][direction_cut] < fixed_value + delta_values[1])
        cut_fixed = np.logical_and(cut_fixed, iv_data['rTES'][direction_cut] > -100e-3)
        cut_fixed = cut_fixed.flatten()
        if cut_fixed.sum() > 0:
            k = ptdata['fit']['k']
            n = ptdata['fit']['n']
            params = [k, n, float(temperature)*1e-3]
            pTES = iv_data['pTES'][direction_cut].flatten()
            Ttes = fitfuncs.tes_temperature_polynomial(pTES, *params)
            data['T'] = np.append(data['T'], np.mean(Ttes[cut_fixed]))
            rTES = iv_data['rTES'][direction_cut].flatten()
            data['R'] = np.append(data['R'], np.mean(rTES[cut_fixed]))
            data['rmsR'] = np.append(data['rmsR'], np.std(rTES[cut_fixed])/np.sqrt(cut_fixed.sum()))
    return data


def compute_alpha(temperatures, fit_result, model_function, dmodel_function):
    """Compute the value of alpha at various temperatures and resistances."""
    # The computation of alpha = dlog(R)/dlog(T) = T/R * dR/dT can be done
    # either through an empirical derivative if the model function allows it
    # or numerically.

    # Compute alpha purely from fit functions
    T = np.linspace(temperatures.min(), temperatures.max(), 1000)

    R = model_function(T, *fit_result.normal.result)
    alpha = (T/R) * dmodel_function(T, *fit_result.normal.result)
    print('The input temperature ranges were: {}'.format([temperatures.min(), temperatures.max()]))
    print('The input parameters are: {}'.format(fit_result.normal.result))
    print('The max value for alpha is: {}'.format(alpha.max()))
    return alpha


def get_corrected_resistance_temperature_curves(output_path, data_channel, number_of_windows, iv_dictionary, pt_data):
    '''Generate resistance vs temperature curves for a TES'''

    # First window the IV data as need be
    # iv_curves = iv_windower(iv_dictionary, number_of_windows, mode='tes')
    # Rtes = R(i,T) so we are really asking for R(i=constant, T).
    # iv_dictionary = find_normal_to_sc_data(iv_dictionary, number_of_windows)
    fixed_name = 'iTES'
    fixed_value = 4.8e-6
    delta_values = [0.1e-6, 0.3e-6]
    r_normal = 0.500

    norm_to_sc = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    sc_to_norm = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    norm_to_sc = get_corrected_RT_values(iv_dictionary, fixed_name, fixed_value, delta_values, 'normal_to_sc', norm_to_sc, pt_data)
    sc_to_norm = get_corrected_RT_values(iv_dictionary, fixed_name, fixed_value, delta_values, 'sc_to_normal', sc_to_norm, pt_data)
    # Now we have arrays of R and T for a fixed iTES so try to fit each domain
    # SC --> N first
    # Model function is a modified tanh(Rn, Rp, Tc, Tw)
    model_func = fitfuncs.exp_tc #fitfuncs.tanh_tc
    dmodel_func = fitfuncs.dexp_tc #fitfuncs.dtanh_tc
    fit_result = iv_results.FitParameters('rt')
    # Try to do a smart Tc0 estimate:
    # Note that Ttes may  not be Tbath -- use P-T curve to extract Ttes given Ptes and Tbath.
    sort_key = np.argsort(norm_to_sc['T'])
    T0 = norm_to_sc['T'][sort_key][np.gradient(norm_to_sc['R'][sort_key], norm_to_sc['T'][sort_key], edge_order=2).argmax()]*1.01
    x_0 = [0.6, 1e-3, T0, 1e-3]
    lbounds = (0, 0, 0, 0)
    ubounds = (2, 2, norm_to_sc['T'].max(), norm_to_sc['T'].max())

    print('For SC to N fit initial guess is {}, and the number of data points are: {}'.format(x_0, sc_to_norm['T'].size))
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True,
               'sigma': sc_to_norm['rmsR'], 'method': 'trf', 'jac': '3-point',
               'xtol': 1e-15, 'ftol': 1e-8, 'loss': 'linear', 'tr_solver': 'exact',
               'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, sc_to_norm['T'], sc_to_norm['R'], **fitargs)
    perr = np.sqrt(np.diag(pcov))
    print('Ascending (SC -> N): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.sc.set_values(result, perr)

    # Attempt to fit the N-->Sc region now
    print('For N to SC fit initial guess is {}, and the number of data points are: {}'.format(x_0, norm_to_sc['T'].size))
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True,
               'sigma': norm_to_sc['rmsR'], 'method': 'trf', 'jac': '3-point',
               'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'soft_l1', 'tr_solver': 'exact',
               'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    #fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'linear', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, norm_to_sc['T'], norm_to_sc['R'], **fitargs)
    perr = np.sqrt(np.diag(pcov))
    print('Descending (N -> SC): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.normal.set_values(result, perr)
    rN = result[0]
    tc = result[2]
    # Get alpha values
    alpha = compute_alpha(norm_to_sc['T'], fit_result, model_func, dmodel_func)
    # Make output plot
    ivplt.make_corrected_resistance_vs_temperature_plots(output_path, data_channel, fixed_name, fixed_value, norm_to_sc, sc_to_norm, alpha, model_func, fit_result)
    return tc, rN, norm_to_sc['T'], norm_to_sc['R'], norm_to_sc['rmsR']


def get_resistance_temperature_curves_new(output_path, data_channel, number_of_windows, iv_dictionary):
    '''Generate resistance vs temperature curves for a TES'''

    # First window the IV data as need be
    # iv_curves = iv_windower(iv_dictionary, number_of_windows, mode='tes')
    # Rtes = R(i,T) so we are really asking for R(i=constant, T).
    # iv_dictionary = find_normal_to_sc_data(iv_dictionary, number_of_windows)
    fixed_name = 'iTES'
    fixed_value = 0.5e-6
    delta_values = [0.1e-6, 0.3e-6]
    r_normal = 0.500

    norm_to_sc = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    sc_to_norm = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    norm_to_sc = get_RT_values(iv_dictionary, fixed_name, fixed_value, delta_values, 'normal_to_sc', norm_to_sc)
    sc_to_norm = get_RT_values(iv_dictionary, fixed_name, fixed_value, delta_values, 'sc_to_normal', sc_to_norm)
    # Now we have arrays of R and T for a fixed iTES so try to fit each domain
    # SC --> N first
    # Model function is a modified tanh(Rn, Rp, Tc, Tw)
    model_func = fitfuncs.exp_tc #fitfuncs.tanh_tc
    dmodel_func = fitfuncs.dexp_tc #fitfuncs.dtanh_tc
    fit_result = iv_results.FitParameters('rt')
    # Try to do a smart Tc0 estimate:
    sort_key = np.argsort(norm_to_sc['T'])
    T0 = norm_to_sc['T'][sort_key][np.gradient(norm_to_sc['R'][sort_key], norm_to_sc['T'][sort_key], edge_order=2).argmax()]*1.01
    x_0 = [0.6, 1e-3, T0, 1e-3]
    lbounds = (0, 0, 0, 0)
    ubounds = (2, 2, norm_to_sc['T'].max(), norm_to_sc['T'].max())

    print('For SC to N fit initial guess is {}, and the number of data points are: {}'.format(x_0, sc_to_norm['T'].size))
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True,
               'sigma': sc_to_norm['rmsR'], 'method': 'trf', 'jac': '3-point',
               'xtol': 1e-15, 'ftol': 1e-8, 'loss': 'linear', 'tr_solver': 'exact',
               'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, sc_to_norm['T'], sc_to_norm['R'], **fitargs)
    perr = np.sqrt(np.diag(pcov))
    print('Ascending (SC -> N): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.sc.set_values(result, perr)

    # Attempt to fit the N-->Sc region now
    print('For N to SC fit initial guess is {}, and the number of data points are: {}'.format(x_0, norm_to_sc['T'].size))
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True,
               'sigma': norm_to_sc['rmsR'], 'method': 'trf', 'jac': '3-point',
               'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'soft_l1', 'tr_solver': 'exact',
               'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    #fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'linear', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, norm_to_sc['T'], norm_to_sc['R'], **fitargs)
    perr = np.sqrt(np.diag(pcov))
    print('Descending (N -> SC): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.normal.set_values(result, perr)
    rN = result[0]
    tc = result[2]
    # Get alpha values
    alpha = compute_alpha(norm_to_sc['T'], fit_result, model_func, dmodel_func)
    # Make output plot
    ivplt.make_resistance_vs_temperature_plots(output_path, data_channel, fixed_name, fixed_value, norm_to_sc, sc_to_norm, alpha, model_func, fit_result)
    return tc, rN, norm_to_sc['T'], norm_to_sc['R'], norm_to_sc['rmsR']


def get_power_temperature_curves(output_path, data_channel, number_of_windows, iv_dictionary, tc=None, rN=500e-3):
    '''Generate a power vs temperature curve for a TES'''
    # Need to select power in the biased region, i.e. where P(R) ~ constant
    # Try something at 0.5*Rn
    # iv_dictionary = find_normal_to_sc_data(iv_dictionary, number_of_windows)
    R = 0.85*rN
    deltaR = 20e-3
    print('The resistance range selected is: {} +/- {} mOhms'.format(R, deltaR))
    temperatures = np.empty(0)
    power = np.empty(0)
    power_rms = np.empty(0)
    for temperature, iv_data in iv_dictionary.items():
        cut_norm_to_sc = iv_data['cut_norm_to_sc'] # (nEvents, )
        # This cut is computed on windowed data to avoid spikey behavior and expanded back to normal size.
        # The shape is (nEvents, ) whereas pTES has a shape of (nEvents, nSamples)
        # Application of the cut will return a subset of the 2d pTES array.
        # Step 1: obtain rTES waveforms
        rTES = iv_data['rTES']
        # Step 2: each indvidual average (nEvents, 1) and select those with average R in range
        rTES_mean = np.mean(rTES, axis=1, keepdims=True)
        print('The shape of rTES_mean is: {}'.format(rTES_mean.shape))
        cut_R = np.logical_and(rTES_mean > R - deltaR, rTES_mean < R + deltaR).flatten() # Make shape (nEvents, ) instead of (nEvents, 1)
        # Step 3: Construct a cut to select powers with the events having R in range and ascending
        select_cut = np.logical_and(cut_norm_to_sc, cut_R) # If these are not both (nEvents, ) then we get (nEvents, nEvents)
        if select_cut.sum() > 0:
            temperatures = np.append(temperatures, float(temperature)*1e-3)
            pTES = iv_data['pTES'][select_cut]  # (nCut, nSamples)
            print('The shape of pTES after select cut is: {}'.format(pTES.shape))
            pTES_mean = np.mean(pTES, axis=1)
            pTES_rms = np.std(pTES, axis=1)/np.sqrt(pTES.shape[1])
            # combine these
            pTES_value = np.mean(pTES_mean)
            pTES_value_rms = np.std(pTES_mean)
            pTES_value = np.sqrt(np.mean(pTES*pTES)) # RMS^2 = mean(p^2) == mean(p)^2 + sigma(p)^2
            pTES_value_rms = np.std(pTES)
            pTES_value_rms = np.sqrt(np.sum(pTES_rms*pTES_rms))
            power = np.append(power, pTES_value)
            power_rms = np.append(power_rms, pTES_value_rms)
        # Cuts get complicated. We will need to make a cut on a cut.
        # cut_fixed_norm_to_sc = np.logical_and(iv_data['rTES'][cut_norm_to_sc] > R - deltaR, iv_data['rTES'][cut_norm_to_sc] < R + deltaR)
        # cut_fixed_norm_to_sc = cut_fixed_norm_to_sc.flatten()
        # if cut_fixed_norm_to_sc.sum() > 0:
        #     temperatures = np.append(temperatures, float(temperature)*1e-3)
        #     pTES = iv_data['pTES'][cut_norm_to_sc].flatten()
        #     pTES_rms = np.std(iv_data['pTES'][cut_norm_to_sc].flatten())
        #     pTES_rms = pTES_rms/np.sqrt(cut_fixed_norm_to_sc.sum())
        #     power = np.append(power, np.mean(pTES[cut_fixed_norm_to_sc]))
        #     # power_rms = np.append(power_rms, np.std(pTES[cut_fixed_norm_to_sc])/np.sqrt(cut_fixed_norm_to_sc.sum()))
        #     power_rms = np.append(power_rms, pTES_rms)
        else:
            print('For T = {} mK there were no values used.'.format(temperature))
    # print('The main T vector is: {}'.format(temperatures))
    # print('The iTES vector is: {}'.format(iTES))
    # TODO: Make these input values?
    #tc = None
    max_temp = tc or 60e-3
    cut_temperature = np.logical_and(temperatures > 10e-3, temperatures < max_temp)  # This should be the expected Tc
    cut_power = power < 1e-6
    cut_temperature = np.logical_and(cut_temperature, cut_power)

    # [k, n, Ttes, Pp]
    pP = None
    if tc is None:
        print('No Tc was passed, floating Tc')
        lbounds = [1e-9, 0, 42e-3]
        ubounds = [1, 10, 47e-3]
        fixedArgs = {'Pp': 0}
        x0 = [20e-9, 4, 45e-3]
    else:
        if pP is None:
            print('Tc = {} mK was passed. Fixing to this value'.format(tc))
            lbounds = [1e-9, 0]
            ubounds = [1, 10]
            fixedArgs = {'Pp': 0, 'Ttes': tc}
            x0 = [1000e-9, 5]
        else:
            print('Tc = {} mK was passed. Fixing to this value'.format(tc))
            lbounds = [1e-9, 0, -3e-5]
            ubounds = [1, 10, 3e-5]
            fixedArgs = {'Ttes': tc}
            x0 = [100e-9, 5, pP]

    ndf = np.sum(cut_temperature) - len(x0)
    # Attempt to fit it to a power function
    # fitargs = {'p0': x0, 'method': 'lm', 'maxfev': int(5e4)}
    use_sigmas = True
    if use_sigmas:
        # This fitarg will use the errors on y
        fitargs = {'p0': x0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True, 'sigma': power_rms[cut_temperature], 'method': 'trf', 'jac': '3-point', 'tr_solver': 'exact', 'x_scale': 'jac', 'xtol': 1e-15, 'ftol': 1e-15, 'gtol': None, 'loss': 'linear', 'max_nfev': 10000, 'verbose': 2}
    else:
        # This fitarg below will not use the errors on y
        fitargs = {'p0': x0, 'bounds': (lbounds, ubounds), 'method': 'trf', 'jac': '3-point', 'tr_solver': 'exact', 'x_scale': 'jac', 'xtol': 1e-15, 'ftol': 1e-15, 'gtol': None, 'loss': 'linear', 'max_nfev': 10000, 'verbose': 2}
    print('The fitargs are: {}'.format(fitargs))
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
    #fit_result.right.set_values(x0, x0)
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
                    'xlim': (10, 60),
                    'ylim': (0, ymax)
                    }
    axes = ivplt.generic_fitplot_with_errors(axes=axes, x=temperatures, y=power, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes, chisq = ivplt.add_model_fits(axes=axes, x=temperatures, y=power, model=fit_result, model_function=fitfuncs.tes_power_polynomial, xscale=xscale, yscale=yscale)
    # compute chisq
    ymodel = fitfuncs.tes_power_polynomial(temperatures[cut_temperature], *fit_result.left.result)
    r = power[cut_temperature] - ymodel
    sigma = power_rms[cut_temperature]
    chisq = np.sum((r / sigma) ** 2)
    axes = ivplt.pt_fit_textbox(axes=axes, model=fit_result, chisq=chisq, ndf=ndf)
    file_name = output_path + '/' + 'pTES_vs_T_ch_' + str(data_channel)
    #for label in axes.get_xticklabels() + axes.get_yticklabels():
    #    label.set_fontsize(32)
    ivplt.save_plot(fig, axes, file_name, dpi=150)
    print('Results: k = {}, n = {}, Tb = {}, Pp = {}'.format(*results))
    print('Error Results: k = {}, n = {}, Tb = {}, Pp = {}'.format(*perr))
    # Compute G
    # P = k*(Ts^n - T^n)
    # G = n*k*T^(n-1)
    print('G(Ttes) = {} pW/K'.format(results[0]*results[1]*np.power(results[2], results[1]-1)*1e12))
    print('G(10 mK) = {} pW/K'.format(results[0]*results[1]*np.power(10e-3, results[1]-1)*1e12))
    fitResults = {'k': results[0], 'n': results[1]}
    # Test lmfit #####

    # print('Trying lmfit')
    # from lmfit import Model
    # # tes_power_polynomial_args(T, k, n, Ttes, Pp)
    # fixedArgs = {'Ttes': tc}
    # x0 = [100e-9, 5, pP]
    # ptModel = Model(fitfuncs.tes_power_polynomial_args)
    # pars = ptModel.make_params(k=x0[0], n=x0[1], Ttes=fixedArgs['Ttes'], Pp=x0[2])
    # pars['Ttes'].vary = False
    # #pars['Pp'].vary = False
    # pars['k'].min = lbounds[0]
    # pars['k'].max = ubounds[0]
    # pars['n'].min = lbounds[1]
    # pars['n'].max = ubounds[1]
    # pars['Pp'].min = lbounds[2]
    # pars['Pp'].max = ubounds[2]
    # pars.pretty_print()
    # result = ptModel.fit(power[cut_temperature], params=pars, T=temperatures[cut_temperature], weights=1.0/power_rms[cut_temperature], method='least_squares')
    # print(result.fit_report())
    # print('Chisq: {}'.format(result.chisqr))
    # print('The covar matrix: {}'.format(result.covar))
    # results = [result.params['k'].value, result.params['n'].value, result.params['Ttes'].value, result.params['Pp'].value]
    # perr = [result.params['k'].stderr, result.params['n'].stderr, result.params['Ttes'].stderr, result.params['Pp'].stderr]
    # perr = [0 if err is None else err for err in perr]
    # print('After lmfit the results are: {} and the err are: {}'.format(results, perr))
    # x0 = [x0[0], x0[1], fixedArgs['Ttes'], x0[2]]
    # fit_result = iv_results.FitParameters()
    # fit_result.left.set_values(results, perr)
    # fit_result.right.set_values(x0, x0)
    # # Next make a P-T plot
    # fig = plt.figure(figsize=(16, 12))
    # axes = fig.add_subplot(111)
    # xscale = 1e3
    # yscale = 1e15
    # ymax = power.max()*1.05*yscale
    # params = {'marker': 'o', 'markersize': 7, 'markeredgecolor': 'black',
    #           'markerfacecolor': 'black', 'markeredgewidth': 0, 'linestyle': 'None',
    #           'xerr': None, 'yerr': power_rms*yscale
    #           }
    # axes_options = {'xlabel': 'Temperature [mK]',
    #                 'ylabel': 'TES Power [fW]',
    #                 'title': None, # 'Channel {} TES Power vs Temperature'.format(data_channel),
    #                 'xlim': (10, 60),
    #                 'ylim': (0, ymax)
    #                 }
    # axes = ivplt.generic_fitplot_with_errors(axes=axes, x=temperatures, y=power, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # axes, chisq = ivplt.add_model_fits(axes=axes, x=temperatures, y=power, model=fit_result, model_function=fitfuncs.tes_power_polynomial, xscale=xscale, yscale=yscale)
    # # compute chisq
    # ymodel = fitfuncs.tes_power_polynomial(temperatures[cut_temperature], *fit_result.left.result)
    # r = power[cut_temperature] - ymodel
    # sigma = power_rms[cut_temperature]
    # chisq = np.sum((r / sigma) ** 2)
    # axes = ivplt.pt_fit_textbox(axes=axes, model=fit_result, chisq=chisq, ndf=ndf)
    # file_name = output_path + '/' + 'pTES_vs_T_ch_' + str(data_channel) + '_lmfit'
    # #for label in axes.get_xticklabels() + axes.get_yticklabels():
    # #    label.set_fontsize(32)
    # ivplt.save_plot(fig, axes, file_name, dpi=150)
    # print('Results: k = {}, n = {}, Tb = {}, Pp = {}'.format(*results))
    # print('Error Results: k = {}, n = {}, Tb = {}, Pp = {}'.format(*perr))
    # # Compute G
    # # P = k*(Ts^n - T^n)
    # # G = n*k*T^(n-1)
    # print('G(Ttes) = {} pW/K'.format(results[0]*results[1]*np.power(results[2], results[1]-1)*1e12))
    # print('G(10 mK) = {} pW/K'.format(results[0]*results[1]*np.power(10e-3, results[1]-1)*1e12))

    return temperatures, power, power_rms, fitResults
