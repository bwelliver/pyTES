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
    print('The size of the master cut is: {}'.format(master_cut.size))
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
    return master_cut


def find_normal_to_sc_data(iv_dictionary, number_of_windows):
    '''A function to try and locate time boundaries to define N --> SC data'''
    # Step 1: Average using the number of windows
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


def get_resistance_temperature_curves_new(output_path, data_channel, number_of_windows, iv_dictionary):
    '''Generate resistance vs temperature curves for a TES'''

    # First window the IV data as need be
    # iv_curves = iv_windower(iv_dictionary, number_of_windows, mode='tes')
    # Rtes = R(i,T) so we are really asking for R(i=constant, T).
    iv_dictionary = find_normal_to_sc_data(iv_dictionary, number_of_windows)
    fixed_name = 'iTES'
    fixed_value = 0.1e-6
    delta_value = 0.05e-6
    r_normal = 0.690

    norm_to_sc = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    sc_to_norm = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    for temperature, iv_data in iv_dictionary.items():
        # This is not good with the un-windowed data because of noise fluctuations
        # It is probably necessary to window the data and extract time boundaries for each case
        # But that will be a bit tricky.
        # dbias = np.gradient(iv_data['iBias'].flatten(), edge_order=2)
        # cut1 = np.logical_and(iv_data['iBias'].flatten() > 0, dbias < 0)   # Positive iBias -slope (High to Low, N-->Sc)
        # cut2 = np.logical_and(iv_data['iBias'].flatten() <= 0, dbias > 0)  # Negative iBias +slope (-High to -Low, N-->Sc)
        # cut_norm_to_sc = np.logical_or(cut1, cut2)
        # cut_fixed_norm_to_sc = np.logical_and(fixed_cut, cut_norm_to_sc)
        # cut_fixed_sc_to_norm = np.logical_and(fixed_cut, ~cut_norm_to_sc)
        cut_norm_to_sc = iv_data['cut_norm_to_sc']
        cut_sc_to_norm = ~iv_data['cut_norm_to_sc']
        # Cuts get complicated. We will need to make a cut on a cut.
        fixed_cut = np.logical_and(iv_data[fixed_name] > fixed_value - delta_value, iv_data[fixed_name] < fixed_value + delta_value)
        # fixed cut is (nEvents, nSamples)
        # cut_norm_to_sc is (nEvents, )
        # ultimately we will need to do data[cut_norm_to_sc][fixed_cut[cut_norm_to_sc]]
        # This means fixed_cut[cut_norm_to_sc] is cut_fixed_norm_to_sc now
        ### Test plot for iBias vs time
#        debug = False
#        if debug and float(temperature) < 31:
#            timestamps0 = iv_data['timestamps'][0]
#            timestamps = iv_data['timestamps'] - timestamps0
#            sample_width = iv_data['sampling_width'][0]
#            iBias = iv_data['iBias']
#            iTES = iv_data['iTES']
#            rTES = iv_data['rTES']
#            vOut = iv_data['vOut']
#            sample_times = np.tile([i*sample_width for i in range(iBias.shape[1])], [timestamps.size, 1])
#            full_timestamps = sample_times + timestamps[:, None]
#            ts = full_timestamps[cut_norm_to_sc].flatten()
#            iBias = iBias[cut_norm_to_sc].flatten()
#            rTES = rTES[cut_norm_to_sc].flatten()
#            iTES = iTES[cut_norm_to_sc].flatten()
#            fixed_cut = np.logical_and(iv_data[fixed_name][cut_norm_to_sc] > fixed_value - delta_value, iv_data[fixed_name][cut_norm_to_sc] < fixed_value + delta_value)
#            fixed_cut = fixed_cut.flatten()
#            fixed_cut = np.logical_and(fixed_cut, ts < 2e2)
#            print('The shape of timestamps is: {} and the shape of iBias is: {}'.format(ts.shape, iBias.shape))
#            fig = plt.figure(figsize=(16, 12))
#            axes = fig.add_subplot(111)
#            xscale = 1e6
#            yscale = 1e3
#            params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
#                      'markeredgewidth': 0, 'linestyle': 'None',
#                      'xerr': None, 'yerr': None
#                      }
#            axes_options = {'xlabel': 'Time', 'ylabel': 'Bias Current [uA]',
#                            'title': 'Channel {} Output Voltage vs t for temperatures = {} mK'.format(data_channel, temperature)}
#
#            axes = ivplt.generic_fitplot_with_errors(axes=axes, x=ts[fixed_cut], y=rTES[fixed_cut], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
#
#            fixed_cut = np.logical_and(iv_data[fixed_name][cut_sc_to_norm] > fixed_value - delta_value, iv_data[fixed_name][cut_sc_to_norm] < fixed_value + delta_value)
#            fixed_cut = fixed_cut.flatten()
#            ts = full_timestamps[cut_sc_to_norm].flatten()
#            iBias = iv_data['iBias'][cut_sc_to_norm].flatten()
#            rTES = iv_data['rTES'][cut_sc_to_norm].flatten()
#            iTES = iv_data['iTES'][cut_sc_to_norm].flatten()
#            fixed_cut = np.logical_and(fixed_cut, ts < 2e2)
#            params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'red', 'markerfacecolor': 'red',
#                      'markeredgewidth': 0, 'linestyle': 'None',
#                      'xerr': None, 'yerr': None
#                      }
#            ivplt.generic_fitplot_with_errors(axes=axes, x=ts[fixed_cut], y=rTES[fixed_cut], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
#            file_name = output_path + '/' + 'vOut_vs_t_ch_' + str(data_channel) + '_' + temperature + 'mK'
#            ivplt.save_plot(fig, axes, file_name)
#            raise Exception('Debug halt')
        cut_fixed_norm_to_sc = np.logical_and(iv_data[fixed_name][cut_norm_to_sc] > fixed_value - delta_value, iv_data[fixed_name][cut_norm_to_sc] < fixed_value + delta_value)
        cut_fixed_norm_to_sc = np.logical_and(cut_fixed_norm_to_sc, iv_data['rTES'][cut_norm_to_sc] > -50e-3)
        cut_fixed_norm_to_sc = cut_fixed_norm_to_sc.flatten()
        if cut_fixed_norm_to_sc.sum() > 0:
            norm_to_sc['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
            rTES = iv_data['rTES'][cut_norm_to_sc].flatten()
            norm_to_sc['R'] = np.append(norm_to_sc['R'], np.mean(rTES[cut_fixed_norm_to_sc]))
            norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], np.std(rTES[cut_fixed_norm_to_sc])/np.sqrt(cut_fixed_norm_to_sc.sum()))
        cut_fixed_sc_to_norm = np.logical_and(iv_data[fixed_name][cut_sc_to_norm] > fixed_value - delta_value, iv_data[fixed_name][cut_sc_to_norm] < fixed_value + delta_value)
        cut_fixed_sc_to_norm = np.logical_and(cut_fixed_sc_to_norm, iv_data['rTES'][cut_sc_to_norm] > -50e-3)
        cut_fixed_sc_to_norm = cut_fixed_sc_to_norm.flatten()
        if cut_fixed_sc_to_norm.sum() > 0:
            sc_to_norm['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
            rTES = iv_data['rTES'][cut_sc_to_norm].flatten()
            sc_to_norm['R'] = np.append(norm_to_sc['R'], np.mean(rTES[cut_fixed_sc_to_norm]))
            sc_to_norm['rmsR'] = np.append(norm_to_sc['rmsR'], np.std(rTES[cut_fixed_sc_to_norm])/np.sqrt(cut_fixed_sc_to_norm.sum()))
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
    ubounds = (np.inf, np.inf, norm_to_sc['T'].max(), norm_to_sc['T'].max())

    print('For SC to N fit initial guess is {}, and the number of data points are: {}'.format(x_0, sc_to_norm['T'].size))
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True, 'sigma': sc_to_norm['rmsR'], 'method': 'trf', 'jac': '3-point', 'xtol': 1e-15, 'ftol': 1e-8, 'loss': 'linear', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, sc_to_norm['T'], sc_to_norm['R'], **fitargs)
    print('The cov matrix is: {}'.format(pcov))
    perr = np.sqrt(np.diag(pcov))
    print('Ascending (SC -> N): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.left.set_values(result, perr)

    # Attempt to fit the N-->Sc region now
    print('For N to SC fit initial guess is {}, and the number of data points are: {}'.format(x_0, norm_to_sc['T'].size))
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True, 'sigma': norm_to_sc['rmsR'], 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'soft_l1', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    #fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'linear', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, norm_to_sc['T'], norm_to_sc['R'], **fitargs)
    perr = np.sqrt(np.diag(pcov))
    print('Descending (N -> SC): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.right.set_values(result, perr)
    tc = result[2]
    # Make output plot
    ivplt.make_resistance_vs_temperature_plots(output_path, data_channel, fixed_name, fixed_value, norm_to_sc, sc_to_norm, model_func, fit_result)
    return tc


def get_power_temperature_curves(output_path, data_channel, number_of_windows, iv_dictionary, tc=None):
    '''Generate a power vs temperature curve for a TES'''
    # Need to select power in the biased region, i.e. where P(R) ~ constant
    # Try something at 0.5*Rn
    iv_dictionary = find_normal_to_sc_data(iv_dictionary, number_of_windows)
    rN = 200e-3
    deltaR = 30e-3
    temperatures = np.empty(0)
    power = np.empty(0)
    power_rms = np.empty(0)
    for temperature, iv_data in iv_dictionary.items():
        # This is not good with the un-windowed data because of noise fluctuations
        # It is probably necessary to window the data and extract time boundaries for each case
        # But that will be a bit tricky.
        # dbias = np.gradient(iv_data['iBias'].flatten(), edge_order=2)
        # cut1 = np.logical_and(iv_data['iBias'].flatten() > 0, dbias < 0)   # Positive iBias -slope (High to Low, N-->Sc)
        # cut2 = np.logical_and(iv_data['iBias'].flatten() <= 0, dbias > 0)  # Negative iBias +slope (-High to -Low, N-->Sc)
        # cut_norm_to_sc = np.logical_or(cut1, cut2)
        # cut_fixed_norm_to_sc = np.logical_and(fixed_cut, cut_norm_to_sc)
        # cut_fixed_sc_to_norm = np.logical_and(fixed_cut, ~cut_norm_to_sc)
        cut_norm_to_sc = iv_data['cut_norm_to_sc']
        # Cuts get complicated. We will need to make a cut on a cut.
        cut_fixed_norm_to_sc = np.logical_and(iv_data['rTES'][cut_norm_to_sc] > rN - deltaR, iv_data['rTES'][cut_norm_to_sc] < rN + deltaR)
        cut_fixed_norm_to_sc = cut_fixed_norm_to_sc.flatten()
        if cut_fixed_norm_to_sc.sum() > 0:
            temperatures = np.append(temperatures, float(temperature)*1e-3)
            pTES = iv_data['pTES'][cut_norm_to_sc].flatten()
            power = np.append(power, np.mean(pTES[cut_fixed_norm_to_sc]))
            power_rms = np.append(power_rms, np.std(pTES[cut_fixed_norm_to_sc])/np.sqrt(cut_fixed_norm_to_sc.sum()))
        else:
            print('For T = {} mK there were no values used.'.format(temperature))
    # print('The main T vector is: {}'.format(temperatures))
    # print('The iTES vector is: {}'.format(iTES))
    # TODO: Make these input values?
    max_temp = tc or 60e-3
    cut_temperature = np.logical_and(temperatures > 35e-3, temperatures < max_temp)  # This should be the expected Tc
    cut_power = power < 1e-6
    cut_temperature = np.logical_and(cut_temperature, cut_power)
    # [k, n, Ttes, Pp]
    if tc is None:
        print('No Tc was passed, floating Tc')
        lbounds = [1e-9, 0, 1e-3]
        ubounds = [1, 10, 250e-3]
        fixedArgs = {'Pp': 0}
        x0 = [100e-9, 5, 50e-3]
    else:
        print('Tc = {} mK was passed. Fixing to this value'.format(tc))
        lbounds = [1e-9, 0]
        ubounds = [1, 10]
        fixedArgs = {'Pp': 0, 'Ttes': tc}
        x0 = [100e-9, 5]
    # Attempt to fit it to a power function
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
    axes = ivplt.generic_fitplot_with_errors(axes=axes, x=temperatures, y=power, axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = ivplt.add_model_fits(axes=axes, x=temperatures, y=power, model=fit_result, model_function=fitfuncs.tes_power_polynomial, xscale=xscale, yscale=yscale)
    axes = ivplt.pt_fit_textbox(axes=axes, model=fit_result)

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
    return True
