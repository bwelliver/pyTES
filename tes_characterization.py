'''Module to handle processing aggregate TES data'''

import numpy as np
from scipy.optimize import curve_fit

from iv_processor import iv_windower
import iv_results
import iv_plots as ivplt
import tes_fit_functions as fitfuncs


def get_resistance_temperature_curves_new(output_path, data_channel, number_of_windows, iv_dictionary):
    '''Generate resistance vs temperature curves for a TES'''

    # First window the IV data as need be
    # iv_curves = iv_windower(iv_dictionary, number_of_windows, mode='tes')
    # Rtes = R(i,T) so we are really asking for R(i=constant, T).

    fixed_name = 'iTES'
    fixed_value = 0.1e-6
    delta_value = 0.05e-6
    r_normal = 0.550

    norm_to_sc = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    sc_to_norm = {'T': np.empty(0), 'R': np.empty(0), 'rmsR': np.empty(0)}
    for temperature, iv_data in iv_dictionary.items():
        fixed_cut = np.logical_and(iv_data[fixed_name].flatten() > fixed_value - delta_value, iv_data[fixed_name].flatten() < fixed_value + delta_value)
        quality_cut = iv_data['rTES'].flatten() < 5*r_normal
        fixed_cut = np.logical_and(fixed_cut, quality_cut)
        # This is not good with the un-windowed data because of noise fluctuations
        # It is probably necessary to window the data and extract time boundaries for each case
        # But that will be a bit tricky.
        dbias = np.gradient(iv_data['iBias'].flatten(), edge_order=2)
        cut1 = np.logical_and(iv_data['iBias'].flatten() > 0, dbias < 0)   # Positive iBias -slope (High to Low, N-->Sc)
        cut2 = np.logical_and(iv_data['iBias'].flatten() <= 0, dbias > 0)  # Negative iBias +slope (-High to -Low, N-->Sc)
        cut_norm_to_sc = np.logical_or(cut1, cut2)
        cut_fixed_norm_to_sc = np.logical_and(fixed_cut, cut_norm_to_sc)
        cut_fixed_sc_to_norm = np.logical_and(fixed_cut, ~cut_norm_to_sc)
        if cut_fixed_norm_to_sc.sum() > 0:
            norm_to_sc['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
            norm_to_sc['R'] = np.append(norm_to_sc['R'], np.mean(iv_data['rTES'].flatten()[cut_fixed_norm_to_sc]))
            norm_to_sc['rmsR'] = np.append(norm_to_sc['rmsR'], np.std(iv_data['rTES'].flatten()[cut_fixed_norm_to_sc])/np.sqrt(cut_fixed_norm_to_sc.sum()))
        if cut_fixed_sc_to_norm.sum() > 0:
            sc_to_norm['T'] = np.append(norm_to_sc['T'], float(temperature)*1e-3)
            sc_to_norm['R'] = np.append(norm_to_sc['R'], np.mean(iv_data['rTES'].flatten()[cut_fixed_sc_to_norm]))
            sc_to_norm['rmsR'] = np.append(norm_to_sc['rmsR'], np.std(iv_data['rTES'].flatten()[cut_fixed_sc_to_norm])/np.sqrt(cut_fixed_sc_to_norm.sum()))
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
    fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'absolute_sigma': True, 'sigma': norm_to_sc['rmsR'], 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'soft_l1', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    #fitargs = {'p0': x_0, 'bounds': (lbounds, ubounds), 'method': 'trf', 'jac': '3-point', 'xtol': 1e-14, 'ftol': 1e-14, 'loss': 'linear', 'tr_solver': 'exact', 'x_scale': 'jac', 'max_nfev': 10000, 'verbose': 2}
    result, pcov = curve_fit(model_func, norm_to_sc['T'], norm_to_sc['R'], **fitargs)
    perr = np.sqrt(np.diag(pcov))
    print('Descending (N -> SC): Rn = {} mOhm, r_p = {} mOhm, Tc = {} mK, Tw = {} mK'.format(*[i*1e3 for i in result]))
    fit_result.right.set_values(result, perr)
    # Make output plot
    ivplt.make_resistance_vs_temperature_plots(output_path, data_channel, fixed_name, fixed_value, norm_to_sc, sc_to_norm, model_func, fit_result)
    return True