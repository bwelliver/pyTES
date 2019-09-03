
import argparse
import glob
import re
from os.path import splitext


import numpy as np
import pandas as pd
import matplotlib as mp
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import IVPlots as ivp
import squid_info


def complex_tes_one_block(f, *args):
    '''Define using 3 free params'''
    Zinf, Z0, tau = args
    Ztes = Zinf + (Zinf - Z0)*1/(-1 + 1j*2*np.pi*f*tau)
    return Ztes


def ztes_model_g(f, *args):
    '''Simple 1 block model for Ztes from Lindeman
    f = actual frequency data
    Independent parameters shall be as follows:
    I0 = current on TES
    R0 = TES resistance
    T0 = TES temperature
    g = TES thermal conductivity
    a = alpha (T/R*dR/dT)
    b = beta (I/R)*dR/dI
    C = TES heat capacity

    Also to keep in mind
    L = TES loop gain
    t = TES time constant
    Z(w) = Rl + jwL + Ztes(w)
    '''
    I0, R0, T0, alpha, beta, C = args
    g = 554.54820e-9 * 5 * np.power(T0, 4)
    tau = 1/((I0*I0*R0/(C*T0))*alpha - (g/C))
    Ztes = R0*((1+beta) + ((2+beta)/2)*((I0*I0*R0)/(C*T0))*alpha*tau * (-1 + (1+1j*(2*np.pi*f)*tau)/(-1+1j*(2*np.pi*f)*tau)))
    return Ztes

def ztes_model(f, *args):
    '''Simple 1 block model for Ztes from Lindeman
    f = actual frequency data
    Independent parameters shall be as follows:
    I0 = current on TES
    R0 = TES resistance
    T0 = TES temperature
    g = TES thermal conductivity
    a = alpha (T/R*dR/dT)
    b = beta (I/R)*dR/dI
    C = TES heat capacity

    Also to keep in mind
    L = TES loop gain
    t = TES time constant
    Z(w) = Rl + jwL + Ztes(w)
    '''
    I0, R0, T0, g, alpha, beta, C = args
    g = 554.54820e-9 * 5 * np.power(T0, 4)
    tau = 1/((I0*I0*R0/(C*T0))*alpha - (g/C))
    Ztes = R0*((1+beta) + ((2+beta)/2)*((I0*I0*R0)/(C*T0))*alpha*tau * (-1 + (1+1j*(2*np.pi*f)*tau)/(-1+1j*(2*np.pi*f)*tau)))
    return Ztes


def ztes_model_confused(f, *args):
    '''Complex valued version of the TES impedance
    Simple 1 block model for Ztes
    f = actual frequency data
    Independent parameters shall be as follows:
    I0 = current on TES
    R0 = TES resistance
    T0 = TES temperature
    g = TES thermal conductivity
    a = alpha (T0/R0)*dR/dT
    b = beta (I0/R0)*dR/dI
    C = TES heat capacity
    Also to keep in mind
    LG = TES loop gain
    t = TES time constant
    Z(w) = Rl + jwL + Ztes(w)
    NOTE: Be sure to use VOLTAGE biased model!!!!!
    '''
    # print(args)
    I0, R0, T0, g, alpha, beta, C = args
    P0 = (I0 * I0) * R0
    LG = (P0 * alpha) / (g * T0)
    # Note for voltage biased mode it might be 1+LG here
    tau = C / (g * (1 + LG))

    # Note: If we switch the (1-LG) denominator term here then the real part
    # has a decaying shape from low to high freq instead of an increasing shape
    # Note that (LG -1) effectively works in real but swaps the order in imag

    # Ultimate note: The tau drives imaginary space and the LG/(1-LG) is real
    # 1-LG is needed to make imag look ok and 1-LG is needed to make real look ok
    # Each part separately can look ok if the other is swapped.

    # Simply switching to 1+LG divisions and making it -1+2jpi*f*tau intstead of 1+2jpi*f*tau
    # seems to work. For now let's use lindeman model.
    Ztes = R0*(1 + beta) + ((LG)/(1 + LG))*(R0*(2 + beta))/(-1 + 2j*np.pi*f*tau)
    return Ztes


def ztes_model_fixed(fixedArgs):
    '''Model with fixed parameters...use function currying
    Parameters used for the model:
        I0 - The TES operating current [A]
        R0 - The TES operating resistance [Ohm]
        T0 - The TES operating temperature [K]
        g - The thermal conductance [W/K]
        alpha - The TES alpha parameter, (T0/R0)*dR/dT
        beta - The TES beta parameter, (I0/R0)*dR/dI
        C - The heat capacity [J/K]
    '''
    I0 = fixedArgs.get('I0', None)
    R0 = fixedArgs.get('R0', None)
    T0 = fixedArgs.get('T0', None)
    g = fixedArgs.get('g', None)
    alpha = fixedArgs.get('alpha', None)
    beta = fixedArgs.get('beta', None)
    C = fixedArgs.get('C', None)
    fixedArgs = [I0, R0, T0, g, alpha, beta, C]

    def ztes_model_wrapper(f, *args):
        args = list(args)
        newargs = (args.pop(0) if item is None else item for item in fixedArgs)
        ztes = ztes_model(f, *newargs)
        return np.append(ztes.real, ztes.imag)
    return ztes_model_wrapper


def ztes_model_wrapper(f, *args):
    '''Flat version of tes model function'''
    ztes = complex_tes_one_block(f, *args)
    return np.append(ztes.real, ztes.imag)


def ztes_model_2block(f, *args):
    '''The intermediate model 2-block'''
    # Here are all parameters
    # Assumed fixed: I0, R0, T0, g0
    I0, R0, T0, g0, gTES1, gB, alpha, beta, Ctes, C1 = args
    w = 2*np.pi*f
    P0 = I0*I0*R0
    loop = P0*alpha/(g0*T0)
    tauI = Ctes/(g0*(1-loop))
    tau1 = C1/(gTES1 + gB)
    Ztes = R0*(1+beta) + (loop/(1-loop))*R0*(2+beta)/(1 + 1j*w*tauI - (gTES1/(gTES1+gB)*(1-loop))*(1/(1+1j*w*tau1)))
    #Ztes = R0*(1+beta) + (loop/(1+loop))*R0*(2+beta)/(-1 + 1j*w*tauI - (gTES1/(gTES1+gB)*(1+loop))*(1/(1+1j*w*tau1)))
    #if Ctes > C1 or gTES1 > g0 or gB < g0:
    #    Ztes = 1e9*Ztes
    return Ztes


def ztes_model_2block_fixed(fixedArgs):
    '''Model with fixed parameters...use function currying
    Parameters used for the model:
        I0 - The TES operating current [A]
        R0 - The TES operating resistance [Ohm]
        T0 - The TES operating temperature [K]
        g - The thermal conductance [W/K]
        alpha - The TES alpha parameter, (T0/R0)*dR/dT
        beta - The TES beta parameter, (I0/R0)*dR/dI
        C - The heat capacity [J/K]
    '''
    # I0, R0, T0, g0, gTES1, gB, alpha, beta, Ctes, C1
    I0 = fixedArgs.get('I0', None)
    R0 = fixedArgs.get('R0', None)
    T0 = fixedArgs.get('T0', None)
    g0 = fixedArgs.get('g0', None)
    #####
    gTES1 = fixedArgs.get('gTES1', None)
    gB = fixedArgs.get('gB', None)
    alpha = fixedArgs.get('alpha', None)
    beta = fixedArgs.get('beta', None)
    Ctes = fixedArgs.get('Ctes', None)
    C1 = fixedArgs.get('C1', None)
    fixedArgs = [I0, R0, T0, g0, gTES1, gB, alpha, beta, Ctes, C1]

    def ztes_model_2block_wrapper(f, *args):
        args = list(args)
        # print('The length of fixedArgs is: {} and for args it is: {}'.format(len(fixedArgs), len(args)))
        newargs = (args.pop(0) if item is None else item for item in fixedArgs)
        ztes = ztes_model_2block(f, *newargs)
        return np.append(ztes.real, ztes.imag)
    return ztes_model_2block_wrapper


def ratio_model_function(f, *args):
    '''Complex version of ratio fit'''
    # rn, rl, lin = [0.71431989, 24.46217e-3 + 21e-3, *args]
    rn, rl, lin = args
    # rn, rl, lin = args
    zl = 1j * 2 * np.pi * f * lin
    zn = rn + rl + zl
    zsc = rl + zl
    return zn/zsc


def ratio_model_fixed(fixedArgs):
    '''Response ratio model curried function wrapper'''
    Rn = fixedArgs.get('Rn', None)
    Rl = fixedArgs.get('Rl', None)
    Lin = fixedArgs.get('Lin', None)
    fixedArgs = [Rn, Rl, Lin]

    def ratio_model_function_wrapper(f, *args):
        '''Wrapper with flat version of the fit'''
        # Here we parse what we got from fixedArgs and args
        args = list(args)
        newArgs = (args.pop(0) if item is None else item for item in fixedArgs)
        ratio = ratio_model_function(f, *newArgs)
        return np.append(ratio.real, ratio.imag)
    return ratio_model_function_wrapper


def ratio_model_function_wrapper(f, *args):
    '''flat version of ratio fit'''
    ratio = ratio_model_function(f, *args)
    return np.append(ratio.real, ratio.imag)


def natural_sort_key(string, _dre=re.compile(r'(\d+)')):
    '''Defines a natural sorting key for use with sorting file lists'''
    key = [int(text) if text.isdigit() else text.lower() for text in _dre.split(string)]
    return key


def invert_ratio(data, invert=False):
    '''Function that will invert the real and imaginary parts of the complex frequency response'''
    if invert is True:
        data.update((f, 1/resp) for f, resp in data.items())
    return data


def gen_plot_points_fit(xdata, ydata, xfit, yfit, results, perr, labels, y0=None, mode='ratio', **kwargs):
    '''Create generic plots that may be semilogx (default)'''
    xlabel = labels['xlabel']
    ylabel = labels['ylabel']
    title = labels['title']
    figname = labels['figname']

    figsize = kwargs.get('figsize', (12, 12))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.plot(xfit, yfit, 'r-', marker='None', linewidth=2)
    if y0 is not None:
        if np.all(np.isreal(y0)):
            ax.plot(xfit, y0, 'b-', marker='None', linewidth=2, label='Initial guess')
        else:
            ax.plot(y0.real, y0.imag, 'b-', marker='None', linewidth=2, label='Initial guess')
    ax.set_xlabel(xlabel, fontsize=18, horizontalalignment='right', x=1.0)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)

    # Parse other relevant kwargs
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    if kwargs.get('minorticks', None) is not None:
        ax.minorticks_on()
        ax.grid(which='minor')
    ax.tick_params(axis='both', which='major', labelsize=22)
    # Set up text strings for fit based on the mode
    if mode == 'ratio':
        rn, rl, lin = results
        rn_err, rl_err, lin_err = perr
        tRn = r'$R_{n} = %.5f \pm %.5f \mathrm{m\Omega}$' % (rn*1e3, rn_err*1e3)
        tRl = r'$R_{L} = %.5f \pm %.5f \mathrm{m\Omega}$' % (rl*1e3, rl_err*1e3)
        tLin = r'$L_{in} = %.5f \pm %.5f \mathrm{\mu H}$' % (lin*1e6, lin_err*1e6)
        text_string = tRn + '\n' + tRl + '\n' + tLin
    if mode == 'ztes':
        I0, R0, T0, g, alpha, beta, C = results
        I0_err, R0_err, T0_err, g_err, alpha_err, beta_err, C_err = perr
        LG = (I0*I0*R0 * alpha) / (g * T0)
        # tau = C / (g * (1 + LG))
        tau = 1/((I0*I0*R0*alpha)/(C*T0) - (g/C))
        tI = r'$I_{0} = %.5f \pm %.5f \mathrm{\mu A}$' % (I0*1e6, I0_err*1e6)
        ta = r'$\alpha = %.5f \pm %.5f$' % (alpha, alpha_err)
        tb = r'$\beta = %.5f \pm %.5f$' % (beta, beta_err)
        tR = r'$R_{0} = %.5f \pm %.5f \mathrm{m\Omega}$' % (R0*1e3, R0_err*1e3)
        tg = r'$G = %.5f \pm %.5f \mathrm{pW/K}$' % (g*1e12, g_err*1e12)
        tC = r'$C = %.5f \pm %.5f \mathrm{pJ/K}$' % (C*1e12, C_err*1e12)
        tT = r'$T_{0} = %.5f \pm %.5f \mathrm{mK}$' % (T0*1e3, T0_err*1e3)
        tLG = r'$\mathcal{L} = %.5f \pm %.5f$' % (LG, 0)
        tTau = r'$\mathrm{\tau} = %.5f \pm %.5f \mathrm{ms}$' % (tau*1e3, 0)
        text_string = tI + '\n' + tR + '\n' + tg + '\n' + tT + '\n' + ta + '\n' + tb + '\n' + tC + '\n' + tLG + '\n' + tTau
    if mode == 'ztes2':
        I0, R0, T0, g0, gTES1, gB, alpha, beta, Ctes, C1 = results
        I0_err, R0_err, T0_err, g0_err, gTES1_err, gB_err, alpha_err, beta_err, Ctes_err, C1_err = perr
        # tau = C / (g * (1 + LG))
        P0 = I0*I0*R0
        loop = P0*alpha/(g0*T0)
        tauI = Ctes/(g0*(1-loop))
        tau1 = C1/(gTES1 + gB)
        tI = r'$I_{0} = %.5f \pm %.5f \mathrm{\mu A}$' % (I0*1e6, I0_err*1e6)
        ta = r'$\alpha = %.5f \pm %.5f$' % (alpha, alpha_err)
        tb = r'$\beta = %.5f \pm %.5f$' % (beta, beta_err)
        tR = r'$R_{0} = %.5f \pm %.5f \mathrm{m\Omega}$' % (R0*1e3, R0_err*1e3)
        tg0 = r'$G = %.5f \pm %.5f \mathrm{pW/K}$' % (g0*1e12, g0_err*1e12)
        tgTES1 = r'$GTES1 = %.5f \pm %.5f \mathrm{pW/K}$' % (gTES1*1e12, gTES1_err*1e12)
        tgB = r'$Gb = %.5f \pm %.5f \mathrm{pW/K}$' % (gB*1e12, gB_err*1e12)
        tCtes = r'$Ctes = %.5f \pm %.5f \mathrm{pJ/K}$' % (Ctes*1e12, Ctes_err*1e12)
        tC1 = r'$C1 = %.5f \pm %.5f \mathrm{pJ/K}$' % (C1*1e12, C1_err*1e12)
        tT = r'$T_{0} = %.5f \pm %.5f \mathrm{mK}$' % (T0*1e3, T0_err*1e3)
        tLG = r'$\mathcal{L} = %.5f \pm %.5f$' % (loop, 0)
        tTauI = r'$\mathrm{\tau I} = %.5f \pm %.5f \mathrm{ms}$' % (tauI*1e3, 0)
        tTau1 = r'$\mathrm{\tau 1} = %.5f \pm %.5f \mathrm{ms}$' % (tau1*1e3, 0)
        text_string = tI + '\n' + tR + '\n' + tg0 + '\n' + tT + '\n' + tgTES1 + '\n' + tgB + '\n' + ta + '\n' + tb + '\n' + tCtes + '\n' + tC1 + '\n' + tLG + '\n' + tTauI + '\n' + tTau1
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.1, 0.6, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=props)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    set_aspect = kwargs.get('set_aspect', None)
    if set_aspect is not None:
        ax.set_aspect(set_aspect, 'datalim')
    fig.savefig(figname, dpi=150, bbox_inches='tight')
    # plt.show()
    # plt.draw()
    plt.close('all')
    return True


def gen_plot_points(xdata, ydata, labels, **kwargs):
    '''Create generic plots that may be semilogx (default)'''
    xlabel = labels['xlabel']
    ylabel = labels['ylabel']
    title = labels['title']
    figname = labels['figname']

    figsize = kwargs.get('figsize', (12, 12))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0.0, linestyle='None')
    ax.set_xlabel(xlabel, fontsize=18, horizontalalignment='right', x=1.0)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)

    # Parse other relevant kwargs
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    if kwargs.get('minorticks', None) is not None:
        ax.minorticks_on()
        ax.grid(which='minor')
    ax.tick_params(axis='both', which='major', labelsize=22)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(18)
    set_aspect = kwargs.get('set_aspect', None)
    if set_aspect is not None:
        ax.set_aspect(set_aspect, 'datalim')
    fig.savefig(figname, dpi=150, bbox_inches='tight')
    # plt.show()
    # plt.draw()
    plt.close('all')
    return True


def generate_multi_plot(outdir, temperature, biases, ztes):
    '''Generate a plot with all the ztes curves in it'''
    # Overlay multiple IV plots
    fig = plt.figure(figsize=(16, 16))
    axes = fig.add_subplot(111)
    xscale = 1
    yscale = 1
    for bias, z in ztes.items():
        tones = np.fromiter(z.keys(), dtype='float')
        z_array = np.fromiter(z.values(), dtype='c16')
        params = {'marker': 'o', 'markersize': 5, 'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None}
        axes_options = {'xlabel': r'Re TES Impedance [$\Omega$]',
                        'ylabel': r'Im TES Impedance [$\Omega$]',
                        'title': 'Nyquist Plot of TES Impedances at T = {} mK'.format(temperature)
                        }
        axes = ivp.generic_fitplot_with_errors(axes=axes, x=z_array.real, y=z_array.imag, params=params, axes_options=axes_options, xscale=xscale, yscale=yscale)
    # Add a legend?
    axes.legend(['Bias = {} uA'.format(bias) for bias in biases], markerscale=5, fontsize=18)
    axes.set_ylim((-1, 1))
    axes.set_xlim((-1, 1))
    axes.set_aspect('equal', 'datalim')
    file_name = outdir + '/' + 'nyquist_plots_zTES_T{}mK'.format(temperature)
    ivp.save_plot(fig, axes, file_name, dpi=200)
    return True


def generate_model_diagnostic_plots(output_directory, ratio, model_function, results, perr, x0=None, bias=None, mode='ratio'):
    '''Generate diagnostic plots for the model fits'''
    # Split into arrays
    freq = np.fromiter(ratio.keys(), dtype='float')
    ratio = np.fromiter(ratio.values(), dtype='c16')
    model_freq = np.linspace(freq.min(), freq.max(), int(1e5))
    print('In diagnostic plot routine the results are: {}'.format(results))
    model_ratio = model_function(model_freq, *results)
    if x0 is not None:
        model_initial = model_function(model_freq, *x0)
    else:
        model_initial = None
    # Plot the real, imaginary, and magnitude vs frequency
    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re Zn/Zsc',
                  'title': 'Power Spectrum of Model Impedance Ratio Real',
                  'figname': output_directory + '/real_ratio_model_tones.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re Ztes',
                  'title': 'Power Spectrum of Model TES Impedance Real',
                  'figname': output_directory + '/real_ztes_model_{}uA_tones.png'.format(bias)
                  }
    if mode == 'ztes2':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re Ztes',
                  'title': 'Power Spectrum of Model TES 2 Block Impedance Real',
                  'figname': output_directory + '/real_ztes_2block_model_{}uA_tones.png'.format(bias)
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear',
                'minorticks': True
                }
    if mode == 'ztes' or mode == 'ztes2':
        formargs['ylim'] = (-0.5, 1)
    y0 = model_initial.real if x0 is not None else None
    gen_plot_points_fit(freq, ratio.real, model_freq, model_ratio.real, results, perr, labels, y0, mode, **formargs)

    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im Zn/Zsc',
                  'title': 'Power Spectrum of Model Impedance Ratio Imaginary',
                  'figname': output_directory + '/imag_ratio_model_tones.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im Ztes',
                  'title': 'Power Spectrum of Model TES Impedance Imaginary',
                  'figname': output_directory + '/imag_ztes_model_{}uA_tones.png'.format(bias)
                  }
    if mode == 'ztes2':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im Ztes',
                  'title': 'Power Spectrum of Model TES 2 Block Impedance Imaginary',
                  'figname': output_directory + '/imag_ztes_2block_model_{}uA_tones.png'.format(bias)
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear',
                'minorticks': True
                }
    if mode == 'ztes' or mode == 'ztes2':
        formargs['ylim'] = (-0.5, 0.1)
    y0 = model_initial.imag if x0 is not None else None
    gen_plot_points_fit(freq, ratio.imag, model_freq, model_ratio.imag, results, perr, labels, y0, mode, **formargs)

    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs Zn/Zsc',
                  'title': 'Power Spectrum of Model Impedance Ratio Magnitude',
                  'figname': output_directory + '/abs_ratio_model_tones.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs Ztes',
                  'title': 'Power Spectrum of Model TES Impedance Magnitude',
                  'figname': output_directory + '/abs_ztes_model_{}uA_tones.png'.format(bias)
                  }
    if mode == 'ztes2':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs Ztes',
                  'title': 'Power Spectrum of Model TES 2 Block Impedance Magnitude',
                  'figname': output_directory + '/abs_ztes_2block_model_{}uA_tones.png'.format(bias)
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear',
                'minorticks': True
                }
    if mode == 'ztes' or mode == 'ztes2':
        formargs['ylim'] = (0, 1)
    y0 = np.absolute(model_initial) if x0 is not None else None
    gen_plot_points_fit(freq, np.absolute(ratio), model_freq, np.absolute(model_ratio), results, perr, labels, y0, mode, **formargs)

    # Make Nyquist plot (Im vs Re)
    if mode == 'ratio':
        labels = {'xlabel': 'Re Zn/Zsc',
                  'ylabel': 'Im Zn/Zsc',
                  'title': 'Nyquist Plot of Model Impedance Ratio',
                  'figname': output_directory + '/nyquist_ratio_model.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Re Ztes',
                  'ylabel': 'Im Ztes',
                  'title': 'Nyquist Plot of Model TES Impedance',
                  'figname': output_directory + '/nyquist_ztes_model_{}uA.png'.format(bias)
                  }
    if mode == 'ztes2':
        labels = {'xlabel': 'Re Ztes',
                  'ylabel': 'Im Ztes',
                  'title': 'Nyquist Plot of Model TES 2 Block Impedance',
                  'figname': output_directory + '/nyquist_ztes_2block_model_{}uA.png'.format(bias)
                  }
    formargs = {'figsize': (16, 16),
                'xscale': 'linear',
                'yscale': 'linear',
                'set_aspect': 'equal'}
    if mode == 'ztes' or mode == 'ztes2':
        formargs['xlim'] = (-0.5, 1)
        formargs['ylim'] = (-0.7, 0.1)
    gen_plot_points_fit(ratio.real, ratio.imag, model_ratio.real, model_ratio.imag, results, perr, labels, model_initial, mode, **formargs)
    return True


def generate_diagnostic_plots(output_directory, ratio, current=None, mode='ratio'):
    '''Catch all function to generate specific plots'''
    # Split into arrays
    freq = np.fromiter(ratio.keys(), dtype='float')
    ratio = np.fromiter(ratio.values(), dtype='c16')

    # Plot the real, imaginary, and magnitude vs frequency
    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re Zn/Zsc',
                  'title': 'Power Spectrum of Measured Impedance Ratio Real',
                  'figname': output_directory + '/psd_real_znsc_tones.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re G',
                  'title': 'Power Spectrum of Transfer Function Real',
                  'figname': output_directory + '/psd_real_g_tones.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Re Ztes',
                  'title': 'Power Spectrum of Measured TES Impedance Real',
                  'figname': output_directory + '/psd_real_ztes_tones_{}uA.png'.format(current)
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points(freq, np.real(ratio), labels, **formargs)

    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im Zn/Zsc',
                  'title': 'Power Spectrum of Measured Impedance Ratio Imaginary',
                  'figname': output_directory + '/psd_imag_znsc_tones.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im G',
                  'title': 'Power Spectrum of Transfer Function Imaginary',
                  'figname': output_directory + '/psd_imag_g_tones.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Im Ztes',
                  'title': 'Power Spectrum of Measured TES Impedance Imaginary',
                  'figname': output_directory + '/psd_imag_ztes_tones_{}uA.png'.format(current)
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points(freq, np.imag(ratio), labels, **formargs)

    if mode == 'ratio':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs Zn/Zsc',
                  'title': 'Power Spectrum of Measured Impedance Ratio Magnitude',
                  'figname': output_directory + '/psd_abs_znsc_tones.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs G',
                  'title': 'Power Spectrum of Transfer Function Magnitude',
                  'figname': output_directory + '/psd_abs_g_tones.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Frequency [Hz]',
                  'ylabel': 'Abs Ztes',
                  'title': 'Power Spectrum of Measured TES Impedance Magnitude',
                  'figname': output_directory + '/psd_abs_ztes_tones_{}uA.png'.format(current)
                  }
    formargs = {'figsize': (16, 8), 'xscale': 'log', 'yscale': 'linear', 'minorticks': True}
    gen_plot_points(freq, np.absolute(ratio), labels, **formargs)

    # Make Nyquist plot (Im vs Re)
    if mode == 'ratio':
        labels = {'xlabel': 'Re Zn/Zsc',
                  'ylabel': 'Im Zn/Zsc',
                  'title': 'Nyquist Plot of Impedance Ratio',
                  'figname': output_directory + '/nyquist_ratio.png'
                  }
    if mode == 'transfer':
        labels = {'xlabel': 'Re G',
                  'ylabel': 'Im G',
                  'title': 'Nyquist Plot of Transfer Function',
                  'figname': output_directory + '/nyquist_g.png'
                  }
    if mode == 'ztes':
        labels = {'xlabel': 'Re Ztes',
                  'ylabel': 'Im Ztes',
                  'title': 'Nyquist Plot of TES Impedance',
                  'figname': output_directory + '/nyquist_ztes_{}uA.png'.format(current)
                  }
    formargs = {'figsize': (16, 16), 'xscale': 'linear', 'yscale': 'linear', 'set_aspect': 'equal'}
    gen_plot_points(ratio.real, ratio.imag, labels)
    return True


# File IO
def get_frequency_list(filename):
    '''Load a fast digitizer txt and get the sweept frequency values
    The first few lines are header so we will skip them
    But also we can have more than one "end of header" string so search backwards.
    This is longer but won't make copies of the array. Note that data starts 2 lines after end of header is printed
    '''
    print('The filename is: {}'.format(filename))
    file_ext = splitext(filename)[1]
    lines_to_skip = 1 if file_ext == '.txt' else 2
    # print('Skipping {} lines'.format(lines_to_skip))
    with open(filename, 'r') as file:
        lines = file.readlines()
    eoh = 0
    for index, line in reversed(list(enumerate(lines))):
        if line.find('***End_of_Header') > -1:
            eoh = index
            break
    # print('End of header line at: {} and the line is: {}'.format(eoh, lines[eoh]))
    tones = []
    start = eoh + lines_to_skip
    for line in lines[start:]:
        line = line.strip('\n').split('\t')
        tones.append(float(line[0]))
    return np.array(tones)


def get_whitenoise_data_pandas(data_files):
    '''Get white noise data and return the tones and zdata'''
    for idx, filename in enumerate(data_files):
        print('The filename is: {}'.format(filename))
        is_real = filename.find('real') != -1
        data = pd.read_csv(filename, delimiter='\t')
        cols = data.columns.tolist()
        df = data[cols[0]][1] - data[cols[0]][0]
        # First order try get all integer data below 30k?
        if df < 1:
            steps = int(1/df)
        else:
            steps = 1
        cut = data[cols[0]] < 30e3
        print('Step size is: {}'.format(steps))
        tones = data[cols[0]][cut][0::steps].to_numpy()
        rdata = data[cols[1]][cut][0::steps].to_numpy()
        if not is_real:
            rdata = rdata*1j
        if idx == 0:
            zdata = rdata
        else:
            zdata += rdata
    return tones, zdata


def get_data_pandas(filename, freq=None):
    '''Load data into memory via pandas and return what we need, namely the specific
    tone and response value. This is faster than readlines()
    '''
    print('The filename is: {}'.format(filename))
    # file_ext = splitext(filename)[1]
    # lines_to_skip = 1 if file_ext == '.txt' else 2
    data = pd.read_csv(filename, delimiter='\t')
    cols = data.columns.tolist()
    df = data[cols[0]][1] - data[cols[0]][0]
    cut = (data[cols[0]] > freq - df/2) & (data[cols[0]] < freq + df/2)
    tones = data[cols[0]][cut].to_numpy()
    zdata = data[cols[1]][cut].to_numpy()
    return tones, zdata


def get_data(filename, freq=None):
    '''Load a fast digitizer lvm and get signal and response columns
    The first few lines are header so we will skip them
    But also we can have more than one "end of header" string so search backwards.
    This is longer but won't make copies of the array. Note that data starts 2 lines after end of header is printed
    '''
    print('The filename is: {}'.format(filename))
    file_ext = splitext(filename)[1]
    lines_to_skip = 1 if file_ext == '.txt' else 2
    # print('Skipping {} lines'.format(lines_to_skip))
    with open(filename, 'r') as file:
        lines = file.readlines()
    eoh = 0
    for index, line in reversed(list(enumerate(lines))):
        if line.find('***End_of_Header') > -1:
            eoh = index
            break
    # print('End of header line at: {} and the line is: {}'.format(eoh, lines[eoh]))
    tones = []
    zdata = []
    start = eoh + lines_to_skip
    # Determine df
    df = []
    for line in lines[start:start+2]:
        line = line.strip('\n').split('\t')
        df.append(float(line[0]))
    df = df[1] - df[0]
    if freq is None:
        for line in lines[start:]:
            line = line.strip('\n').split('\t')
            tones.append(float(line[0]))
            zdata.append(float(line[1]))
    else:
        for line in lines[start:]:
            line = line.strip('\n').split('\t')
            lfreq = float(line[0])
            if freq - df/2 < lfreq < freq + df/2:
                tones.append(lfreq)
                zdata.append(float(line[1]))
                break
    return np.asarray(tones), np.asarray(zdata)


def parse_lvm_file(infile, intype='response', freq=None):
    '''Function to parse a LVM file containing complex Z data
    There are two bits of information needed: the frequency and the data
    Frequency can be obtained from file name via the pattern '*_frequencyHz*'
    '''

    if intype == 'response':
        freq, zdata = get_data_pandas(infile, freq)
        return freq, zdata
    if intype == 'frequency_list':
        freq = get_frequency_list(infile)
        return freq
    return None


def get_list_of_files(input_directory, subdir, run, temperature, current):
    '''Get the list of files
    In general useful information is encoded in the file names and also user specified
    input_directory/*run{}/*T{}mK*/*{}uA*.txt''
    '''
    globstring = '{}/*run{}/*T{}mK*/{}/*_{}uA*.txt'.format(input_directory, run, temperature, subdir, current)
    print('The glob is: {}'.format(globstring))
    list_of_files = glob.glob('{}/*run{}/*T{}mK*/{}/*_{}uA*.txt'.format(input_directory, run, temperature, subdir, current))
    list_of_files.sort(key=natural_sort_key)
    return list_of_files


def split_files_by_type(list_of_files):
    '''Split master list of files into frequency files and data files'''
    frequency_files = []
    new_list_of_files = []
    for file in list_of_files:
        isfrequency = file.find('tones') > -1 or file.find('freq') > -1
        if isfrequency is True:
            frequency_files.append(file)
        else:
            new_list_of_files.append(file)
    return frequency_files, new_list_of_files


def get_tones(frequency_files):
    '''Get the tones to be used in this sweep from the corresponding tone file
    '''
    # Step 1: Find the right tone file for this current
    tone_file = frequency_files[0]
    # Step 2: Load up the file
    tones = parse_lvm_file(tone_file, intype="frequency_list")
    return tones


def parse_file(file_name, tones):
    '''Parse data from the specified file'''
    # We have a file that has some data in it...what data is it and where does it go?
    is_real = file_name.find('real') != -1
    frequency_index = int(file_name.strip('.txt').split('_')[-1])
    if frequency_index >= tones.size:
        print('Frequency index of {} is found but frequency list is only {} entries long. Skipping...'.format(frequency_index, len(tones)))
        return None
    # Load the file...we know what frequency we are looking for so use that to our advantage
    desired_tone = tones[frequency_index]
    frequencies, response = parse_lvm_file(file_name, freq=desired_tone)
    # It is possible the file contains multiple records for a given frequency. If so average them together
    tone = frequencies[0]
    response = np.mean(response)
    # Now we have a single point (f, r). If it is imaginary multiply the response by 1j.
    if not is_real:
        response *= 1j
    return tone, response


def get_whitenoise_response(data_files):
    '''Get the tones and response from white noise only'''
    tones, responses = get_whitenoise_data_pandas(data_files)
    # Convert response to a dictionary with keys of frequency
    response = dict(zip(tones, responses))
    return tones, response


def get_response(data_files, tones):
    '''Get the response data to be used in this sweep from corresponding files'''
    # Here we need to return set of ratios. The general format is freq -> ratio
    # Each file is full of data we do not need.
    # We will store these in a dictionary such that d[f] = response
    data = {}
    for file in data_files:
        tone, response = parse_file(file, tones)
        if tone in data.keys():
            data[tone] += response
        else:
            data[tone] = response
    # Now we have a data dictionary with keys of frequencies and values are complex response
    return data


def get_tones_and_response(input_directory, subdir, run, temperature, current, invert=True, whiteNoise=False):
    '''Function that will return a dictionary of tones and corresponding ratios
    '''
    # The value of current tells what files go grab so in principle we do NOT need to pass it
    # to other functions since the list we have to work with should only contain the correct currents
    if whiteNoise is False:
        list_of_files = get_list_of_files(input_directory, subdir, run, temperature, current)
        frequency_files, data_files = split_files_by_type(list_of_files)
        tones = get_tones(frequency_files)
        response = get_response(data_files, tones)
    if whiteNoise is True:
        list_of_files = get_list_of_files(input_directory, subdir, run, temperature, current)
        print('The list of files for current {} is: {}'.format(current, list_of_files))
        tones, response = get_whitenoise_response(list_of_files)
    response = invert_ratio(response, invert=invert)
    return tones, response


def get_ratio(input_directory, subdir, run, temperature, sc, normal, whiteNoise=False):
    '''Process steps to return a dictionary comprised of keys that are the tones and values
    that are the complex response at that particular tone
    '''
    # Step 1: Get the SC and normal tones and ratios
    sc_tones, sc_response = get_tones_and_response(input_directory, subdir, run, temperature, current=sc, whiteNoise=whiteNoise)
    n_tones, n_response = get_tones_and_response(input_directory, subdir, run, temperature, current=normal, whiteNoise=whiteNoise)
    if np.any(sc_tones != n_tones):
        print('Warning: SC tones and Normal tones do not agree!')
        raise Exception('SC and Normal tone lists are not the same!')
    if np.any(sc_tones != list(sc_response.keys())):
        print('Warning! Response dictionary keys not the same as tone list!')
        raise Exception('Warning! Response dictionary keys not the same as tone list!')
    if sc_response.keys() != n_response.keys():
        raise Exception('Warning! SC response keys are different from Normal response keys!')
    # Step 2: Compute the ratio of the normal response to the sc response
    ratio = {f: n_response[f]/sc_response[f] for f in sc_response}
    return ratio, n_response


def fit_tes_model(ztes, model_func, fixedArgs, **kwargs):
    '''Function to perform fitting of the complex TES impedance values to an electrothermal model'''
    tones = np.fromiter(ztes.keys(), dtype=float)
    ztes = np.fromiter(ztes.values(), dtype=np.complex128)
    flat_ztes = np.append(ztes.real, ztes.imag)
    result, pcov = curve_fit(model_func(fixedArgs), tones, flat_ztes, **kwargs)
    perr = np.sqrt(np.diag(pcov))
    return result, perr


def fit_ratio_model(ratio, model_func, fixedArgs, **kwargs):
    '''Function to perform fitting of the frequency response ratio to a model'''

    tones = np.fromiter(ratio.keys(), dtype=float)
    ratio = np.fromiter(ratio.values(), dtype=np.complex128)
    flatratio = np.append(ratio.real, ratio.imag)
    result, pcov = curve_fit(model_func(fixedArgs), tones, flatratio, **kwargs)
    perr = np.sqrt(np.diag(pcov))
    return result, perr


def get_transfer_function(squid, n_response, results):
    '''Using the results and the normal mode response compute the empirical SQUID transfer function
    G(f)
    '''
    squid_parameters = squid_info.SQUIDParameters(squid)
    Rfb = squid_parameters.Rfb
    Zbias = squid_parameters.Rbias
    M = squid_parameters.M
    Rsh = squid_parameters.Rsh
    dc_factor = (Rsh * Rfb * M)/Zbias
    Rn, Rl, L = results
    # zcirc_normal = Rn + Rl + (2 * 1j * np.pi) * L * np.fromiter(n_response.keys(), dtype=float)
    # Keeping in mind n_response is a dictionary with keys = tones and values = complex response
    # we compute g to be a similar thingsa
    # zcirc_normal(f) = Rn + Rl + 2j*pi*f*L
    g = {}
    for tone, response in n_response.items():
        g[tone] = (response * dc_factor) / (Rn + Rl + (2j * np.pi * L * tone))
    return g


def compute_transfer_function(input_directory, output_directory, subdir, run, squid, normalR, loadR, temperature, sc, normal, whiteNoise=False):
    '''Function to handle computation and diagnostic plots related
    to the transfer function.
    Outputs:
        G: the transfer function
        Rn: the TES normal resistance
        Rl: the Thevenin equivalent load resistance (R_shunt + R_para)
        L: the TES input line inductance
    These outputs are fit results
    '''
    # Step 1: Get the ratio dictionary
    ratio, n_response = get_ratio(input_directory, subdir, run, temperature, sc, normal, whiteNoise=whiteNoise)
    # Step 2: Diagnostic plots of the ratio?
    print('Generating diagnostic plots in {}'.format(output_directory))
    generate_diagnostic_plots(output_directory, ratio, mode='ratio')
    # Step 3: Fit to a model [rn, rl, lin]
    fixedArgs = {'Rn': normalR, 'Rl': loadR}
    print('Attempting to fit ratio model')
    fitargs = {'p0': [10e-3, 0.1e-8], 'method': 'lm'}
    # fitargs = {'p0': (10e-3, 0.1e-8), 'bounds': ((0, 0), (np.inf, np.inf)), 'method': 'trf'}
    results, perr = fit_ratio_model(ratio, ratio_model_fixed, fixedArgs, **fitargs)
    # Join results and fixed values into set order: Rn, Rl, Lin
    fixedResults = [fixedArgs.get('Rn'), fixedArgs.get('Rl'), fixedArgs.get('Lin')]
    results, perr = list(results), list(perr)
    results = [results.pop(0) if item is None else item for item in fixedResults]
    perr = [perr.pop(0) if item is None else 0 for item in fixedResults]
    print('The fit results are: Rn = {} mOhm, Rl = {} mOhm, Lin = {} uH'.format(results[0]*1e3, results[1]*1e3, results[2]*1e6))
    # Step 4: Diagnostic plots of the model
    generate_model_diagnostic_plots(output_directory, ratio, ratio_model_function, results, perr)
    # Step 5: Now we can create G(w)
    g = get_transfer_function(squid, n_response, results)
    generate_diagnostic_plots(output_directory, g, mode='transfer')
    Rn, Rl, L = results
    return (g, Rn, Rl, L)


def get_zcirc(squid, bias_response, G):
    '''Compute the circuit impedance, Zcirc.
    Z_circ(w) = Z_meas(w)/G(w)
    Also Z_tes = Z_circ - Rl - iwL
    '''

    # SQUID Parameters
    squid_parameters = squid_info.SQUIDParameters(squid)
    Rfb = squid_parameters.Rfb
    Zbias = squid_parameters.Rbias
    M = squid_parameters.M
    Rsh = squid_parameters.Rsh
    # Zfb = Rfb + 2*np.pi*1j*tones*Lfb
    Zfb = Rfb
    # Zcirc = Zmeas/G and Ztes = Zcirc - Rl - 2pi*i*f*L
    # Zmeas = (MRfRsh)/(Zbias) * response
    sqfactor = (M*Zfb*Rsh)/(Zbias)
    zcirc = {}
    for tone, response in bias_response.items():
        zcirc[tone] = sqfactor*response/G[tone]
    return zcirc


def get_ztes(squid, bias_response, G, Rl, Lin):
    '''Given input data return the TES Z value'''
    # SQUID Parameters
    squid_parameters = squid_info.SQUIDParameters(squid)
    Rfb = squid_parameters.Rfb
    Zbias = squid_parameters.Rbias
    M = squid_parameters.M
    Rsh = squid_parameters.Rsh
    # Zfb = Rfb + 2*np.pi*1j*tones*Lfb
    Zfb = Rfb
    # Zcirc = Zmeas/G and Ztes = Zcirc - Rl - 2pi*i*f*L
    # Zmeas = (MRfRsh)/(Zbias) * response
    sqfactor = (M*Zfb*Rsh)/(Zbias)
    ztes = {}
    for tone, response in bias_response.items():
        ztes[tone] = sqfactor*response/G[tone] - Rl - 2j * np.pi * tone * Lin
    return ztes


def compute_z(input_directory, output_directory, subdir, run, squid, temperature, bias, G, Rn, Rl, Lin, fitModel=False, whiteNoise=False):
    '''Function to compute the complex impedance given information about the transfer function'''
    # Step 1: Get the tones and response for the particular bias current
    tones, response = get_tones_and_response(input_directory, subdir, run, temperature, current=bias, whiteNoise=whiteNoise)
    # Step 2: Compute the complex TES impedance
    ztes = get_ztes(squid, response, G, Rl, Lin)
    # For fun get zcirc
    # zcirc = get_zcirc(squid, response, G)
    # Step 3: Diagnostic plots of the TES impedance
    generate_diagnostic_plots(output_directory, ztes, current=bias, mode='ztes')
    # Step 4: Fit the TES impedance to an electrothermal model
    # Order of values: (Ib, Rb, T, g, a, b, C)
    # Some of these probably need to be constrained, such as Ib, Rb and possibly T or g
    if fitModel is True:
        print('Attemping to fit the TES thermal model')
        # fitargs = {'p0': (24e-12, 10, 1, 20e-12), 'method': 'lm'}
        # (a, b, C)
        init_dict = {'10.0': {'I0': 1.849e-6, 'R0': 0.0663},
                     '12.0': {'I0': 1.256-6, 'R0': 0.1507},
                     '13.0': {'I0': 1.12e-6, 'R0': 0.1936},
                     '14.0': {'I0': 1.001e-6, 'R0': 0.2381},
                     '15.0': {'I0': 0.930e-6, 'R0': 0.2856},
                     '18.0': {'I0': 0.7553e-6, 'R0': 0.4406},
                     '20.0': {'I0': 0.704e-6, 'R0': 0.5402}
                     }
        T0 = 55e-3
        g0 = 23.2e-12
        p0 = [700, 0.9, 0.1e-12]
        lbounds = (100, 0.8, 1e-14)
        ubounds = (np.inf, 2, np.inf)
        fixedArgs = {'I0': init_dict[str(bias)]['I0'], 'R0': init_dict[str(bias)]['R0'], 'T0': T0, 'g': g0}  # Ib = 13.0 uA
        # fixedArgs = {'I0': 0.946e-6, 'R0': 0.2892, 'T0': 55e-3, 'g': 25.372e-12}  # Ib = 15
        fitargs = {'p0': p0, 'bounds': (lbounds, ubounds), 'method': 'trf'}
        results, perr = fit_tes_model(ztes, ztes_model_fixed, fixedArgs, **fitargs)
        #g = 554.54820e-9 * 5 * np.power(results[0], 4)
        #results = [results[0], g, results[1], results[2], results[3]]
        #perr = [perr[0], 0, perr[1], perr[2], perr[3]]

        # Join results and fixed values into set order: Rn, Rl, Lin
        fixedResults = [fixedArgs.get('I0'), fixedArgs.get('R0'), fixedArgs.get('T0'),
                        fixedArgs.get('g'), fixedArgs.get('alpha'), fixedArgs.get('beta'),
                        fixedArgs.get('C')]
        results, perr = list(results), list(perr)
        results = [results.pop(0) if item is None else item for item in fixedResults]
        perr = [perr.pop(0) if item is None else 0 for item in fixedResults]
        print('The results of the TES fit are as follows:')
        print('I0 = {} uA, R0 = {} mOhm, T0 = {} mK, g = {} pW/K'.format(results[0]*1e6, results[1]*1e3, results[2]*1e3, results[3]*1e12))
        print('alpha = {}, beta = {}, C = {} pJ'.format(results[4], results[5], results[6]*1e12))
        # Step 5: Diagnostic plots of the model
        fixedX0 = [fixedArgs.get('I0'), fixedArgs.get('R0'), fixedArgs.get('T0'),
                   fixedArgs.get('g'), fixedArgs.get('alpha'), fixedArgs.get('beta'),
                   fixedArgs.get('C')]
        #g0 = 554.54820e-9 * 5 * np.power(p0[0], 4)
        #p0 = [p0[0], g0, p0[1], p0[2], p0[3]]
        x0 = [p0.pop(0) if item is None else item for item in fixedX0]
        print('The actual initial values are: {}'.format(x0))
        generate_model_diagnostic_plots(output_directory, ztes, ztes_model, results, perr, x0, bias, mode='ztes')
        # Try to fit the 2 block model
        # I0, R0, T0, g0,| gTES1, gB, alpha, beta, Ctes, C1
# =============================================================================
#         T0 = 55e-3
#         g0 = 23.2e-12
#         fixedArgs = {'I0': 1.12e-6, 'R0': 0.1936, 'T0': T0, 'g0': g0}
#         # gTES1, gB, alpha, beta, Ctes, C1
#         p0 = [1.5e-12, 100e-12, 600, 1, 0.1e-13, 1e-11]
#         lbounds = (1e-14, 1e-14, 10, 0.2, 1e-14, 1e-14)
#         ubounds = (24e-12, 1e-10, 1e4, 2, 3e-12, 1e-6)
#         fitargs = {'p0': p0, 'bounds': (lbounds, ubounds), 'method': 'trf'}
#         results, perr = fit_tes_model(ztes, ztes_model_2block_fixed, fixedArgs, **fitargs)
#         fixedResults = [fixedArgs.get('I0'), fixedArgs.get('R0'), fixedArgs.get('T0'), fixedArgs.get('g0'),
#                         fixedArgs.get('gTES1'), fixedArgs.get('gB'),
#                         fixedArgs.get('alpha'), fixedArgs.get('beta'),
#                         fixedArgs.get('Ctes'), fixedArgs.get('C1')]
#         results, perr = list(results), list(perr)
#         results = [results.pop(0) if item is None else item for item in fixedResults]
#         perr = [perr.pop(0) if item is None else 0 for item in fixedResults]
#         print('The results of the TES fit are as follows:')
#         print('I0 = {} uA, R0 = {} mOhm, T0 = {} mK, g = {} pW/K'.format(results[0]*1e6, results[1]*1e3, results[2]*1e3, results[3]*1e12))
#         print('gTES1 = {} pW/K, gB = {} pW/K, alpha = {}, beta = {}, Ctes = {} pJ/K, C1 = {} pJ/K '.format(results[4]*1e12, results[5]*1e12, results[6], results[7], results[8]*1e12, results[9]*1e12))
#         # Step 5: Diagnostic plots of the model
#         fixedX0 = [fixedArgs.get('I0'), fixedArgs.get('R0'), fixedArgs.get('T0'),
#                    fixedArgs.get('g0'), fixedArgs.get('gTES1'), fixedArgs.get('gB'),
#                    fixedArgs.get('alpha'), fixedArgs.get('beta'),
#                    fixedArgs.get('Ctes'), fixedArgs.get('C1')]
#         x0 = [p0.pop(0) if item is None else item for item in fixedX0]
#         print('The actual initial values are: {}'.format(x0))
#         generate_model_diagnostic_plots(output_directory, ztes, ztes_model_2block, results, perr, x0, mode='ztes2')
# =============================================================================
    return ztes


def process_complex_impedance(indir, outdir, subdir, only_transfer, run, squid, normalR, loadR, temperature, sc, normal, biases, fitModel=False, whiteNoise=False):
    '''Main function that implements complex impedance computations'''

    # Step 1: Generate the transfer function if we don't have it already
    # IMPORTANT: Since the ratio function is basically (Rn + Rl + 2jpifL)/(Rl + 2jpifL)
    # an infinite number of solutions exist of the form a*(Rn + Rl + 2jpifL) / a*(Rl + 2jpifL)
    # (i.e., a*Rn, a*Rl, a*Lin). We *MUST* provide at least 1 of these as fixed values to get an
    # appropriate scaling factor.
    G, Rn, Rl, Lin = compute_transfer_function(indir, outdir, subdir, run, squid, normalR, loadR, temperature, sc, normal, whiteNoise=whiteNoise)

    # Step 2: Compute the complex impedance
    ztes = {}
    if not only_transfer:
        for bias in biases:
            ztes[bias] = compute_z(indir, outdir, subdir, run, squid, temperature, bias, G, Rn, Rl, Lin, fitModel=fitModel, whiteNoise=whiteNoise)
        # Make plot with all ztes curves
        generate_multi_plot(outdir, temperature, biases, ztes)
    return ztes


def get_args():
    '''Get input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDirectory',
                        help='Specify the full path of the directory that contains all the files you wish to use')
    parser.add_argument('-o', '--outputDirectory',
                        help='Specify the full path of the output directory to put plots and root files.\
                        If it is not a full path, a plots and root subdirectory will be added in the input directory')
    parser.add_argument('-d', '--subDirectory',
                        help='Specify a subdirectory name inside the inputDirectory/run$/T$mK/ root directory to get files from')
    parser.add_argument('-f', '--fitModel', action='store_true', help='Indicates whether to perform the fit of the TES models or not')
    parser.add_argument('-g', '--onlyTransfer', action='store_true',
                        help='Only compute the transfer function step. Stops computation prior to impedance step')
    parser.add_argument('-T', '--temperature',
                        help='Specify the temperature in mK')
    parser.add_argument('-r', '--run', type=int,
                        help='Specify the run number')
    parser.add_argument('-R', '--normalResistance', type=float, help='Specify the TES normal resistance in Ohms')
    parser.add_argument('-L', '--loadResistance', type=float,
                        help='Specify the TES load resistance in Ohms. This is the sum of series parasitic resistance and the shunt resistor')
    parser.add_argument('-s', '--sc', default=0.0, type=float,
                        help='Specify the superconducting mode bias current in uA. Default is 0')
    parser.add_argument('-S', '--squid', help='Specify the SQUID number to use for the computation of DC terms')
    parser.add_argument('-n', '--normal', type=float,
                        help='Specify the normal mode bias current in uA')
    parser.add_argument('-b', '--bias', nargs='+', type=float,
                        help='Specify the bias mode bias current in uA')
    parser.add_argument('-w', '--whiteNoise', action='store_true', help='Read in white noise file instead of sweep')
    args = parser.parse_args()
    plotDir = '{}/run{}/T{}mK/{}'.format(args.inputDirectory, args.run, args.temperature, args.subDirectory)
    args.outputDirectory = args.outputDirectory if args.outputDirectory else plotDir
    return args


if __name__ == '__main__':
    ARGS = get_args()
    print('The bias argument is: {}'.format(ARGS))
    ztes = process_complex_impedance(ARGS.inputDirectory, ARGS.outputDirectory, ARGS.subDirectory, ARGS.onlyTransfer,
                                     ARGS.run, ARGS.squid, ARGS.normalResistance, ARGS.loadResistance,
                                     ARGS.temperature, ARGS.sc, ARGS.normal, ARGS.bias, ARGS.fitModel, ARGS.whiteNoise)
