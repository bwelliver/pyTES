import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numpy import square as pow2
from numpy import power
from numpy import sqrt as sqrt
from scipy.optimize import curve_fit

import tes_fit_functions as fitfuncs
import iv_processor as ivp
import iv_results

import ROOT as rt

matplotlib.use('Agg')


# Container for IV related plot functions
def axis_option_parser(axes, options):
    '''A function to parse some common plot options for limits and scales and log/linear'''
    axes.set_xscale(options.get('logx', 'linear'))
    axes.set_yscale(options.get('logy', 'linear'))
    axes.set_xlim(options.get('xlim', None))
    axes.set_ylim(options.get('ylim', None))
    axes.tick_params(labelsize=26)
    axes.xaxis.set_tick_params(labelsize=26)
    axes.yaxis.set_tick_params(labelsize=26)
    axes.set_xlabel(options.get('xlabel', None), fontsize=26, horizontalalignment='right', x=1.0)
    axes.set_ylabel(options.get('ylabel', None), fontsize=26)
    axes.set_title(options.get('title', None), fontsize=26)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(26)
    return axes


def test_plot(x, y, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    axes.plot(x, y, marker='o', markersize=2, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
    # axes.set_xscale(log)
    axes.set_xlabel(xlab)
    axes.set_ylabel(ylab)
    # axes.set_title(title)
    axes.grid()
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    # plt.draw()
    # plt.show()
    return None


def test_steps(x, y, v, t0, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    axes.plot(x, y, marker='o', markersize=1, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
    # Next add horizontal lines for each step it thinks it found
    for item in v:
        axes.plot([item[0]-t0, item[1]-t0], [item[2], item[2]], marker='.', linestyle='-', color='r')
    # axes.set_xscale(log)
    axes.set_xlabel(xlab)
    axes.set_ylabel(ylab)
    # axes.set_title(title)
    axes.grid()
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    # plt.draw()
    # plt.show()
    return None


def generic_fitplot_with_errors(axes, x, y, params, axes_options, xscale=1, yscale=1):
    '''A function that puts data on a specified axis with error bars'''
    axes.errorbar(x*xscale, y*yscale, elinewidth=3, capsize=2, **params)
    # Parse options
    axes = axis_option_parser(axes, axes_options)
    # axes.yaxis.label.set_size(18)
    # axes.xaxis.label.set_size(18)
    axes.grid(True)
    # for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    #    label.set_fontsize(18)
    return axes


def fancy_fitplot_with_errors(axes, x, y, params, axes_options, xscale=1, yscale=1):
    '''A function that puts data on a specified axis with error bars'''
    axes.errorbar(x*xscale, y*yscale, elinewidth=3, capsize=2, **params)
    # Parse options
    axes = axis_option_parser(axes, axes_options)
    axes.yaxis.label.set_size(18)
    axes.xaxis.label.set_size(18)
    axes.grid(True)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    return axes


def add_model_fits(axes, x, y, model, model_function, xscale=1, yscale=1):
    '''Add model fits to plots'''
    xModel = np.linspace(x.min(), x.max(), 10000)
    if model.left.result is not None:
        yFit = model_function(xModel, *model.left.result)
        axes.plot(xModel*xscale, yFit*yscale, 'r-', marker='None', linewidth=5)
    if model.right.result is not None:
        yFit = model_function(xModel, *model.right.result)
        axes.plot(xModel*xscale, yFit*yscale, 'g-', marker='None', linewidth=5)
    if model.sc.result is not None:
        yFit = model_function(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        axes.plot(x[cut]*xscale, yFit[cut]*yscale, 'b-', marker='None', linewidth=5)
    return axes


def iv_fit_textbox(axes, R, model):
    '''Add decoration textbox to a plot'''

    lR = r'$\mathrm{Left R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (R.left.value*1e3, R.left.rms*1e3)
    lOff = r'$\mathrm{Left V_{off}} = %.5f \pm %.5f \mathrm{mV}$' % (model.left.result[1]*1e3, model.left.error[1]*1e3)

    sR = r'$\mathrm{SC R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (R.parasitic.value*1e3, R.parasitic.rms*1e3)
    sOff = r'$\mathrm{SC V_{off}} = %.5f \pm %.5f \mathrm{mV}$' % (model.sc.result[1]*1e3, model.sc.error[1]*1e3)

    rR = r'$\mathrm{Right R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (R.right.value*1e3, R.right.rms*1e3)
    rOff = r'$\mathrm{Right V_{off}} = %.5f \pm %.5f \mathrm{mV}$' % (model.right.result[1]*1e3, model.right.error[1]*1e3)

    textStr = lR + '\n' + lOff + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    # anchored_text = AnchoredText(textstr, loc=4)
    # axes.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    axes.text(0.65, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    return axes


def pr_fit_textbox(axes, model):
    '''Add dectoration textbox for a power vs resistance fit'''
    lR = r'$\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (1/model.left.result[0]*1e3, model.left.error[0]/pow2(model.left.result[0])*1e3)
    lI = r'$\mathrm{I_{para}} = %.5f \pm %.5f \mathrm{uA}$' % (model.left.result[1]*1e6, model.left.error[1]*1e6)
    lP = r'$\mathrm{P_{para}} = %.5f \pm %.5f \mathrm{fW}$' % (model.left.result[2]*1e15, model.left.error[2]*1e15)

    textStr = lR + '\n' + lI + '\n' + lP
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    axes.text(0.65, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return axes


def rt_fit_textbox(axes, model):
    '''Add dectoration textbox for a power vs resistance fit'''

    # First is the ascending (SC to N) parameters
    textStr = ''
    if model.left.result is not None:
        lR = r'SC $\rightarrow$ N: $\mathrm{R_{n}} = %.2f \pm %.2f \mathrm{m \Omega}$' % (model.left.result[0]*1e3, model.left.error[0]*1e3)
        lRp = r'SC $\rightarrow$ N: $\mathrm{R_{p}} = %.2f \pm %.2f \mathrm{m \Omega}$' % (model.left.result[1]*1e3, model.left.error[1]*1e3)
        lTc = r'SC $\rightarrow$ N: $\mathrm{T_{c}} = %.2f \pm %.2f \mathrm{mK}$' % (model.left.result[2]*1e3, model.left.error[2]*1e3)
        lTw = r'SC $\rightarrow$ N: $\mathrm{\Delta T_{c}} = %.2f \pm %.2f \mathrm{mK}$' % (model.left.result[3]*1e3, model.left.error[3]*1e3)
        textStr += lR + '\n' + lRp + '\n' + lTc + '\n' + lTw
    # Next the descending (N to SC) parameters...these are the main physical ones
    if model.right.result is not None:
        rR = r'N $\rightarrow$ SC: $\mathrm{R_{n}} = %.2f \pm %.2f \mathrm{m \Omega}$' % (model.right.result[0]*1e3, model.right.error[0]*1e3)
        rRp = r'N $\rightarrow$ SC: $\mathrm{R_{p}} = %.2f \pm %.2f \mathrm{m \Omega}$' % (model.right.result[1]*1e3, model.right.error[1]*1e3)
        rTc = r'N $\rightarrow$ SC: $\mathrm{T_{c}} = %.2f \pm %.2f \mathrm{mK}$' % (model.right.result[2]*1e3, model.right.error[2]*1e3)
        rTw = r'N $\rightarrow$ SC: $\mathrm{\Delta T_{c}} = %.2f \pm %.2f \mathrm{mK}$' % (model.right.result[3]*1e3, model.right.error[3]*1e3)
        if textStr != '':
            textStr += '\n'
        textStr += rR + '\n' + rRp + '\n' + rTc + '\n' + rTw
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    axes.text(0.05, 0.5, textStr, transform=axes.transAxes, fontsize=26, verticalalignment='top', bbox=props)
    return axes


def pt_fit_textbox(axes, model):
    '''Add decoration textbox for a power vs temperature fit'''
    k = model.left.result[0]
    dk = model.left.error[0]
    n = model.left.result[1]
    dn = model.left.error[1]
    Ttes = model.left.result[2]
    dTtes = model.left.error[2]
    # Pp = model.left.result[3]
    # dPp = model.left.error[3]
    # Pn = model.left.result[4]
    # dPn = model.left.error[4]
    lk = r'$k = %.2f \pm %.2f \mathrm{ nW/K^{%.2f}}$' % (k*1e9, dk*1e9, n)
    ln = r'$n = %.2f \pm %.2f$' % (n, dn)
    lTt = r'$T_{TES} = %.2f \pm %.2f \mathrm{ mK}$' % (Ttes*1e3, dTtes*1e3)
    # lPp = r'$P_{0} = %.2f \pm %.2f \mathrm{ fW}$' % (Pp*1e15, dPp*1e15)
    # lPn = r'$P_{N} = %.5f \pm %.5f \mathrm{ fW}$' % (Pn*1e15, dPn*1e15)
    # Compute G at T = Ttes
    # G = dP/dT
    G = n*k*power(Ttes, n-1)
    dG_k = n*power(Ttes, n-1)*dk
    dG_T = n*(n-1)*k*power(1e-4, n-2)  # RMS on T not Ttes
    dG_n = dn*(k*power(Ttes, n-1)*(n*np.log(Ttes) + 1))
    dG = sqrt(pow2(dG_k) + pow2(dG_T) + pow2(dG_n))
    lG = r'$G(T_{TES}) = %.2f \pm %.2f \mathrm{ pW/K}$' % (G*1e12, dG*1e12)
    textStr = lk + '\n' + ln + '\n' + lTt + '\n' + lG
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.7)
    axes.text(0.05, 0.3, textStr, transform=axes.transAxes, fontsize=26, verticalalignment='top', bbox=props)
    return axes


def save_plot(fig, axes, fName, dpi=150):
    '''Save a specified plot'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    # for label in (axes.get_xticklabels() + axes.get_yticklabels()):
    #    label.set_fontsize(18)
    fig.savefig(fName, dpi=dpi, bbox_inches='tight')
    plt.close('all')
    return None


def iv_fitplot(data, model, R, Rp, fName, axes_options, xscale=1, yscale=1):
    '''Wrapper for plotting an iv curve with fit parameters'''
    x, y, xerr, yerr = data
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    # yFit1 = lin_sq(x, *model.right.result)
    axes.errorbar(x*xscale, y*yscale, marker='o', markersize=2, markeredgecolor='black',
                  markerfacecolor='black', markeredgewidth=0, linestyle='None', xerr=xerr*xscale, yerr=yerr*yscale)
    if model.left.result is not None:
        yFit = fitfuncs.lin_sq(x, *model.left.result)
        axes.plot(x*xscale, yFit*yscale, 'r-', marker='None', linewidth=2)
    if model.right.result is not None:
        yFit = fitfuncs.lin_sq(x, *model.right.result)
        axes.plot(x*xscale, yFit*yscale, 'g-', marker='None', linewidth=2)
    if model.sc.result is not None:
        # Need to plot only a subset of data
        yFit = fitfuncs.lin_sq(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        axes.plot(x[cut]*xscale, yFit[cut]*yscale, 'b-', marker='None', linewidth=2)
    axes = axis_option_parser(axes, axes_options)
    axes.grid()
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    # Now generate text strings
    # model values are [results, perr] --> [[m, b], [merr, berr]]
    # R = convert_fit_to_resistance(model, fit_type='iv', Rp=Rp.value, Rp_rms=Rp.rms)
    lR = r'$\mathrm{Left \ R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (R.left.value*1e3, R.left.rms*1e3)
    lOff = r'$\mathrm{Left \ V_{off}} = %.5f \pm %.5f \mathrm{mV}$' % (model.left.result[1]*1e3, model.left.error[1]*1e3)

    sR = r'$\mathrm{R_{sc} - R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (R.sc.value*1e3, R.sc.rms*1e3)
    sOff = r'$\mathrm{V_{sc,off}} = %.5f \pm %.5f \mathrm{mV}$' % (model.sc.result[1]*1e3, model.sc.error[1]*1e3)

    rR = r'$\mathrm{Right \ R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (R.right.value*1e3, R.right.rms*1e3)
    rOff = r'$\mathrm{Right \ V_{off}} = %.5f \pm %.5f \mathrm{mV}$' % (model.right.result[1]*1e3, model.right.error[1]*1e3)

    pR = r'$\mathrm{R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$' % (Rp.value*1e3, Rp.rms*1e3)

    textStr = lR + '\n' + lOff + '\n' + pR + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    # anchored_text = AnchoredText(textstr, loc=4)
    # axes.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    axes.text(0.65, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return None


def make_root_plot(output_path, data_channel, temperature, iv_data, model, Rp, xscale=1, yscale=1):
    '''A helper function to generate a TMultiGraph for the IV curve
    Recipe for this type of plot:
    Create a TCanvas object and adjust its parameters
    Create a TMultiGraph
    Create a TGraph - adjust its parameters and plant it in the TMultiGraph (which now owns the TGraph)
    Finish styling for the TMultiGraph and then save the canvas as .png and .C

    There will be 4 TGraphs
        vOut vs iBias
        Left Normal fit
        Right Normal fit
        SC fit
    '''

    # Create TCanvas
    w = 1600
    h = 1200
    c = rt.TCanvas("iv", "iv", w, h)
    c.SetWindowSize(w + (w - c.GetWw()), h + (h - c.GetWh()))
    c.cd()
    c.SetGrid()
    mg = rt.TMultiGraph()

    # Now let us generate some TGraphs!
    x = iv_data['iBias']
    xrms = iv_data['iBias_rms']
    y = iv_data['vOut']
    yrms = iv_data['vOut_rms']
    # First up: vOut vs iBias
    g0 = rt.TGraphErrors(x.size, x*xscale, y*yscale, xrms*xscale, yrms*yscale)
    g0.SetMarkerSize(0.5)
    g0.SetLineWidth(1)
    g0.SetName("vOut_iBias")
    g0.SetTitle("OutputVoltage vs Bias Current")

    mg.Add(g0)

    # Next up let's add the fit lines
    if model.left.result is not None:
        yFit = fitfuncs.lin_sq(x, *model.left.result)
        gLeft = rt.TGraph(x.size, x*xscale, yFit*yscale)
        gLeft.SetMarkerSize(0)
        gLeft.SetLineWidth(2)
        gLeft.SetLineColor(rt.kRed)
        gLeft.SetTitle("Left Normal Branch Fit")
        gLeft.SetName("left_fit")
        mg.Add(gLeft)
    if model.right.result is not None:
        yFit = fitfuncs.lin_sq(x, *model.right.result)
        gRight = rt.TGraph(x.size, x*xscale, yFit*yscale)
        gRight.SetMarkerSize(0)
        gRight.SetLineWidth(2)
        gRight.SetLineColor(rt.kGreen)
        gRight.SetTitle("Right Normal Branch Fit")
        gRight.SetName("right_fit")
        mg.Add(gRight)
    if model.sc.result is not None:
        yFit = fitfuncs.lin_sq(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        gSC = rt.TGraph(x[cut].size, x[cut]*xscale, yFit[cut]*yscale)
        gSC.SetMarkerSize(0)
        gSC.SetLineWidth(2)
        gSC.SetLineColor(rt.kBlue)
        gSC.SetTitle("Superconducting Branch Fit")
        gSC.SetName("sc_fit")
        mg.Add(gSC)
    # Now I guess we format the multigraph
    xLabel = 'Bias Current [uA]'
    yLabel = 'Output Voltage [mV]'
    titleStr = 'Channel {} Output Voltage vs Bias Current for T = {} mK'.format(data_channel, temperature)
    fName = output_path + '/root/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
    mg.Draw("APL")
    mg.GetXaxis().SetTitle(xLabel)
    mg.GetYaxis().SetTitle(yLabel)
    mg.SetTitle(titleStr)
    # Construct Legend
    leg = rt.TLegend(0.6, 0.7, 0.9, 0.9)
    leg.AddEntry(g0, "IV Data", "le")
    leg.AddEntry(gLeft, "Left Normal Branch Fit", "l")
    leg.AddEntry(gRight, "Right Normal Branch Fit", "l")
    leg.AddEntry(gSC, "Superconducting Branch Fit", "l")
    leg.SetTextSize(0.02)
    # leg.SetTextFont(2)
#    TLegend *leg = new TLegend(0.55, 0.7, 0.9, 0.9);
#    //TLegendEntry *le = leg->AddEntry(h2, "All PhaseIDs: noise distribution across all channels", "fl");
#    leg->AddEntry(h2, "All PhaseIDs: noise distribution across all channels", "fl");
#    //TLegendEntry *le_optimal = leg->AddEntry(hc_optimal, Form("PhaseID %d: noise distribution across all channels", optimal_phaseid), "fl");
#    leg->AddEntry(hc_optimal, Form("PhaseID %d: noise distribution across all channels", optimal_phaseid), "fl");
#    leg->SetTextSize(0.02);
#    leg->SetTextFont(2);
#    leg->Draw();
    leg.Draw()
    # Add some annotations?
    tN = rt.TLatex()
    tN.SetNDC()
    tN.SetTextSize(0.025)
    tN.SetTextAlign(12)
    tN.SetTextAngle(343)
    tN.DrawLatex(0.14, 0.66, "Normal Branch")

    tB = rt.TLatex()
    tB.SetNDC()
    tB.SetTextSize(0.025)
    tB.SetTextAlign(12)
    tB.SetTextAngle(0)
    tB.DrawLatex(0.55, 0.41, "Biased Region")

    tS = rt.TLatex()
    tS.SetNDC()
    tS.SetTextSize(0.025)
    tS.SetTextAlign(12)
    tS.SetTextAngle(282)
    tS.DrawLatex(0.49, 0.77, "SC Region")

#    t = new TLatex();
#    t->SetNDC();
#    t->SetTextFont(62);
#    t->SetTextColor(36);
#    t->SetTextSize(0.08);
#    t->SetTextAlign(12);
#    t->DrawLatex(0.6,0.85,"p - p");
#
#    t->SetTextSize(0.05);
#    t->DrawLatex(0.6,0.79,"Direct #gamma");
#    t->DrawLatex(0.6,0.75,"#theta = 90^{o}");
#
#    t->DrawLatex(0.70,0.55,"H(z)");
#    t->DrawLatex(0.68,0.50,"(barn)");
#
#    t->SetTextSize(0.045);
#    t->SetTextColor(46);
#    t->DrawLatex(0.20,0.30,"#sqrt{s}, GeV");
#    t->DrawLatex(0.22,0.26,"63");
#    t->DrawLatex(0.22,0.22,"200");
#    t->DrawLatex(0.22,0.18,"500");
#
#    t->SetTextSize(0.05);
#    t->SetTextColor(1);
#    t->DrawLatex(0.88,0.06,"z");
    # c.BuildLegend()
    c.Update()
    # Now save the damned thing
    c.SaveAs(fName + '.png')
    fName = output_path + '/root/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + str(int(float(temperature)*1e3)) + 'uK'
    c.SaveAs(fName + '.C')
    c.Close()
    del mg
    del c
    return None


def plot_current_vs_voltage(output_path, data_channel, squid, temperature, data):
    '''Plot the current vs voltage for a TES'''
    # Convert fit parameters to R values
    fit_parameters = data['tes_fit_parameters']
    resistance = ivp.convert_fit_to_resistance(fit_parameters, squid, fit_type='tes')
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

    axes = generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['iTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=fit_parameters, model_function=fitfuncs.lin_sq, xscale=xscale, yscale=yscale)
    axes = iv_fit_textbox(axes=axes, R=resistance, model=fit_parameters)

    file_name = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, axes, file_name)
    return True


def plot_resistance_vs_current(output_path, data_channel, temperature, data):
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
    axes = generic_fitplot_with_errors(axes=axes, x=data['iTES'], y=data['rTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'rTES_vs_iTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, axes, file_name)
    return True


def plot_resistance_vs_voltage(output_path, data_channel, temperature, data):
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
    axes = generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['rTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, axes, file_name)
    return True


def plot_resistance_vs_bias_current(output_path, data_channel, temperature, data):
    '''Plot the resistance vs bias current for a TES'''
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    ylim = (0, 1*yscale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
              'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['iBias_rms']*xscale, 'yerr': data['rTES_rms']*yscale}
    axes_options = {'xlabel': 'Bias Current [uA]',
                    'ylabel': 'TES Resistance [mOhm]',
                    'title': 'Channel {} TES Resistance vs Bias Current for temperatures = {} mK'.format(data_channel, temperature),
                    'ylim': ylim
                    }
    axes = generic_fitplot_with_errors(axes=axes, x=data['iBias'], y=data['rTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, axes, file_name)
    return True


def plot_power_vs_resistance(output_path, data_channel, temperature, data):
    '''Plot the resistance vs bias current for a TES'''
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1e12
    xlim = (0, 1*xscale)
    params = {'marker': 'o', 'markersize': 2, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
              'markeredgewidth': 0, 'linestyle': 'None', 'xerr': data['rTES_rms']*xscale, 'yerr': data['pTES_rms']*yscale}
    axes_options = {'xlabel': 'TES Resistance [mOhm]',
                    'ylabel': 'TES Power [pW]',
                    'title': 'Channel {} TES Power vs TES Resistance for temperatures = {} mK'.format(data_channel, temperature),
                    'xlim': xlim
                    }
    axes = generic_fitplot_with_errors(axes=axes, x=data['rTES'], y=data['pTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    file_name = output_path + '/' + 'pTES_vs_rTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, axes, file_name)
    return True


def plot_power_vs_voltage(output_path, data_channel, temperature, data):
    '''Plot the TES power vs TES voltage'''
    # Note this ideally is a parabola
    cut = np.logical_and(data['rTES'] > 450e-3, data['rTES'] < 2*500e-3)
    if np.sum(cut) < 3:
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
    axes = generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['pTES'], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = add_model_fits(axes=axes, x=data['vTES'], y=data['pTES'], model=fit_result, model_function=fitfuncs.quad_sq, xscale=xscale, yscale=yscale)
    axes = pr_fit_textbox(axes=axes, model=fit_result)

    file_name = output_path + '/' + 'pTES_vs_vTES_ch_' + str(data_channel) + '_' + temperature + 'mK'
    save_plot(fig, axes, file_name)
    return True


def make_tes_multiplot(output_path, data_channel, squid, iv_dictionary):
    '''Make a plot of all temperatures at once
    rTES vs iBias

    '''
    # Current vs Voltage
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e6
    yscale = 1e3
    idx = 0
    tmax = 57
    # Make R vs i plot
    for temperature, data in iv_dictionary.items():
        if idx % 4 != 0 or float(temperature) > tmax:
            idx += 1
            continue
        resistance = ivp.convert_fit_to_resistance(data['tes_fit_parameters'], squid, fit_type='tes')
        params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'black', 'markerfacecolor': 'black',
                  'markeredgewidth': 0, 'linestyle': 'None', 'xerr': None, 'yerr': None
                  }
        axes_options = {'xlabel': 'Bias Current [uA]',
                        'ylabel': r'TES Resistance [m \Omega]',
                        'title': 'Channel {} TES Resistance vs Bias Current'.format(data_channel)
                        }
        axes = generic_fitplot_with_errors(axes=axes, x=data['iBias'], y=data['rTES'], params=params, axes_options=axes_options, xscale=xscale, yscale=yscale)
        axes = add_model_fits(axes=axes, x=data['vTES'], y=data['iTES'], model=data['tes_fit_parameters'], model_function=fitfuncs.lin_sq, xscale=xscale, yscale=yscale)
        axes = iv_fit_textbox(axes=axes, R=resistance, model=data['tes_fit_parameters'])
        idx += 1
    axes.set_ylim((0*yscale, 1*yscale))
    axes.set_xlim((-20, 20))
    file_name = output_path + '/' + 'rTES_vs_iBias_ch_' + str(data_channel)
    save_plot(fig, axes, file_name)

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
                            'title': None  # 'Channel {} TES Current vs Voltage'.format(data_channel)
                            }
            axes = generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['iTES'], params=params,
                                               axes_options=axes_options, xscale=xscale, yscale=yscale)
        idx += 1
    # Add legend?
    axes.legend(['T = {} mK'.format(temperature) for temperature in temperature_names], markerscale=5, fontsize=24)
    axes.set_ylim((0, 2))
    axes.set_xlim((-0.5, 2))
    file_name = output_path + '/' + 'iTES_vs_vTES_ch_' + str(data_channel)
    save_plot(fig, axes, file_name, dpi=200)

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
                            'title': None  # 'Channel {} TES Resistance vs Voltage'.format(data_channel)
                            }
            axes = generic_fitplot_with_errors(axes=axes, x=data['vTES'], y=data['rTES'], params=params,
                                               axes_options=axes_options, xscale=xscale, yscale=yscale)
        idx += 1
    # Add legend?
    axes.legend(['T = {} mK'.format(temperature) for temperature in temperature_names], markerscale=5, fontsize=24)
    axes.set_ylim((0, 720))
    axes.set_xlim((0, 2))
    file_name = output_path + '/' + 'rTES_vs_vTES_ch_' + str(data_channel)
    save_plot(fig, axes, file_name, dpi=200)
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
    axes = generic_fitplot_with_errors(axes=axes, x=sc_to_norm['T'][sort_key], y=sc_to_norm['R'][sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)

    sort_key = np.argsort(norm_to_sc['T'])
    params = {'marker': 'o', 'markersize': 4, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': norm_to_sc['rmsR'][sort_key]*yscale}
    axes = generic_fitplot_with_errors(axes=axes, x=norm_to_sc['T'][sort_key], y=norm_to_sc['R'][sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    axes = add_model_fits(axes=axes, x=norm_to_sc['T'], y=norm_to_sc['R'], model=fit_result, model_function=model_func, xscale=xscale, yscale=yscale)
    axes = rt_fit_textbox(axes=axes, model=fit_result)
    axes.legend(['SC to N', 'N to SC'], markerscale=6, fontsize=26)
    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + fixed_name + '_' + str(np.round(fixed_value*1e6, 3)) + 'uA'
    #for label in axes.get_xticklabels() + axes.get_yticklabels():
    #    label.set_fontsize(26)
    save_plot(fig, axes, file_name)

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
    axes = generic_fitplot_with_errors(axes=axes, x=norm_to_sc['T'][sort_key], y=norm_to_sc['R'][sort_key], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # Let us pad the T values so they are smoooooooth
    model_temperatures = np.linspace(norm_to_sc['T'].min(), 70e-3, 100000)
    axes = add_model_fits(axes=axes, x=model_temperatures, y=norm_to_sc['R'], model=normal_to_sc_fit_result, model_function=model_func, xscale=xscale, yscale=yscale)
    axes = rt_fit_textbox(axes=axes, model=normal_to_sc_fit_result)
    # axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'rTES_vs_T_ch_' + str(data_channel) + '_fixed_' + fixed_name + '_' + str(np.round(fixed_value*1e6, 3)) + 'uA_normal_to_sc_only'
    axes.set_xlim((10, 70))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    save_plot(fig, axes, file_name)
    ######## ALPHA PLOTS
    # TODO: MAKE THIS WORK
    # We can also make plots of alpha = (T0/R0)*dR/dT
    # use model values
    #T = np.linspace(norm_to_sc['T'].min(), norm_to_sc['T'].max(), 1000)
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
    cutR = R > 10e-3
    print('The largest alpha = {} at T = {} mK'.format(np.nanmax(alpha[cutR]), T[np.nanargmax(alpha[cutR])]))
    # Use a model function as well!
    model_T = np.linspace(norm_to_sc['T'].min(), norm_to_sc['T'].max(), 100000)
    model_R = model_func(model_T, *normal_to_sc_fit_result.right.result)
    #model_dR_dT = fitfuncs.dtanh_tc(model_T, *normal_to_sc_fit_result.right.result)
    model_dR_dT = np.gradient(model_R, model_T, edge_order=2)
    model_alpha = (model_T/model_R) * model_dR_dT
    #model_alpha = (model_T/model_R)*np.gradient(model_R, model_T, edge_order=2)
    model_cutR = model_R > 10e-3
    # Remove it
    #alpha[np.nanargmax(alpha)] = 0
    #print('The largest alpha is now = {} at T = {} mK'.format(np.nanmax(alpha), T[np.nanargmax(alpha)]))
    # make plot
    ### alpha vs R
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1
    yscale = 1
    axes_options = {'xlabel': r'Resistance [m$\Omega$]',
                    'ylabel': 'TES ' + r'$\alpha$',
                    'logy': 'linear',
                    'xlim': (0, 0.700*xscale),
                    'ylim': (0, model_alpha[model_cutR].max()),
                    'title': 'Channel {}'.format(data_channel) + ' TES ' + r'$\alpha$' + ' vs Resistance for TES Current = {}'.format(np.round(fixed_value*1e6, 3)) + r'$\mu$' + 'A'
                    }
    params = {'marker': 'o', 'markersize': 5, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': None}
    axes = generic_fitplot_with_errors(axes=axes, x=R[cutR], y=alpha[cutR], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # axes2 = axes.twinx()
    params = {'marker': 'None', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': '-',
              'xerr': None, 'yerr': None}
    generic_fitplot_with_errors(axes=axes, x=model_R[model_cutR], y=model_alpha[model_cutR], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # Let us pad the T values so they are smoooooooth
    # model_temperatures = np.linspace(norm_to_sc['T'].min(), 70e-3, 100000)
    #axes = ivp.add_model_fits(axes=axes, x=model_temperatures, y=norm_to_sc['R'], model=normal_to_sc_fit_result, model_function=model_func, xscale=xscale, yscale=yscale)
    #axes = ivp.rt_fit_textbox(axes=axes, model=normal_to_sc_fit_result)
    # axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'alpha_vs_R_ch_' + str(data_channel) + '_fixed_' + fixed_name + '_' + str(np.round(fixed_value*1e6, 3)) + 'uA_normal_to_sc_only'
    #axes.set_xlim((10, 70))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    save_plot(fig, axes, file_name)

    ### alpha vs T
    fig = plt.figure(figsize=(16, 12))
    axes = fig.add_subplot(111)
    xscale = 1e3
    yscale = 1
    axes_options = {'xlabel': 'Temperature [mK]',
                    'ylabel': 'TES ' + r'$\alpha$',
                    'logy': 'linear',
                    'xlim': (0, model_T[-1]*xscale),
                    'ylim': (0, model_alpha[model_cutR].max()),
                    'title': 'Channel {}'.format(data_channel) + ' TES ' + r'$\alpha$' + ' vs Temperature for TES Current = {}'.format(np.round(fixed_value*1e6, 3)) + r'$\mu$' + 'A'
                    }
    params = {'marker': 'o', 'markersize': 5, 'markeredgecolor': 'green',
              'markerfacecolor': 'green', 'markeredgewidth': 0, 'linestyle': 'None',
              'xerr': None, 'yerr': None}
    axes = generic_fitplot_with_errors(axes=axes, x=T[cutR], y=alpha[cutR], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # axes2 = axes.twinx()
    params = {'marker': 'None', 'markersize': 4, 'markeredgecolor': 'red',
              'markerfacecolor': 'red', 'markeredgewidth': 0, 'linestyle': '-',
              'xerr': None, 'yerr': None}
    generic_fitplot_with_errors(axes=axes, x=model_T[model_cutR], y=model_alpha[model_cutR], axes_options=axes_options, params=params, xscale=xscale, yscale=yscale)
    # Let us pad the T values so they are smoooooooth
    # model_temperatures = np.linspace(norm_to_sc['T'].min(), 70e-3, 100000)
    #axes = ivp.add_model_fits(axes=axes, x=model_temperatures, y=norm_to_sc['R'], model=normal_to_sc_fit_result, model_function=model_func, xscale=xscale, yscale=yscale)
    #axes = ivp.rt_fit_textbox(axes=axes, model=normal_to_sc_fit_result)
    # axes.legend(['SC to N', 'N to SC'])
    file_name = output_path + '/' + 'alpha_vs_T_ch_' + str(data_channel) + '_fixed_' + fixed_name + '_' + str(np.round(fixed_value*1e6, 3)) + 'uA_normal_to_sc_only'
    #axes.set_xlim((10, 70))
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(18)
    save_plot(fig, axes, file_name)
    return True