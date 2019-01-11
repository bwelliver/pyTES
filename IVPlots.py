import matplotlib as mp
from matplotlib import pyplot as plt
import numpy as np
from numpy import square as pow2
from numpy import power
from numpy import sqrt as sqrt
from numpy import sum as nsum
from pyTESFitFunctions import lin_sq
import ROOT as rt

# Container for IV related plot functions
def axis_option_parser(axes, options):
    '''A function to parse some common plot options for limits and scales and log/linear'''
    axes.set_xscale(options.get('logx', 'linear'))
    axes.set_yscale(options.get('logy', 'linear'))
    axes.set_xlim(options.get('xlim', None))
    axes.set_ylim(options.get('ylim', None))
    axes.set_xlabel(options.get('xlabel', None), fontsize=18, horizontalalignment='right', x=1.0)
    axes.set_ylabel(options.get('ylabel', None), fontsize=18)
    axes.set_title(options.get('title', None), fontsize=18)
    return axes


def test_plot(x, y, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    axes.plot(x, y, marker='o', markersize=2, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
    #axes.set_xscale(log)
    axes.set_xlabel(xlab)
    axes.set_ylabel(ylab)
    #axes.set_title(title)
    axes.grid()
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    #plt.draw()
    #plt.show()
    return None


def test_steps(x, y, v, t0, xlab, ylab, fName):
    """Create generic plots that may be semilogx (default)"""
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111)
    axes.plot(x, y, marker='o', markersize=1, markeredgecolor='black', markeredgewidth=0.0, linestyle='None')
    # Next add horizontal lines for each step it thinks it found
    for item in v:
        axes.plot([item[0]-t0,item[1]-t0], [item[2], item[2]], marker='.', linestyle='-', color='r')
    #axes.set_xscale(log)
    axes.set_xlabel(xlab)
    axes.set_ylabel(ylab)
    #axes.set_title(title)
    axes.grid()
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    #plt.draw()
    #plt.show()
    return None


def generic_fitplot_with_errors(axes, x, y, params, axes_options, xScale=1, yScale=1):
    '''A function that puts data on a specified axis with error bars'''
    out = axes.errorbar(x*xScale, y*yScale, elinewidth=3, capsize=2, **params)
    # Parse options
    axes = axis_option_parser(axes, axes_options)
    axes.yaxis.label.set_size(18)
    axes.xaxis.label.set_size(18)
    axes.grid(True)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    return axes


def fancy_fitplot_with_errors(axes, x, y, params, axes_options, xScale=1, yScale=1):
    '''A function that puts data on a specified axis with error bars'''
    out = axes.errorbar(x*xScale, y*yScale, elinewidth=3, capsize=2, **params)
    # Parse options
    axes = axis_option_parser(axes, axes_options)
    axes.yaxis.label.set_size(18)
    axes.xaxis.label.set_size(18)
    axes.grid(True)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    return axes


def add_model_fits(axes, x, y, model, model_function, xScale=1, yScale=1):
    '''Add model fits to plots'''
    xModel = np.linspace(x.min(), x.max(), 10000)
    if model.left.result is not None:
        yFit = model_function(xModel, *model.left.result)
        axes.plot(xModel*xScale, yFit*yScale, 'r-', marker='None', linewidth=4)
    if model.right.result is not None:
        yFit = model_function(xModel, *model.right.result)
        axes.plot(xModel*xScale, yFit*yScale, 'g-', marker='None', linewidth=2)
    if model.sc.result is not None:
        yFit = model_function(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        axes.plot(x[cut]*xScale, yFit[cut]*yScale, 'b-', marker='None', linewidth=2)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    return axes


def iv_fit_textbox(axes, R, model):
    '''old: add_fit_textbox'''
    '''Add decoration textbox to a plot'''

    lR = r'$\mathrm{Left R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.left.value*1e3, R.left.rms*1e3)
    lOff = r'$\mathrm{Left V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)

    sR = r'$\mathrm{SC R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.parasitic.value*1e3, R.parasitic.rms*1e3)
    sOff = r'$\mathrm{SC V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.sc.result[1]*1e3, model.sc.error[1]*1e3)

    rR = r'$\mathrm{Right R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.right.value*1e3, R.right.rms*1e3)
    rOff = r'$\mathrm{Right V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)

    textStr = lR + '\n' + lOff + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #axes.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    out = axes.text(0.65, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    return axes


def pr_fit_textbox(axes, model):
    '''old: add_power_voltage_textbox'''
    '''Add dectoration textbox for a power vs resistance fit'''
    lR = r'$\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(1/model.left.result[0]*1e3, model.left.error[0]/pow2(model.left.result[0])*1e3)
    lI = r'$\mathrm{I_{para}} = %.5f \pm %.5f \mathrm{uA}$'%(model.left.result[1]*1e6, model.left.error[1]*1e6)
    lP = r'$\mathrm{P_{para}} = %.5f \pm %.5f \mathrm{fW}$'%(model.left.result[2]*1e15, model.left.error[2]*1e15)

    textStr = lR + '\n' + lI + '\n' + lP
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    axes.text(0.65, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return axes


def rt_fit_textbox(axes, model):
    '''old: add_resistance_temperature_textbox'''
    '''Add dectoration textbox for a power vs resistance fit'''

    # First is the ascending (SC to N) parameters
    textStr = ''
    if model.left.result is not None:
        lR = r'SC to N: $\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.left.result[0]*1e3, model.left.error[0]*1e3)
        lRp = r'SC to N: $\mathrm{R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)
        lTc = r'SC to N: $\mathrm{T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.left.result[2]*1e3, model.left.error[2]*1e3)
        lTw = r'SC to N: $\mathrm{\Delta T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.left.result[3]*1e3, model.left.error[3]*1e3)
        textStr += lR + '\n' + lRp + '\n' + lTc + '\n' + lTw
    # Next the descending (N to SC) parameters...these are the main physical ones
    if model.right.result is not None:
        rR = r'N to SC: $\mathrm{R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.right.result[0]*1e3, model.right.error[0]*1e3)
        rRp = r'N to SC: $\mathrm{R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)
        rTc = r'N to SC: $\mathrm{T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.right.result[2]*1e3, model.right.error[2]*1e3)
        rTw = r'N to SC: $\mathrm{\Delta T_{c}} = %.5f \pm %.5f \mathrm{mK}$'%(model.right.result[3]*1e3, model.right.error[3]*1e3)
        if textStr is not '':
            textStr += '\n'
        textStr += rR + '\n' + rRp + '\n' + rTc + '\n' + rTw
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    axes.text(0.10, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return axes


def pt_fit_textbox(axes, model):
    '''old: add_power_temperature_textbox'''
    '''Add decoration textbox for a power vs temperature fit'''
    k = model.left.result[0]
    dk = model.left.error[0]
    n = model.left.result[1]
    dn = model.left.error[1]
    Ttes = model.left.result[2]
    dTtes = model.left.error[2]
    lk = r'$k = %.5f \pm %.5f \mathrm{nW/K^{%.5f}}$'%(k*1e9, dk*1e9, n)
    ln = r'$n = %.5f \pm %.5f$'%(n, dn)
    lTt = r'$T_{TES} = %.5f \pm %.5f \mathrm{mK}$'%(Ttes*1e3, dTtes*1e3)
    # Compute G at T = Ttes
    # G = dP/dT
    G = n*k*power(Ttes, n-1)
    dG_k = n*power(Ttes, n-1)*dk
    dG_T = n*(n-1)*k*power(1e-4, n-2) # RMS on T not Ttes
    dG_n = dn*(k*power(Ttes, n-1)*(n*np.log(Ttes) + 1))
    dG = sqrt(pow2(dG_k) + pow2(dG_T) + pow2(dG_n))
    lG = r'$G(T_{TES}) = %.5f \pm %.5f \mathrm{pW/K}$'%(G*1e12, dG*1e12)
    textStr = lk + '\n' + ln + '\n' + lTt + '\n' + lG
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    axes.text(0.65, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    return axes


def save_plot(fig, axes, fName, dpi=150):
    '''Save a specified plot'''
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    fig.savefig(fName, dpi=dpi, bbox_inches='tight')
    plt.close('all')
    return None


def iv_fitplot(data, model, R, Rp, fName, axes_options, xScale=1, yScale=1):
    '''Wrapper for plotting an iv curve with fit parameters'''
    x,y,xerr,yerr = data
    fName = fName + '.png' if fName.split('.png') != 2 else fName
    fig = plt.figure(figsize=(16,12))
    axes = fig.add_subplot(111)
    yFit1 = lin_sq(x, *model.right.result)
    axes.errorbar(x*xScale, y*yScale, marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', markeredgewidth=0, linestyle='None', xerr=xerr*xScale, yerr=yerr*yScale)
    if model.left.result is not None:
        yFit = lin_sq(x, *model.left.result)
        axes.plot(x*xScale, yFit*yScale, 'r-', marker='None', linewidth=2)
    if model.right.result is not None:
        yFit = lin_sq(x, *model.right.result)
        axes.plot(x*xScale, yFit*yScale, 'g-', marker='None', linewidth=2)
    if model.sc.result is not None:
        # Need to plot only a subset of data
        yFit = lin_sq(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        axes.plot(x[cut]*xScale, yFit[cut]*yScale, 'b-', marker='None', linewidth=2)
    axes = axis_option_parser(axes, axes_options)
    axes.grid()
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        label.set_fontsize(18)
    # Now generate text strings
    # model values are [results, perr] --> [[m, b], [merr, berr]]
    #R = convert_fit_to_resistance(model, fit_type='iv', Rp=Rp.value, Rp_rms=Rp.rms)
    lR = r'$\mathrm{Left \ R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.left.value*1e3, R.left.rms*1e3)
    lOff = r'$\mathrm{Left \ V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.left.result[1]*1e3, model.left.error[1]*1e3)

    sR = r'$\mathrm{R_{sc} - R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.sc.value*1e3, R.sc.rms*1e3)
    sOff = r'$\mathrm{V_{sc,off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.sc.result[1]*1e3, model.sc.error[1]*1e3)

    rR = r'$\mathrm{Right \ R_{n}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(R.right.value*1e3, R.right.rms*1e3)
    rOff = r'$\mathrm{Right \ V_{off}} = %.5f \pm %.5f \mathrm{mV}$'%(model.right.result[1]*1e3, model.right.error[1]*1e3)

    pR = r'$\mathrm{R_{p}} = %.5f \pm %.5f \mathrm{m \Omega}$'%(Rp.value*1e3, Rp.rms*1e3)

    textStr = lR + '\n' + lOff + '\n' + pR + '\n' + sR + '\n' + sOff + '\n' + rR + '\n' + rOff
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    #anchored_text = AnchoredText(textstr, loc=4)
    #axes.add_artist(anchored_text)
    # place a text box in upper left in axes coords
    axes.text(0.65, 0.9, textStr, transform=axes.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    fig.savefig(fName, dpi=150, bbox_inches='tight')
    plt.close('all')
    return None


def make_root_plot(output_path, data_channel, temperature, iv_data, model, Rp, xScale=1, yScale=1):
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
    g0 = rt.TGraphErrors(x.size, x*xScale, y*yScale, xrms*xScale, yrms*yScale)
    g0.SetMarkerSize(0.5)
    g0.SetLineWidth(1)
    g0.SetName("vOut_iBias")
    g0.SetTitle("OutputVoltage vs Bias Current")

    mg.Add(g0)

    # Next up let's add the fit lines
    if model.left.result is not None:
        yFit = lin_sq(x, *model.left.result)
        gLeft = rt.TGraph(x.size, x*xScale, yFit*yScale)
        gLeft.SetMarkerSize(0)
        gLeft.SetLineWidth(2)
        gLeft.SetLineColor(rt.kRed)
        gLeft.SetTitle("Left Normal Branch Fit")
        gLeft.SetName("left_fit")
        mg.Add(gLeft)
    if model.right.result is not None:
        yFit = lin_sq(x, *model.right.result)
        gRight = rt.TGraph(x.size, x*xScale, yFit*yScale)
        gRight.SetMarkerSize(0)
        gRight.SetLineWidth(2)
        gRight.SetLineColor(rt.kGreen)
        gRight.SetTitle("Right Normal Branch Fit")
        gRight.SetName("right_fit")
        mg.Add(gRight)
    if model.sc.result is not None:
        yFit = lin_sq(x, *model.sc.result)
        cut = np.logical_and(yFit < y.max(), yFit > y.min())
        gSC = rt.TGraph(x[cut].size, x[cut]*xScale, yFit[cut]*yScale)
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
    #leg.SetTextFont(2)
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
    #c.BuildLegend()
    c.Update()
    # Now save the damned thing
    c.SaveAs(fName + '.png')
    fName = output_path + '/root/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + str(int(float(temperature)*1e3)) + 'uK'
    c.SaveAs(fName + '.C')
    c.Close()
    del mg
    del c
    return None

