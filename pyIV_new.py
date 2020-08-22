import os
import time
from os.path import isabs, dirname, basename
import argparse

import numpy as np
import pandas as pan

from iv_input_arguments import InputArguments
import iv_processor
import tes_parameters
import iv_plots as ivplt
import tes_characterization as tes_char
import squid_info

import ROOT as rt
from readROOT import readROOT
from writeROOT import writeROOT


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


def save_iv_fits_to_root(output_file, iv_dictionary, branches=None):
    '''Function to save iv fit data to a root file
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
            print('The key is: {}'.format(key))
            if key == 'fit_parameters':
                value = value.get_dict()
            if branches is not None:
                if key in branches:
                    data['TTree']['T' + temperature]['TBranch'][key] = value
            else:
                data['TTree']['T' + temperature]['TBranch'][key] = value
    # print(data)
    # We should also make an object that tells us what the other tree values are
    # data['TDirectory']['iv']['TTree']['names']['TBranch'] =
    status = writeROOT(output_file, data)
    return status


def save_iv_to_root(output_file, iv_dictionary, branches=None):
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
            if branches is not None:
                if key in branches:
                    data['TTree']['T' + temperature]['TBranch'][key] = value
            else:
                data['TTree']['T' + temperature]['TBranch'][key] = value
    # print(data)
    # We should also make an object that tells us what the other tree values are
    # data['TDirectory']['iv']['TTree']['names']['TBranch'] =
    status = writeROOT(output_file, data)
    return status


def format_and_make_output_path(path, output_path):
    '''This function will format the output path into an absolute path and make
    the directory if needed.
    '''
    output_path = output_path if output_path else dirname(path) + '/' + basename(path).replace('.root', '')
    if not isabs(output_path):
        output_path = dirname(path) + '/' + output_path
    mkdpaths(output_path)
    return output_path


def get_iv_data_from_file(input_path, new_format=False, thermometer='EP'):
    '''Load IV data from specified directory'''
    if thermometer == 'EP':
        thermometer_name = 'EPCal_K'
    elif thermometer == 'NT':
        thermometer_name = 'NT'
    elif thermometer == 'ExpRuOx':
        thermometer_name = 'ExpRuOx_K'
    if new_format is False:
        tree = 'data_tree'
        branches = ['Channel', 'NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', 'Waveform', thermometer_name]
        method = 'chain'
        iv_data = readROOT(input_path, tree, branches, method)
        # Make life easier:
        iv_data = iv_data['data']
    else:
        chlist = 'ChList'
        channels = readROOT(input_path, tree=None, branches=None, method='single', tobject=chlist)
        channels = channels['data'][chlist]
        branches = ['NumberOfSamples', 'Timestamp_s', 'Timestamp_mus', 'SamplingWidth_s', thermometer_name]
        branches = branches + ['Waveform' + '{:03d}'.format(int(i)) for i in channels]
        print('Branches to be read are: {}'.format(branches))
        tree = 'data_tree'
        method = 'chain'
        iv_data = readROOT(input_path, tree, branches, method)
        iv_data = iv_data['data']
        iv_data['Channel'] = channels
    return iv_data


def parse_temperature_steps(output_path, time_values, temperatures, pid_log, tz_correction):
    '''Run through the PID log and parse temperature steps
    The PID log has as the first column the timestamp a PID setting STARTS
    The second column is the power or temperature setting point
    '''
    times = pan.read_csv(pid_log, delimiter='\t', header=None)
    times = times.values[:, 0]
    times = times + tz_correction  # adjust for any timezone issues
    # Each index of times is now the starting time of a temperature step.
    # Include an appropriate offset for mean computation BUT only a softer one for time boundaries
    # time_list is a list of tuples.
    time_list = []
    start_offset = 60
    end_offset = 10
    if times.size > 1:
        for index in range(times.size - 1):
            cut = np.logical_and(time_values > times[index]+start_offset, time_values < times[index+1] - end_offset)
            mean_temperature = np.mean(temperatures[cut])
            serr_temperature = np.std(temperatures[cut])/np.sqrt(temperatures[cut].size)
            start_time = times[index] + start_offset
            stop_time = times[index + 1] - end_offset
            time_list.append((start_time, stop_time, mean_temperature, serr_temperature))
        # Handle the last step
        # How long was the previous step?
        d_t = time_list[0][1] - time_list[0][0]
        start_time = time_list[-1][1] + start_offset
        end_time = start_time + d_t - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mean_temperature = np.mean(temperatures[cut])
        serr_temperature = np.std(temperatures[cut])/np.sqrt(temperatures[cut].size)
        time_list.append((start_time, end_time, mean_temperature, serr_temperature))
        d_t = time_values - time_values[0]
    else:
        # Only 1 temperature step defined
        start_time = times[0] + start_offset
        end_time = time_values[-1] - end_offset
        cut = np.logical_and(time_values > start_time, time_values < end_time)
        mean_temperature = np.mean(temperatures[cut])
        time_list.append((start_time, end_time, mean_temperature, serr_temperature))
        d_t = time_values - time_values[0]
    ivplt.test_steps(d_t, temperatures, time_list, time_values[0], 'Time', 'T', output_path + '/' + 'test_temperature_steps.png')
    return time_list


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


def chop_data_by_temperature_steps(iv_data, timelist, thermometer_name, bias_channel, data_channel, squid):
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
    cut_temperature_min = -1000  # Should be the minimum rejected temperature
    expected_duration = 2600  # TODO: make this an input argument or auto-determined somehow
    # Now chop up the IV data into steps keyed by the mean temperature
    for values in timelist:
        start_time, stop_time, mean_temperature, serr_temperature = values
        print('The mean temperature is: {}'.format(mean_temperature))
        times = iv_data['Timestamp_s'] + iv_data['Timestamp_mus']/1e6
        cut = np.logical_and(times >= start_time + time_buffer, times <= np.min([stop_time, start_time + expected_duration]))
        # Warning: iv_data[WaveformXYZ] is a dictionary! Its keys are event numbers and its values are the samples.
        n_events = len(iv_data['Waveform' + '{:03d}'.format(int(bias_channel))])
        sz_array = iv_data['Waveform' + '{:03d}'.format(int(bias_channel))][0].size
        print('Size check: The size of times is: {}, the number of events in waveform is: {}, the size of the waveform itself is: {}'.format(times.size, n_events, sz_array))
        print('Converting iBias from dictionary of arrays to a 2d array')
        bias = np.empty((n_events, sz_array))
        for event, sample in iv_data['Waveform' + '{:03d}'.format(int(bias_channel))].items():
            bias[event] = sample
        bias = bias/r_bias
        n_events = len(iv_data['Waveform' + '{:03d}'.format(int(data_channel))])
        sz_array = iv_data['Waveform' + '{:03d}'.format(int(data_channel))][0].size
        v_out = np.empty((n_events, sz_array))
        print('Converting vOut from dictionary of arrays to a 2d array')
        for event, sample in iv_data['Waveform' + '{:03d}'.format(int(data_channel))].items():
            v_out[event] = sample
        times = times[cut]
        bias = bias[cut]
        v_out = v_out[cut]
        temperatures = iv_data[thermometer_name][cut]
        sampling_width = iv_data['SamplingWidth_s'][cut]
        temperature_key = str(np.round(mean_temperature*1e3, 3))
        # Toss out any T values where the digitizer has railed
        if np.std(v_out)/np.sqrt(v_out.size) < 1e-15:
            print('Invalid digitizer response for T: {} mK'.format(temperature_key))
            continue
        if stop_time - start_time > expected_duration:
            print('Temperature step is too long for T: {} mK. End: {}, Start: {}, Duration: {}'.format(temperature_key, stop_time, start_time, stop_time - start_time))
            continue
        if cut_temperature_min < mean_temperature*1e3 < cut_temperature_max:
            continue
        else:
            # Make sure things are sorted by increasing time
            sort_key = np.argsort(times)
            times = times[sort_key]
            bias = bias[sort_key]
            v_out = v_out[sort_key]
            temperatures = temperatures[sort_key]
            print('Creating dictionary entry for T: {} mK'.format(temperature_key))
            iv_dictionary[temperature_key] = {
                    'iBias': bias,
                    'vOut': v_out,
                    'timestamps': times,
                    'temperatures': temperatures,
                    'sampling_width': sampling_width
                    }
    return iv_dictionary, sz_array


def get_iv_temperature_data(argin):
    '''Based on requested input data, load the IV data into memory and split by temperatures'''
    # Step 1: Load IV data into memory

    iv_data = get_iv_data_from_file(input_path=argin.inputPath, new_format=argin.newFormat, thermometer=argin.thermometer)
    # Step 2: Get temperature steps
    time_values = iv_data['Timestamp_s'] + iv_data['Timestamp_mus']/1e6
    if argin.thermometer == 'EP':
        thermometer_name = 'EPCal_K'
    elif argin.thermometer == 'NT':
        thermometer_name = 'NT'
    elif argin.thermometer == 'ExpRuOx':
        thermometer_name = 'ExpRuOx_K'
    timelist = get_temperature_steps(argin.outputPath, time_values=time_values, temperatures=iv_data[thermometer_name], pid_log=argin.pidLog, thermometer=argin.thermometer, tz_correction=argin.tzOffset)
    iv_dictionary, number_samples = chop_data_by_temperature_steps(iv_data, timelist, thermometer_name, argin.biasChannel, argin.dataChannel, argin.squid)
    return iv_dictionary, number_samples


def plot_iv_curves(output_path, data_channel, number_of_windows, squid, iv_dictionary):
    '''Simple function to loop through IV data and generate the plots'''
    # Next loop through to generate plots
    iv_curves = iv_processor.iv_windower(iv_dictionary, number_of_windows)
    min_temperature = list(iv_dictionary.keys())[np.argmin([float(temperature) for temperature in iv_dictionary.keys()])]
    r_parasitic = iv_dictionary[min_temperature]['parasitic']
    for temperature, iv_data in sorted(iv_dictionary.items()):
        # Make I-V plot
        file_name = output_path + '/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
        plt_data = [iv_curves[temperature]['iBias'], iv_curves[temperature]['vOut'], iv_curves[temperature]['iBias_rms'], iv_curves[temperature]['vOut_rms']]
        axes_options = {'xlabel': 'Bias Current [uA]',
                        'ylabel': 'Output Voltage [mV]',
                        'title': 'Channel {} Output Voltage vs Bias Current for temperatures = {} mK'.format(data_channel, temperature)
                        }
        model_resistance = iv_processor.convert_fit_to_resistance(iv_data['fit_parameters'], squid, fit_type='iv', r_p=r_parasitic.value, r_p_rms=r_parasitic.rms)
        ivplt.iv_fitplot(plt_data, iv_data['fit_parameters'], model_resistance, r_parasitic, file_name, axes_options, xscale=1e6, yscale=1e3)
        # Let's make a ROOT style plot (yuck)
        # ivplt.make_root_plot(output_path, data_channel, temperature, iv_data, fit_parameters_dictionary[temperature], parasitic_dictionary[min_temperature], xscale=1e6, yscale=1e3)
    return True


def process_tes_curves(iv_dictionary, number_of_windows=0, slew_rate=1, number_samples=None):
    '''Process the IV TES data and find Rsc and Rn values
    Here we assume incoming data streams are all ndarrays of some shape and not dictionaries
    '''

    # 1. window the data
    iv_curves = iv_processor.iv_windower(iv_dictionary, number_of_windows, mode='tes')
    # 2. Get fit parameters
    for temperature, iv_data in sorted(iv_curves.items()):
        print('Processing TES data for temperature = {} mK'.format(temperature))
        st = time.time()
        fit_params = iv_processor.fit_iv_regions(xdata=iv_data['vTES'], ydata=iv_data['iTES'], sigma_y=iv_data['vTES_rms'], number_samples=number_samples, sampling_width=iv_dictionary[temperature]['sampling_width'][0], number_of_windows=number_of_windows, slew_rate=slew_rate, plane='tes')
        print('It took the fit protocol {} s to complete.'.format(time.time() - st))
        iv_dictionary[temperature]['tes_fit_parameters'] = fit_params
    return iv_dictionary, iv_curves


def process_iv_curves(squid, iv_dictionary, number_of_windows=0, slew_rate=1, number_samples=None):
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

    # First determine if we need to split these data up into windows
    # Note: iBias and vOut is at this point a nEvent x nSamples array and not a dict
    # Variable lookup:
    # iv_dictionary: Contains all the raw IV data, i, v, t, dt, T
    # iv_curves: Contains the 'windowed' IV data for just i, ierr, v, verr, t
    iv_curves = iv_processor.iv_windower(iv_dictionary, number_of_windows)
    # Next try to obtain a measure of the parasitic series resistance. Note that this value is subtracted
    # from subsquent fitted values of the TES resistance and is assumed to be T indepdent.
    sampling_width = iv_dictionary[list(iv_dictionary.keys())[0]]['sampling_width'][0]
    parasitic_dictionary, min_temperature = iv_processor.get_parasitic_resistances(iv_curves, squid, number_samples, sampling_width, number_of_windows, slew_rate)
    # Loop through the iv data now and obtain fit parameters and correct alignment
    # fit_parameters_dictionary = {}
    for temperature, iv_data in sorted(iv_curves.items()):
        fit_parameters = iv_processor.fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigma_y=iv_data['vOut_rms'], number_samples=number_samples, sampling_width=iv_dictionary[temperature]['sampling_width'][0], number_of_windows=number_of_windows, slew_rate=slew_rate, plane='iv')
        # Make it pass through zero. Correct offset.
        # i_offset, v_offset = correct_offsets(fit_parameters_dictionary[temperature], iv_data, 'interceptbalance')
        i_offset, v_offset = iv_processor.correct_offsets(fit_parameters, iv_data, 'dual')
        print('The maximum iBias={} and the minimum iBias={} with a total size={}'.format(iv_data['iBias'].max(), iv_data['iBias'].min(), iv_data['iBias'].size))
        print('For temperature {} the normal offset adjustment value to subtract from vOut is: {} and from iBias: {}'.format(temperature, v_offset, i_offset))
        iv_data['vOut'] -= v_offset
        iv_data['iBias'] -= i_offset
        # Get fit information
        fit_parameters = iv_processor.fit_iv_regions(xdata=iv_data['iBias'], ydata=iv_data['vOut'], sigma_y=iv_data['vOut_rms'], number_samples=number_samples, sampling_width=iv_dictionary[temperature]['sampling_width'][0], number_of_windows=number_of_windows, slew_rate=slew_rate, plane='iv')
        # Correct the main dictionary as well and stuff other things inside
        iv_dictionary[temperature]['vOut'] -= v_offset
        iv_dictionary[temperature]['iBias'] -= i_offset
        iv_dictionary[temperature]['fit_parameters'] = fit_parameters
        iv_dictionary[temperature]['parasitic'] = parasitic_dictionary[temperature]
#    # Next loop through to generate plots
#    for temperature, iv_data in sorted(iv_curves.items()):
#        # Make I-V plot
#        file_name = output_path + '/' + 'vOut_vs_iBias_ch_' + str(data_channel) + '_' + temperature + 'mK'
#        plt_data = [iv_data['iBias'], iv_data['vOut'], iv_data['iBias_rms'], iv_data['vOut_rms']]
#        axes_options = {'xlabel': 'Bias Current [uA]',
#                        'ylabel': 'Output Voltage [mV]',
#                        'title': 'Channel {} Output Voltage vs Bias Current for temperatures = {} mK'.format(data_channel, temperature)
#                        }
#        model_resistance = iv_processor.convert_fit_to_resistance(fit_parameters_dictionary[temperature], squid, fit_type='iv', r_p=parasitic_dictionary[min_temperature].value, r_p_rms=parasitic_dictionary[min_temperature].rms)
#        ivplt.iv_fitplot(plt_data, fit_parameters_dictionary[temperature], model_resistance, parasitic_dictionary[min_temperature], file_name, axes_options, xscale=1e6, yscale=1e3)
        # Let's make a ROOT style plot (yuck)
        # ivplt.make_root_plot(output_path, data_channel, temperature, iv_data, fit_parameters_dictionary[temperature], parasitic_dictionary[min_temperature], xscale=1e6, yscale=1e3)
    # Let's figure out what this function should return:
    # 1. iv_dictionary --> now offset corrected
    # 2. min_temp, fit_parameters, parasitics --> last 2 are keyed by T like iv_dictionary...combine into it?
    # 3. Do we return iv_curves or since we use numba it is probably OK to re-do it as need be later :)
    return iv_dictionary


def get_tes_values(iv_dictionary, squid):
    '''From the (now offset corrected) IV data, compute the TES values of:
        iTES, vTES, rTES, pTES
    and insert these into the iv_dictionary
    '''
    squid_parameters = squid_info.SQUIDParameters(squid)
    r_sh = squid_parameters.Rsh
    m_ratio = squid_parameters.M
    r_fb = squid_parameters.Rfb
    # Test: Select the parasitic resistance from the lowest temperature fit to use for everything
    min_temperature = list(iv_dictionary.keys())[np.argmin([float(temperature) for temperature in iv_dictionary.keys()])]
    r_parasitic = iv_dictionary[min_temperature]['parasitic']
    r_p = r_parasitic.value
    for iv_data in iv_dictionary.values():
        iTES = tes_parameters.compute_iTES(iv_data['vOut'], r_fb, m_ratio)
        rTES = tes_parameters.compute_rTES(iv_data['iBias'], iTES, r_sh, r_p)
        vTES = tes_parameters.compute_vTES(iv_data['iBias'], iTES, r_sh, r_p)
        pTES = tes_parameters.compute_pTES(iTES, vTES)
        print('Shape check:\n\tThe shape of vOut is: {}\n\tThe shape of iTES is: {}'.format(iv_data['vOut'].shape, iTES.shape))
        iv_data['iTES'] = iTES
        iv_data['vTES'] = vTES
        iv_data['rTES'] = rTES
        iv_data['pTES'] = pTES
    return iv_dictionary


def tes_plots(output_path, data_channel, squid, temperature, data):
    '''Helper function to generate standard TES plots
    iTES vs vTES
    rTES vs iTES
    rTES vs vTES
    rTES vs iBias
    pTES vs rTES
    pTES vs vTES
    '''
    # Current vs Voltage
    ivplt.plot_current_vs_voltage(output_path, data_channel, squid, temperature, data)
    # Resistance vs Current
    ivplt.plot_resistance_vs_current(output_path, data_channel, temperature, data)
    # Resistance vs Voltage
    ivplt.plot_resistance_vs_voltage(output_path, data_channel, temperature, data)
    # Resistance vs Bias Current
    ivplt.plot_resistance_vs_bias_current(output_path, data_channel, temperature, data)
    # Power vs rTES
    ivplt.plot_power_vs_resistance(output_path, data_channel, temperature, data)
    # Power vs vTES
    ivplt.plot_power_vs_voltage(output_path, data_channel, temperature, data)
    return True


def make_tes_plots(output_path, data_channel, squid, number_of_windows, iv_dictionary, individual=False):
    '''Loop through data to generate TES specific plots'''
    # Generate windowed values if needed.
    iv_curves = iv_processor.iv_windower(iv_dictionary, number_of_windows, mode='tes')
    for temperature, data in iv_curves.items():
        data['tes_fit_parameters'] = iv_dictionary[temperature]['tes_fit_parameters']
    if individual is True:
        for temperature, data in iv_curves.items():
            print('The keys in the data are: {}'.format(data.keys()))
            tes_plots(output_path, data_channel, squid, temperature, data)
    # Make a for all temperatures here
    ivplt.make_tes_multiplot(output_path=output_path, data_channel=data_channel, squid=squid, iv_dictionary=iv_curves)
    return True


def iv_main(argin):
    '''The main IV processing function that will call other functions in order
    to process the entire IV dataset
    '''
    argin.outputPath = format_and_make_output_path(argin.inputPath, argin.outputPath)
    print('We will run with the following options:')
    print('The squid run is {}'.format(argin.run))
    print('The SQUID is: {}'.format(argin.squid))
    print('The output path is: {}'.format(argin.outputPath))
    mkdpaths(argin.outputPath + '/root')
    if argin.readROOT is False and argin.readTESROOT is False:
        # Step 1: Load IV data into memory from the base data files and split it up into distinct temperature steps
        # It will be a dictionary whose keys are temperatures (in mK). The sub-dictionary has keys that are iBias, vOut, timestamps, temperatures, sampling_width
        iv_dictionary, number_samples = get_iv_temperature_data(argin)
        # Step 2: Save these chopped up IV data
        output_file = argin.outputPath + '/root/iv_data.root'
        save_iv_to_root(output_file, iv_dictionary)
    if argin.readROOT is True and argin.readTESROOT is False:
        # This loads saved data from steps 1 and 2 if it has been performed already and will put us in a state to proceed with TES quantity computations
        iv_dictionary = read_from_ivroot(argin.outputPath + '/root/iv_data.root', branches=['iBias', 'vOut', 'timestamps', 'temperatures', 'sampling_width'])
        # Things will be dicts here so we should convert to arrays
        for temperature, iv_data in iv_dictionary.items():
            for key, value in iv_data.items():
                if isinstance(value, dict):
                    iv_data[key], number_samples = iv_processor.convert_dict_to_ndarray(value)
    if argin.readTESROOT is False:
        # Step 3: Process IV data and correct (i,v) offsets, get Rp, and generate plots
        iv_dictionary = process_iv_curves(argin.squid, iv_dictionary, argin.numberOfWindows, argin.slewRate, number_samples)
        plot_iv_curves(argin.outputPath, argin.dataChannel, argin.numberOfWindows, argin.squid, iv_dictionary)
        # TODO: Save fit and parasitics?
        #output_file = argin.outputPath + '/root/iv_fit_parameters.root'
        #save_iv_fits_to_root(output_file, iv_dictionary, branches=['fit_parameters', 'parasitic'])
        # Step 4: Using (i,v) compute TES quantities and insert into the iv_dictionary
        iv_dictionary = get_tes_values(iv_dictionary, argin.squid)
        # Save actual data
        print('Saving ROOT file with TES quantities computed')
        output_file = argin.outputPath + '/root/iv_data_processed.root'
        save_iv_to_root(output_file, iv_dictionary, branches=['iBias', 'vOut', 'timestamps', 'temperatures', 'sampling_width', 'iTES', 'rTES', 'vTES', 'pTES'])
    if argin.readTESROOT is True:
        # NOTE! IV data loaded is is already processed so the 0-offset correction is applied.
        iv_dictionary = read_from_ivroot(argin.outputPath + '/root/iv_data_processed.root', branches=['iBias', 'vOut', 'timestamps', 'temperatures', 'sampling_width', 'iTES', 'rTES', 'vTES', 'pTES'])
        # Things will be dicts here so we should convert to arrays
        for temperature, iv_data in iv_dictionary.items():
            print('Converting data to ndarray for temperature: {}'.format(temperature))
            for key, value in iv_data.items():
                if isinstance(value, dict):
                    iv_data[key], number_samples = iv_processor.convert_dict_to_ndarray(value)
                    # iv_dictionary[temperature][key] = new_value
    # Step 5: Process the TES values
    print('The number_samples argument is: {}'.format(number_samples))
    iv_dictionary, iv_curves = process_tes_curves(iv_dictionary, number_of_windows=argin.numberOfWindows, slew_rate=argin.slewRate, number_samples=number_samples)
    if argin.plotTES is True:
        make_tes_plots(output_path=argin.outputPath, data_channel=argin.dataChannel, squid=argin.squid, number_of_windows=argin.numberOfWindows, iv_dictionary=iv_dictionary, individual=False)
    # Step 6: Compute interesting curves
    iv_dictionary = tes_char.find_normal_to_sc_data(iv_dictionary, argin.numberOfWindows, iv_curves=iv_curves)
    tc, rN, temp, R, R_sigma = tes_char.get_resistance_temperature_curves_new(argin.outputPath, argin.dataChannel, argin.numberOfWindows, iv_dictionary)
    output_file = argin.outputPath + '/root/rt_data.root'
    rt_data = {'rtdata': {'temperature': temp, 'rTES': R, 'rTES_sigma': R_sigma}}
    save_iv_to_root(output_file, rt_data, branches=['rTES', 'rTES_sigma', 'temperature'])
    temp, power, power_sigma, pfitResults = tes_char.get_power_temperature_curves(argin.outputPath, argin.dataChannel, argin.numberOfWindows, iv_dictionary, tc=tc, rN=rN)
    output_file = argin.outputPath + '/root/pt_data.root'
    pt_data = {'ptdata': {'temperature': temp, 'pTES': power, 'pTES_sigma': power_sigma}}
    save_iv_to_root(output_file, pt_data, branches=['pTES', 'pTES_sigma', 'temperature'])
    # Let's generate corrected RT curves now using PT data to get Ttes.
    pt_data['fit'] = pfitResults
    tes_char.get_corrected_resistance_temperature_curves(argin.outputPath, argin.dataChannel, argin.numberOfWindows, iv_dictionary, pt_data)
    return True


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
    parser.add_argument('-R', '--slewRate', type=float,
                        help='Specify the slew rate of the underlying ramp function used to generate the IV curve in units of uA/s.')
    parser.add_argument('-T', '--thermometer', default='EP',
                        help='Specify the name of the thermometer to use. Can be either EP for EPCal (default) or NT for the noise thermometer, or ExpRuOx for experimental RuOx')
    parser.add_argument('-S', '--squid', help='Specify the serial number of the SQUID being used.')
    parser.add_argument('-z', '--tzOffset', default=0.0, type=float,
                        help='The number of hours of timezone offset to use.\
                        Default is 0 and assumes timestamps to convert are from the same timezone.\
                        If you need to convert to an earlier timezone use a negative number.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = input_parser()
    ARGIN = InputArguments()
    ARGIN.set_from_args(ARGS)
    iv_main(ARGIN)
    print('done')
