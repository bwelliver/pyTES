import tomllib
import argparse
from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pan
import ROOT as rt
import readROOT as rr
import iv_plots as ivplt

def SortRDFNumpy(data: dict[str, Any], sort_key: str) -> dict[str, Any]:
    idx = np.argsort(data[sort_key])
    data = {key: value[idx] for key, value in data.items()}
    return data

def LoadConfig(config_file: str) -> dict[str, Any]:
    """Given input path to a TOML file, parse it.

    Arguments:
        config_file -- Path to a config file in TOML format

    Returns:
        Dictionary containing config options
    """
    
    with open(config_file, "rb") as infile:
        config = tomllib.load(infile)
    return config

def GetRDataFrame(root_file_path: str | list[str], tree_name: str="data_tree") -> Any:
    """Return a ROOT RDataFrame given input file(s) and tree name

    Arguments:
        root_file_path -- A path or list of paths to ROOT files

    Keyword Arguments:
        tree_name -- The tree name to use to create the data frame (default: {"data_tree"})

    Returns:
        RDataFrame
    """
    df = rr.GetRDataFrame(root_file_path, tree_name)
    return df

def input_parser() -> argparse.Namespace:
    """Parse input arguments and return an argument object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="",
                        help='Specify full file path of TOML config file')
    
    args = parser.parse_args()
    return args

def ParseTemperatureSteps(config: dict[str, Any], root_data: Any) -> tuple[Any, int]:
    """Run through the PID log and parse temperature steps.

    The PID log has as the first column the timestamp a PID setting STARTS
    The second column is the power or temperature setting point
    """
    times = pan.read_csv(config["pid_log"], delimiter='\t', header=None, dtype=np.float64) # type: ignore
    have_duration = times.shape[1] == 3
    if have_duration:
        times.columns = ['start_time', 'power', 'duration']
    else:
        # we will construct the durations array based on the difference between start times
        times.columns = ['start_time', 'power']
        times['duration'] = times['start_time'].shift(-1) - times['start_time']
        times.loc[times.shape[0], 'duration'] = times.loc[times.shape[0]-1, 'duration'] # assume same duration for final step as previous step
    times['start_time'] = times['start_time'] + config.get("tz_correction", 0) + config.get("step_start_offset", 300) # adjust for any timezone issues and start offset
    times['end_time'] = times['start_time'] + times['duration'] - config.get("step_stop_offset", 0) # adjust for any stop time offset
    times['step'] = range(0, times.shape[0])
    n_steps = len(times['step']) # type: ignore
    start_times_str = ', '.join(map(str, times['start_time'])) # type: ignore
    end_times_str = ', '.join(map(str, times['end_time'])) # type: ignore
    step_numbers_str = ', '.join(map(str, times['step'])) # type: ignore
    # Declare the arrays in C++
    get_step_str = f"""
    #include <vector>
    const std::vector<double> start_times = {{{start_times_str}}};
    const std::vector<double> end_times = {{{end_times_str}}};
    const std::vector<int> step_numbers = {{{step_numbers_str}}};
    const int n_steps = {n_steps};
    int get_step(const double& timestamp) {{
        int left = 0;
        int right = n_steps - 1;
        while (left <= right) {{
            int mid = (left + right) / 2;
            if (start_times[mid] <= timestamp && timestamp < end_times[mid]) {{
                return step_numbers[mid];
            }} else if (timestamp < start_times[mid]) {{
                right = mid - 1;
            }} else {{
                left = mid + 1;
            }}
        }}
        return -1;  // Return -1 if not found
    }}
    """
    rt.gInterpreter.Declare(get_step_str) # type: ignore
    root_data = root_data.Define("step", "get_step(Timestamp)")
    # Find also the step that corresponds to the lowest temperature
    thermometer: str = config.get("NT", "EPCal")
    data: dict[str, npt.NDArray] = root_data.AsNumpy(columns=[thermometer, "step"]) # type: ignore
    df = pan.DataFrame(data)
    mean_per_step = df.groupby("step")[thermometer].mean()
    std_per_step = df.groupby("step")[thermometer].std()
    min_step: int = mean_per_step.idxmin()
    return root_data, min_step

def GetTempStepsFromPID(config: dict[str, Any], root_data: Any):
    """Return a list of tuples that corresponds to temperature steps.

    Depending on the value of pid_log we will either parse an existing pid log file or if it is None
    attempt to find the temperature steps
    """
    if config.get("pid_log", None) is None or config["pid_log"] is None:
        print("This is not implemented as yet")
        raise Exception("Unsupported configuration. Please supply a pid_log argument")
        # step_values = find_temperature_steps(output_path, time_values, temperatures, thermometer)
    else:
        root_data, min_step = ParseTemperatureSteps(config, root_data)
    # Make diagnostic plot
    t0: float = root_data.Min("Timestamp").GetValue() # type: ignore
    plot_data: dict[str, Any] = root_data.AsNumpy(columns=[config.get("thermometer", "EPCal"), "Timestamp", "step"])
    plot_data = SortRDFNumpy(plot_data, "Timestamp")
    ivplt.test_steps(plot_data["Timestamp"]-t0, plot_data[config.get("thermometer", "EPCal")], plot_data["step"], t0, 'Time', 'T', config.get("output", "") + '/' + 'test_temperature_steps.png')
    return root_data, min_step


def iv_main(config: dict[str, Any]) -> None:
    rdf = GetRDataFrame(config['inputPath'], "data_tree")
    #  We need to identify the various temperature steps to define filters for specific temperature data
    rdf = rdf.Define("Timestamp", "Timestamp_s + Timestamp_mus/1e6")
    rdf, min_step = GetTempStepsFromPID(config, rdf)
    print(f"The minimum step is located at {step=}")


if __name__ == '__main__':
    ARGS = input_parser()
    config = LoadConfig(ARGS.config)
    iv_main(config)
    print('done')