"""Module to write root files based on an input dictionary that mimics the data structure to be written"""
from typing import Any
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt

from array import array
import ROOT as rt


def writeTDir(source : Any, dir_data : dict[str, Any], allowExistingDir : bool=False) -> bool:
    """Write contents of current TDirectory into file.

    Arguments:
        source -- Input filename or path to file to write into
        dir_data -- dictionary describing TDirectory structure

    Keyword Arguments:
        allowExistingDir -- Whether existing directories are permissible (default: {False})

    Returns:
        _description_
    """
    '''Write contents of current TDirectory into file
    dir_data is a dictionary whose keys are the directory names
    The values of each key represent data structures to put INTO the directory
    '''
    result = False
    for dir_name, dir_values in dir_data.items():
        tdir = source.mkdir(dir_name, dir_name, allowExistingDir)
        tdir.cd()
        for dir_key, subvalue in dir_values.items():
            if dir_key == 'TDirectoryFile':
                result = writeTDirF(subvalue)
            elif dir_key == 'TDirectory':
                result = writeTDir(tdir, subvalue)
            elif dir_key == 'TTree':
                result = writeTTree(subvalue)
            elif dir_key == 'vector<double>':
                result = writeVector(tdir, subvalue)
            else:
                result = False
        del tdir
        source.cd()
        # print(rt.gDirectory.pwd())
    return result


def writeVector(tdir: Any, vecdata: dict[str, Any]) -> bool:
    """Write a std::vector<double> object into the current directory directly.

    Arguments:
        tdir -- TDirectory object
        vecdata -- dictionary describing the vectors to be written

    Returns:
        bool indicating success
    """
    std_vector = getattr(getattr(rt, 'std'), 'vector')
    for key, value in vecdata.items():
        vec = std_vector("double")()
        vec.assign(value)
        tdir.WriteObject(vec, key)
        vec.clear()
    return True


def writeTVectorT(tdata : dict[str, Any]) -> bool:
    """Write a TObject into the currently open ROOT file

    Arguments:
        tdata -- dictionary describing the contents of the TObject and its data

    Returns:
        boolean describing success or failure
    """
    result = True
    for key, value in tdata.items():
        TVectorT = getattr(rt, "TVectorT")
        tvector = TVectorT("double")(value.size, value.astype("double"))
        tvector.Write(key)
        del tvector
    return result


def writeTDirF(data : dict[str, Any]):
    '''Write contents of TDirectoryFile dictionary to a particular directory
    Directory file dictionary has list of TNamed objects that can be written
    Structure is a dictionary that contains 2 keys: TDirFileName and TNamed
    TDirFileName is used to create the TDirectoryFile and contains a tuple
    TNamed is a list of tuples, each tuple gives the file name and file data
    '''
    TDirectoryFile = getattr(rt, 'TDirectoryFile')
    tdirfile = TDirectoryFile(data['TDirFileName'][0], data['TDirFileName'][1])
    for item in data['TNamed']:
        TNamed = getattr(rt, 'TNamed')
        tdirfile.Add(TNamed(item[0], item[1]))
    tdirfile.Write()
    return True


def writeTTree(tdata : dict[str, Any]) -> bool:
    '''Create and write a tree to the root file in whatever directory we are in
    Incoming tdata is a dictionary whose keys specify the tree names
    The value of a tree is a dictionary whose keys are branch names and values are branch values
    '''
    result = False
    TTree = getattr(rt, 'TTree')
    for tree_name, tree_values in tdata.items():
        tree = TTree(tree_name, tree_name)
        for subkey, subvalue in tree_values.items():
            if subkey == 'TBranch':
                result = writeTBranch(tree, subvalue)
        del tree
    return result

# branch_data['branchA'] = some kind of array
# branch_data['branchB'] = dictionary of arrays
# branch_data['branchC'] = 2d numpy array of numpy arrays
# branch_data['branchD'] = 2d numpy array of ROOT.RVec objects
# for now I guess type-hint as Any since pyROOT doesn't play nice
def writeTBranch(tree : Any, branch_data : dict[str, Sequence[float | int] | dict[int, Any] | npt.NDArray[np.float64] | npt.NDArray[Any]]) -> bool:
    """Create and write branches to the root file in whatever tree.
    Incoming branch_data is a dictionary where the key specifies the branch name and the value is the branch value
    If the incoming branch_data[key] is itself an array (either a dict of arrays or numpy.ndarray, 2D) it will be 
    a vector<double> branch. In this case the length of the dictionary should be the number of entries

    Arguments:
        tree -- ROOT TTree object
        branch_data -- dictionary describing ROOT data

    Returns:
        Bool to indicate whether it succeeded or not
    """
    
    # Again because pyROOT objects are made dynamically.
    std_vector = getattr(getattr(rt, 'std'), 'vector')
    rvec = getattr(rt, "RVec")
    dloc: dict[str, Any] = {}
    scalar_branches: list[str] = []
    vector_branches: list[str] = []
    nentries = -1
    for branch_key, branch_value in branch_data.items():
        is_vector = isinstance(branch_value, dict) or (
            isinstance(branch_value, np.ndarray) and (
                len(branch_value.shape) == 2 or isinstance(branch_value[0], rvec)))
        # Figure out how many entries we need to write
        # if the branch is a vector we need the length of the container of vectors not the vector
        # if the branch is not a vector the length of the array
        nentries_branch = len(branch_value)
        if nentries < 0:
            nentries = nentries_branch
        elif nentries_branch != nentries:
            raise ValueError(f"Inconsistent number of entries for branch '{branch_key}'.")        
        if is_vector:
            dloc[branch_key] = std_vector("double")()
            tree.Branch(branch_key, "std::vector<double>", dloc[branch_key])
            vector_branches.append(branch_key)
        else:
            if branch_key in ["Timestamp_s", "Timestamp_mus", "NumberOfSamples"]:
                dloc[branch_key] = array("I", [0])
                tree.Branch(branch_key, dloc[branch_key], f"{branch_key}/i")
            else:
                dloc[branch_key] = array("d", [0.0])
                tree.Branch(branch_key, dloc[branch_key], f"{branch_key}/D")
            scalar_branches.append(branch_key)
    # Branches named and set up -- now fill
    for event in range(nentries):
        # Parse the scalars first
        for branch_name in scalar_branches:
            if branch_name in ["Timestamp_s", "Timestamp_mus", "NumberOfSamples"]:
                dloc[branch_name][0] = int(branch_data[branch_name][event])
            else:
                dloc[branch_name][0] = branch_data[branch_name][event]
        # Parse the vectors
        for branch_name in vector_branches:
            dloc[branch_name].assign(np.asarray(branch_data[branch_name][event]))
    tree.Fill()
    return True

def writeROOT(input_file : str, data : dict[str, Any], mode : str ="RECREATE") -> bool:
    """Write a ROOT file. Write a root file. This function works by passing in data as a dictionary. Dictionary keys should be "TDirectory", "TTree", "TBranch", and "TObject". 
    Nested inside each can be appropriate items. For example a TTree key may contain as a value a dictionary of branches, or a TDirectory can contain a list 
    of trees and other such things.
    Example:
    dict = {'TDirectory': {'topLevel': {},'newDir': {'TTree':{'newTree': {'TBranch': {'branchA': 'branchA'}}}}},'TTree':{'tree1': {'TBranch': {'branch1': 'branch1', 'branch2': 'branch2'}}}}

    In this way we basically encode the ROOT hiearchy into a dictionary 

    Arguments:
        input_file -- Name or path of root file to write.
        data -- Data dictionary structured somewhat similar to:
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

    Keyword Arguments:
        mode -- What ROOT mode to pass in for file writing (default: {"RECREATE"})

    Returns:
        Nothing
    """

    # Make the ROOT file first
    TFile = getattr(rt, 'TFile')
    tfile = TFile(input_file, mode)
    allowExistingDir = mode == 'update'
    # tfile.SetCompressionLevel(0)
    # Now we parse the dictionary. It is a good idea here to separate things by type
    result = False
    for key, value in data.items():
        if key == 'TDirectoryFile':
            result = writeTDirF(value)
        if key == 'TDirectory':
            result = writeTDir(tfile, value, allowExistingDir)
        elif key == 'TTree':
            result = writeTTree(value)
        elif key == 'TVectorT':
            result = writeTVectorT(value)
        else:
            result = False
        tfile.cd()
    del tfile
    return result
