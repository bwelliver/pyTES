import numpy as np
from numpy.typing import NDArray
import ROOT as rt
from typing import Any
print('Enabling explicit MT')
rt.ROOT.EnableImplicitMT()

#TODO: REPLACE ROOT_DICTIONARY WITH A CLASS
#class ROOTDictionary:
#    '''A class to store root objects'''
#
#    def __init__(self):
#        self = PYTDirectory()
#        return None
#
#class PYTDirectory:
#    '''Class to store TDirectories within'''
#
#    def __init__(self):
#        self.TTree = PYTree()
#        self.TBranch = PYBranch()
#        return None
#
#class PYTree:
#    '''Class to store TTrees'''
#
#    def __init__(self):
#        self.TBranch = PYBranch()
#        return None
#
#class PYBranch:
#    '''Class to store branch names'''
#
#    def __init__(self, list_of_branch_names):
#        self.branches = self.branch_names(list_of_branch_names)
#        return Noned
#    def branch_names(list_of_branch_names):
#        return list_of_branch_names
#


def ReadTree(input_files: str | list[str], tree_name: str, branches: list[str], entries: None | int | list[int]=None) -> dict[str, NDArray[Any]]:
    """_summary_

    Arguments:
        input_files -- _description_
        tree_name -- _description_
        branches -- _description_

    Keyword Arguments:
        entries -- _description_ (default: {None})

    Returns:
        _description_
    """
    rdf = getattr(rt, "RDataFrame")
    df = rdf(tree_name, input_files)
    if entries is not None:
        if isinstance(entries, int):
            df = df.Range(entries)
        elif isinstance(entries, list):
            entries_set = set(entries)
            df = df.Define("entry", "rdfentry_").Filter(lambda e: e in entries_set, ["entry"])
    arrays = df.AsNumpy(columns=branches)
    return arrays


def ReadTVector(input_files: str | list[str], tvector_name: str) -> NDArray[Any] | None:
    """_summary_

    Arguments:
        input_files -- _description_
        tvector_name -- _description_

    Returns:
        _description_
    """
    file_name = input_files[0] if isinstance(input_files, list) else input_files
    TFile = getattr(rt, "TFile")
    tfile = TFile.Open(file_name, "READ")
    tvector = tfile.Get(tvector_name)
    if tvector and tvector.ClassName() == "TVectorT<double>":
        data = np.array(tvector)
    else:
        obj_type = tvector.ClassName() if tvector else "None"
        print(f"Requested object {tvector_name} is not a TVectorT<double>. It is a {obj_type}")
        data = None
    tfile.Close()
    return data
    

def GetROOTData(input_files: str | list[str], 
             root_dictionary: dict[str, Any], 
             path: str="", 
             entries: None | int | list[int]=None) -> dict[str, Any]:
    """Based on input pyTES ROOT dictionary fetch the requested data

    Arguments:
        input_files -- _description_
        root_dictionary -- _description_
        read_method -- _description_
        entries -- _description_

    Returns:
        _description_
    """
    data: dict[str, Any] = {}
    for rkey, rval in root_dictionary.items():
        if rkey == 'TDirectory':
            for dir_name, dir_content in rval.items():
                dpath = f"{path}/{dir_name}".strip("/")
                dir_data = GetROOTData(input_files, dir_content, path=dpath, entries=entries)
                data.update(dir_data)
        elif rkey == "TTree":
            for tree_name, branches in rval.items():
                full_tree = f"{path}/{tree_name}".strip("/")
                tree_data = ReadTree(input_files, full_tree, branches, entries)
                data[full_tree] = tree_data
        elif rkey == "TBranch":
            pass
        elif rkey == "TVector":
            for vector_name in rval:
                full_vector_name = f"{path}/{vector_name}".strip("/")
                vector_data = ReadTVector(input_files, full_vector_name)
                data[full_vector_name] = vector_data
        else:
            print(f"Unknown key type {rkey} in root_dictionary")
    return data

def ReadROOT(input_files: str | list[str], 
             root_dictionary: dict[str, Any], 
             read_method: str="", 
             entries: None | int | list[int]=None) -> dict[str, Any]:
    """Primary interface to read data from a ROOT file and output into a pyTES ROOT dictionary format.

    Arguments:
        input_files -- The file or list of files to read. Can use wildcards as well.
        root_dictionary -- pyTES ROOT dictionary descriptor
        {'TDirectory': {DirectoryName1: {'TTree': {Tree1: [Branch1, Branch2, Branch3 ...], 'TBranch': [BranchA, BranchB, ...]} } }, 'TTree': {...} }

    Keyword Arguments:
        read_method -- Method to use when dealing with multiple files (default: {"chain"})
        entries -- Get specific entry or entries. Default of 'None' means get all (default: {None})

    Returns:
        ROOT data in pyTES dictionary format
    """
    data = GetROOTData(input_files, root_dictionary, read_method, entries)
    
    return data
