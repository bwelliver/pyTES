'''Module to write root files based on an input dictionary that mimics the data structure to be written'''

import numpy as np

from array import array
import ROOT as rt
# rt.gROOT.ProcessLine("#include<vector>")
# rt.gROOT.ProcessLine("std::vector<double> vec;")
# from ROOT import vec


def writeTDir(source, dir_data):
    '''Write contents of current TDirectory into file
    dir_data is a dictionary whose keys are the directory names
    The values of each key represent data structures to put INTO the directory
    '''
    for key, value in dir_data.items():
        tdir = source.mkdir(key)
        tdir.cd()
        for subkey, subvalue in value.items():
            if subkey == 'TDirectoryFile':
                result = writeTDirF(subvalue)
            elif subkey == 'TDirectory':
                result = writeTDir(tdir, subvalue)
            elif subkey == 'TTree':
                result = writeTTree(subvalue)
            else:
                result = None
        del tdir
        source.cd()
        # print(rt.gDirectory.pwd())
    return result


def writeTVectorT(tdata):
    '''Write a TObject into the ROOT file'''
    for key, value in tdata.items():
        TVectorT = getattr(rt, 'TVectorT')
        tvector = TVectorT("double")(value.size, value.astype('double'))
        tvector.Write(key)
        del tvector
    return True


def writeTDirF(data):
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


def writeTTree(tdata):
    '''Create and write a tree to the root file in whatever directory we are in
    Incoming tdata is a dictionary whose keys specify the tree names
    The value of a tree is a dictionary whose keys are branch names and values are branch values
    '''
    for key, value in tdata.items():
        TTree = getattr(rt, 'TTree')
        tree = TTree(key, key)
        for subkey, subvalue in value.items():
            if subkey == 'TBranch':
                writeTBranch(tree, subvalue)
        del tree
    return True


def writeTBranch(tree, branch_data):
    '''Create and write branches to the root file in whatever tree we have
    Incoming branch_data is a dictionary where the key specifies the branch name and the value is the branch value
    If the incoming branch_data[key] is itself an array (either a dict of arrays or numpy.ndarray, 2D) it will be a vector<double> branch.
    '''
    dloc = {}
    for branchkey in branch_data.keys():
        if isinstance(branch_data[branchkey], dict) or (isinstance(branch_data[branchkey], np.ndarray) and len(branch_data[branchkey].shape) == 2):
            # v = vec
            # print('Vector found')
            nentries = branch_data[branchkey][0].size
            std = getattr(rt, 'std')
            # We can try to pre-allocate the vector by putting nentries at the end
            dloc[branchkey] = std.vector('double')(nentries)
            # dloc[branchkey] = v
            # Caution: We may need to do something else here about the vectors
            # print('The address of the vector is {}'.format(dloc[branchkey]))
            AddressOf = getattr(rt, 'AddressOf')
            tree.Branch(branchkey, 'std::vector<double>', AddressOf(dloc[branchkey]))
        else:
            nentries = branch_data[branchkey].size
            dloc[branchkey] = array('d', [0.0])
            tree.Branch(branchkey, dloc[branchkey], branchkey + '/D')
    # Now loop over the branches and fill them
    for event in range(nentries):
        for branchkey in branch_data.keys():
            if isinstance(branch_data[branchkey], dict) or (isinstance(branch_data[branchkey], np.ndarray) and len(branch_data[branchkey].shape) == 2):
                # keys of branch_data[branchkey] are event
                # To get values of event: waveform = branch_data[branchkey][event]
                if dloc[branchkey].size() == 0:
                    # print('Using push_back for branch {} and event {}'.format(branchkey, event))
                    for value in branch_data[branchkey][event]:
                        dloc[branchkey].push_back(float(value))
                else:
                    # print('Using indexed assignment for branch {} and event {}'.format(branchkey, event))
                    for idx, value in enumerate(branch_data[branchkey][event]):
                        dloc[branchkey][idx] = float(value)
                        # print(dloc[branchkey][idx])
            else:
                dloc[branchkey][0] = branch_data[branchkey][event]
        tree.Fill()
    tree.Write()
    return True


def writeROOT(input_file, data):
    '''Write a root file. This function works by passing in data as a dictionary. Dictionary keys should be "TDirectory", "TTree", "TBranch", and "TObject". Nested inside each can be appropriate items. For example a TTree key may contain as a value a dictionary of branches, or a TDirectory can contain a list of trees and other such things.
    Example:
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
    dict = {'TDirectory': {'topLevel': {},'newDir': {'TTree':{'newTree': {'TBranch': {'branchA': 'branchA'}}}}},'TTree':{'tree1': {'TBranch': {'branch1': 'branch1', 'branch2': 'branch2'}}}}

    In this way we basically encode the ROOT hiearchy into a dictionary'''

    # Make the ROOT file first
    TFile = getattr(rt, 'TFile')
    tfile = TFile(input_file, 'RECREATE')
    # Now we parse the dictionary. It is a good idea here to separate things by type
    for key, value in data.items():
        if key == 'TDirectoryFile':
            result = writeTDirF(value)
        if key == 'TDirectory':
            result = writeTDir(tfile, value)
        elif key == 'TTree':
            result = writeTTree(value)
        elif key == 'TVectorT':
            result = writeTVectorT(value)
        else:
            result = None
        tfile.cd()
    del tfile
    return result
