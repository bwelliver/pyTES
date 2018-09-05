import ROOT as rt
from array import array
import numpy as np
#rt.gROOT.ProcessLine("#include<vector>")
#rt.gROOT.ProcessLine("std::vector<double> vec;")
#from ROOT import vec

def writeTDir(source, dirData):
    '''Write contents of current TDirectory into file
    dirData is a dictionary whose keys are the directory names
    The values of each key represent data structures to put INTO the directory
    '''
    for key,value in dirData.items():
        tDir = source.mkdir(key)
        tDir.cd()
        for subkey, subvalue in value.items():
            if subkey == 'TDirectoryFile':
                result = writeTDirF(tDir, subvalue)
            elif subkey == 'TDirectory':
                result = writeTDir(tDir, subvalue)
            elif subkey == 'TTree':
                result = writeTTree(tDir, subvalue)
        del tDir
        source.cd()
        #print(rt.gDirectory.pwd())
    return True


def writeTVectorT(tFile, tData):
    '''Write a TObject into the ROOT file'''
    for key, value in tData.items():
        tVector = rt.TVectorT("double")(value.size, value.astype('double'))
        tVector.Write(key)
        del tVector
    return True


def writeTDirF(tDir, data):
    '''Write contents of TDirectoryFile dictionary to a particular directory
    Directory file dictionary has list of TNamed objects that can be written
    Structure is a dictionary that contains 2 keys: TDirFileName and TNamed
    TDirFileName is used to create the TDirectoryFile and contains a tuple
    TNamed is a list of tuples, each tuple gives the file name and file data
    '''
    tDF = rt.TDirectoryFile(data['TDirFileName'][0], data['TDirFileName'][1])
    for item in data['TNamed']:
        tDF.Add(rt.TNamed(item[0], item[1]))
    tDF.Write()
    return True


def writeTTree(tFile, tData):
    '''Create and write a tree to the root file in whatever directory we are in
    Incoming tData is a dictionary whose keys specify the tree names
    The value of a tree is a dictionary whose keys are branch names and values are branch values
    '''
    for key, value in tData.items():
        tTree = rt.TTree(key, key)
        for subkey, subvalue in value.items():
            if subkey == 'TBranch':
                writeTBranch(tTree, subvalue)
        del tTree
    return True


def writeTBranch(tTree, bData):
    '''Create and write branches to the root file in whatever tree we have
    Incoming bData is a dictionary where the key specifies the branch name and the value is the branch value
    If the incoming bData[key] is itself an array it will be a vector<double> branch.
    '''
    dLoc = {}
    for branchKey in bData.keys():
        if isinstance(bData[branchKey], dict):
            #v = vec
            print('Vector found')
            nEntries = bData[branchKey][0].size
            dLoc[branchKey] = rt.std.vector('double')()
            #dLoc[branchKey] = v
            # Caution: We may need to do something else here about the vectors
            print('The address of the vector is {}'.format(dLoc[branchKey]))
            tTree.Branch(branchKey, 'std::vector<double>', rt.AddressOf(dLoc[branchKey]))
        else:
            nEntries = bData[branchKey].size
            dLoc[branchKey] = array('d', [0.0])
            tTree.Branch(branchKey, dLoc[branchKey], branchKey + '/D')
    # Now loop over the branches and fill them
    for event in range(nEntries):
        for branchKey in bData.keys():
            if isinstance(bData[branchKey], dict):
                #keys of bData[branchKey] are event
                #To get values of event: waveform = bData[branchKey][event]
                if dLoc[branchKey].size() == 0:
                    for value in bData[branchKey][event]:
                        dLoc[branchKey].push_back(float(value))
                else:
                    for idx,value in enumerate(bData[branchKey][event]):
                        dLoc[branchKey][idx] = float(value)
                        #print(dLoc[branchKey][idx])
            else:
                dLoc[branchKey][0] = bData[branchKey][event]
        tTree.Fill()
    tTree.Write()
    return True


def writeROOT(inFile, data):
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
    tFile = rt.TFile(inFile, 'RECREATE')
    # Now we parse the dictionary. It is a good idea here to separate things by type
    for key, value in data.items():
        if key == 'TDirectoryFile':
            result = writeTDirF(tFile, value)
        if key == 'TDirectory':
            result = writeTDir(tFile, value)
        elif key == 'TTree':
            result = writeTTree(tFile, value)
        elif key == 'TVectorT':
            result = writeTVectorT(tFile, value)
        tFile.cd()
    del tFile
    return True