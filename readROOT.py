import numpy as np
import ROOT as rt

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



def root_dictionary_walker():
    '''Function that will walk through a root_dictionary and do something?'''
    for key, value in root_dictionary:
        if key == 'TDirectory':
            root_dictionary_walker(value, 'TDirectory')
        elif key == 'TTree':
            root_dictionary_walker(value, 'TTree')
        elif key == 'TBranch':
            root_dictionary_walk(value, 'TBranch')



def readROOT_new(input_file, root_dictionary, read_method='single'):
    '''A function to make loading data from ROOT files into numpy arrays more flexible.
    This function reads from an input root file
    The root_dictionary is a specially formatted dictionary of names, similar to the interface in writeROOT.
    root_dictionary = {'TDirectory': {DirectoryName1: {'TTree': {Tree1: [Branch1, Branch2, Branch3 ...], 'TBranch': [BranchA, BranchB, ...]} } }, 'TTree': {...} }
    
    As an example we can get a list of branch names in Dir/TreeA as follows:
    branches = root_dictionary['TDirectory'][Dir]['TTree'][TreeA]
    
    read_method defines if we are opening the TFile directly (single) or using a TChain (chain).
    
    '''
    
    # The course of action depends if we are chaining or not
    # If we have a TChain note that these chain together *TREES* across files.
    # The standard format is to call the chain as "dirName/treeName"
    # So let's recreate the hiearchy as need be then I guess
    
    if read_method == 'chain':
        for key, value in root_dictionary:
            if key == 'TDirectory':
                # value is a dictionary whose keys are TDirectory names which themselves are keys to further dicts
                pychain = rt.TChain(tree)
    return None




def readROOT(inFile, tree, branches, method='single', tobject=None, directory=None, info=None):
    '''Read in root file'''

    if isinstance(branches,str):
        branches = [branches]
    print('The method is: {}'.format(method))
    if method is 'chain':
        # TChain Method
        #fPrefix = '/Volumes/Lantea/data/CUORE0/OfficialProcessed_v02.30/ds2049/'
        # lof = rt.TString(fPrefix + 'background_Blinded_ds2049.list")
        #lof = rt.TString('/Users/bwelliver/cuore/data/Blinded_200450_B.list')
        #lof = rt.TString('/Users/bwelliver/cuore/data/CUORE0/OfficialProcessed_v02.30/ds2070/background_Blinded_ds2070.list')
        #fileList = lof.Data()
        #lof = rt.TString('/Users/bwelliver/cuore/data/ds2160/physics_Production_ds2160.list')
        # Here inFile could be a specific list of files or a wildcard pattern
        tChain = rt.TChain(tree)
        print('The filelist is {0}'.format(inFile))
        if isinstance(inFile, list):
            for file in inFile:
                tChain.Add(file)
        else:
            tChain.Add(inFile)
        tDir = tChain.Get(directory) if directory else None
        tChain.SetBranchStatus('*', 0)
        # Need wildcard to set the whole thing active or you get segfaults
        for branch in branches:
            tChain.SetBranchStatus(branch, 1)
        # We should use an entrylist to get events. I should find out if there is a more Pythonic way to do this though...
        nEntries = tChain.GetEntries()
        #tChain.Draw(">>eventlist", "DAQ@PulseInfo.fChannelId==" + str(int(ch)))
        #from ROOT import eventlist
        getEv = tChain.GetEntry
        tTree = None
    elif method == 'single':
        # Single file mode
        tFile = rt.TFile.Open(inFile)
        tDir = tFile.Get(directory) if directory else None
        tTree = tFile.Get(tree) if tree else None
        print('TTree is {}'.format(tTree))
        if tTree is not None:
            tTree.SetBranchStatus('*', 0)
            # Need wildcard to set the whole thing active or you get segfaults
            for branch in branches:
                tTree.SetBranchStatus(branch, 1)
            nEntries = tTree.GetEntries()
            # tdir.Draw(">>eventlist", "DAQ@PulseInfo.fChannelId==" + str(int(ch)))
            # from ROOT import eventlist
            getEv = tTree.GetEntry

    # Here on out we are invariant to single file or chain mode
    # Grab any info from the directory specified.
    if method == 'chain':
        obj = tChain
    elif tTree is not None:
        obj = tTree
    else:
        obj = None
    infoList = None
    if tDir is not None:
        keys = tDir.GetListOfKeys()
        keyList = [key.GetName() for key in keys]
        keyList.sort()
        # Check all members of info are in keyList
        validInfo = list(set(info) & set(keyList)) if info else keyList
        if info != validInfo:
            print('The following items are not found in the ROOT file: {0}'.format(list( set(info) - set(validInfo) )))
        if len(validInfo) > 0:
            infoList = [tDir.Get(key).GetTitle() for key in validInfo]
        else:
            infoList = None
        del tDir
    # Beaware that chain mode can result in file sizes that are super-big
    # nEvent_list = eventlist.GetN()
    if obj is not None:
        print('Starting ROOT entry grabbing. There are {0} entries'.format(nEntries))
        nTen = np.floor(nEntries/10)
        npData = {}
        vectorDict = {}
        for branch in branches:
            tBranch = obj.GetBranch(branch)
            if tBranch.GetClassName() == 'vector<double>':
                # Create std vector double. It is SUPER important that we pass the ADDRESS OF THE POINTER TO THE VECTOR AND NOT THE ADDRESS OF THE VECTOR!!!!!
                # NOTE: If you just make var = rt.std.vector('double')() and the address of that in each loop, python WILL point to the same vector each time
                # The net result is a frustrating time trying to figure out WHY your vectors are the same. >:(
                vectorDict[branch] = rt.std.vector('double')()
                #pt = rt.std.vector('double')()
                obj.SetBranchAddress(branch, rt.AddressOf(vectorDict[branch]))
                npData[branch] = {}
            else:
                npData[branch] = np.zeros(nEntries)
        for entry in range(nEntries):
            getEv(entry)
            for branch in branches:
                if isinstance(npData[branch], dict):
                    npData[branch][entry] = np.array(vectorDict[branch])
                else:
                    data = getattr(obj, branch)
                    npData[branch][entry] = data
#                data = getattr(obj, branch)
#                # data could be a scalar or it could be an array
#                if isinstance(data, rt.vector('double')):
#                    # Here npData is a dictionary with key branch and value dictionary
#                    # The subdictionary has key entry and value array
#                    # It is vitally important that the ORDER be preserved! Use an ordered dict
#                    npData[branch][entry] = np.array(pt)
#                else:
#                    npData[branch][entry] = data
            # Print a notification every N events
            if entry%nTen == 0:
                print('Grabbing entry Number {0} ({1} %)'.format(entry, round(100*entry/nEntries,2)))
    elif tobject is not None:
        # For now this is to load a TVector object to a numpy vector
        tObject = tFile.Get(tobject)
        if tObject.ClassName() == 'TVectorT<double>':
            npData = {tobject: np.asarray(tObject)}
    # Destroy chain and other things
    if tTree is not None:
        del getEv
        del obj
    if method is 'chain':
        del tChain
    else:
        del tTree, tFile
    return {'data': npData, 'info': infoList}

