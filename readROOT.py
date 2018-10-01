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



def get_root_object(file_list, method, full_tree_name, branches):
    '''Function that returns a TTree or a TChain for the requested tree name based on the method requested'''
    if method == 'single':
        # This method probably can be deprecated since everything, even 1 file, can be loaded as a chain!
        root_object = None
    if method == 'chain':
        root_object = rt.TChain(full_tree_name)
        print('The filelist is {}'.format(file_list))
        if isinstance(file_list, list):
            for file in file_list:
                root_object.Add(file)
        else:
            root_object.Add(file_list)
        # Need wildcard to set the whole thing active or you get segfaults
        root_object.SetBranchStatus('*', 0)
        for branch in branches:
            root_object.SetBranchStatus(branch, 1)
    return root_object


def create_branch_arrays(root_object, branches, number_of_entries=None):
    '''Create storage dictionaries for the branches
    In order to be compatible with writeROOT we must ensure each TBranch array is a subdictionary with master dictionary key of 'TBranch'
    '''
    data = {'TBranch': {}}
    vector_data = {}
    if number_of_entries is None:
        number_of_entries = root_object.GetEntries()
    for branch in branches:
        tBranch = root_object.GetBranch(branch)
        if tBranch.GetClassName() == 'vector<double>':
            vector_data[branch] = rt.std.vector('double')()
            root_object.SetBranchAddress(branch, rt.AddressOf(vector_data[branch]))
            data['TBranch'][branch] = {}
        else:
            data['TBranch'][branch] = np.zeros(number_of_entries)
    return vector_data, data


def fill_branch_entry(root_object, entry, branches, vector_data, data):
    '''Function to fill a branch array with the requested entry value'''
    for branch in branches:
        if isinstance(data['TBranch'][branch], dict):
            data['TBranch'][branch][entry] = np.array(vector_data[branch])
        else:
            value = getattr(root_object, branch)
            data['TBranch'][branch][entry] = value
    return data


def get_entries(root_object, branches, vector_data, data):
    '''Function to actually get all the entries'''
    nEntries = root_object.GetEntries()
    nTen = np.floor(nEntries/10)
    # Create alias
    getEntry = root_object.GetEntry
    for entry in range(nEntries):
        getEntry(entry)
        data = fill_branch_entry(root_object, entry, branches, vector_data, data)
        # Print a notification every N events
        if entry%nTen == 0:
            print('Grabbing entry Number {} ({} %)'.format(entry, round(100*entry/nEntries, 2)))
    # All entries are looped over here so return the data object
    return data


def fetch_branches(root_object, branches):
    '''Function that will load branches based on specified root object and a list of branches
    We accomplish this by first creating storage arrays for the branches and then filling them
    root_object:    This is a rt.TChain or rt.TTree object
    branches:       A list of branch names to query from the root_object
    '''
    nEntries = root_object.GetEntries()
    print('Starting ROOT entry grabbing. There are {} entries'.format(nEntries))
    # Construct data storage dictionaries
    vector_data, data = create_branch_arrays(root_object, branches=branches, number_of_entries=nEntries)
    # Now let us loop over all the entries in the object and fill the branches
    data = get_entries(root_object, branches, vector_data, data)
    # We have looped over all the entries for this particular root object and filled the requested branches into a numpy array.
    return data


def walk_branches(file_list, method, full_tree_name, branches):
    '''Function that walks the branches
    In order to do this we must create the appropriate root object here (e.g., rt.TChain)
    '''
    root_object = get_root_object(file_list, method, full_tree_name, branches)
    data = fetch_branches(root_object, branches)
    return data


def walk_trees(file_list, method, directory_name, tree_dictionary):
    '''A helper function to walk through the tree and get all the branches
    '''
    # Here directory_name is the current directory we are in
    # tree_dictionary = {Tree1: list of branches, Tree2: list of branches, ...}
    data_dictionary = {}
    for key, value in tree_dictionary.items():
        # key = tree name
        # value = list of branches in the tree
        if directory_name is not None:
            full_tree_name = directory_name + '/' + key
        else:
            full_tree_name = key
        data = walk_branches(file_list, method, full_tree_name=full_tree_name, branches=value)
        data_dictionary[key] = data
    return data_dictionary


def walk_a_directory(file_list, method, directory_name, directory_contents):
    '''A function to walk the contents of a specified TDirectory'''
    # Here directory_contents is a dictionary whose keys are either 'TTree' or 'TBranch'
    data_dictionary = {}
    for key, value in directory_contents.items():
        if key == 'TTree':
            # directory_contents[key] = {TTree: {Tree1: listofbranches, Tree2: listofbranches, ...}}
            # Each tree must be accessed as 'DirectoryName/Tree1' for example
            data = walk_trees(file_list, method, directory_name=directory_name, tree_dictionary=value)
        elif key == 'TBranch':
            # directory_contents[key] = {TBranch: listofbranches}
            # Not sure how to load these
            data = None
        data_dictionary[key] = data
    return data_dictionary


def walk_directories(file_list, method, directory_dictionary):
    '''A helper function to walk through TDirectories
    The directory_dictionary has keys that are DirectoryNames
    '''
    # Our root_dictionary can specify a TDirectory that contains multiple trees. Each tree can contain multiple branches. When creating the TChain we must do it in the form
    # chain("TDirectory/TTree") so this function should iterate over all TDirectories and call the TTree walker.
    data_dictionary = {}
    for directory_name, value in directory_dictionary.items():
        # Key is a DirectoryName
        # Value is a dictionary whose keys can be ['TTree' or 'TBranch']
        data = walk_a_directory(file_list, method, directory_name=directory_name, directory_contents=value)
        # root_dictionary['TDirectory'][directory] = {'TTree: {Tree1: list of branches, ..., Tree2: list of branches }, TBranch: list of branches}
        data_dictionary[directory_name] = data
    return data_dictionary


def readROOT_new(input_files, root_dictionary, read_method='chain'):
    '''A function to make loading data from ROOT files into numpy arrays more flexible.
    This function reads from an input root file
    The root_dictionary is a specially formatted dictionary of names, similar to the interface in writeROOT.
    root_dictionary = {'TDirectory': {DirectoryName1: {'TTree': {Tree1: [Branch1, Branch2, Branch3 ...], 'TBranch': [BranchA, BranchB, ...]} } }, 'TTree': {...} }
    
    As an example we can get a list of branch names in Dir/TreeA as follows:
    branches = root_dictionary['TDirectory'][Dir]['TTree'][TreeA]
    
    read_method defines if we are opening the TFile directly (single, deprecated) or using a TChain (chain).
    
    '''
    
    # The course of action depends if we are chaining or not
    # If we have a TChain note that these chain together *TREES* across files.
    # The standard format is to call the chain as "dirName/treeName"
    # Therefore we must iterate through as follows:
    # iterate_tdir --> iterate_ttree --> construct chain names --> get associated branches
    # So let's recreate the hiearchy as need be then I guess
    
    # root_dictionary.keys() can be any of ['TDirectory', 'TTree', or 'TBranch']
    data_dictionary = {}
    for key, value in root_dictionary.items():
        if key == 'TDirectory':
            # Here value is a dictionary whose keys are DirectoryNames
            data = walk_directories(input_files, read_method, directory_dictionary=value)
        if key == 'TTree':
            # Here value is a dictionary whose keys are TreeNames
            data = walk_trees(input_files, read_method, directory_name=None, tree_dictionary=value)
        if key == 'TBranch':
            # Here value is a list of branch names
            # TODO: How do we handle these? Can they exist?
            data = None
        data_dictionary[key] = data
    return data_dictionary




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

