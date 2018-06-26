import numpy as np
import ROOT as rt

def readROOT(inFile, tree, branches, method='single', directory=None, info=None):
    '''Read in root file'''

    if isinstance(branches,str):
        branches = [branches]

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
    elif method is 'single':
        # Single file mode
        tFile = rt.TFile.Open(inFile)
        tDir = tFile.Get(directory) if directory else None
        tTree = tFile.Get(tree)
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
    obj = tChain if method == 'chain' else tTree
    infoList = None
    if tDir:
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
    print('Starting ROOT entry grabbing. There are {0} entries'.format(nEntries))
    # nEvent_list = eventlist.GetN()
    nTen = np.floor(nEntries/10)
    npData = {}
    for branch in branches:
        tBranch = obj.GetBranch(branch)
        if tBranch.GetClassName() == 'vector<double>':
            npData[branch] = {}
        else:
            npData[branch] = np.zeros(nEntries)
    for entry in range(nEntries):
        getEv(entry)
        for branch in branches:
            data = getattr(obj, branch)
            # data could be a scalar or it could be an array
            if isinstance(data, rt.vector('double')):
                npData[branch][entry] = np.asarray(data)
            else:
                npData[branch][entry] = data
        # Print a notification every N events
        if entry%nTen == 0:
            print('Grabbing entry Number {0} ({1} %)'.format(entry, round(100*entry/nEntries,2)))
    # Destroy chain and other things
    del getEv
    del obj
    if method is 'chain':
        del tChain
    else:
        del tTree, tFile
    return {'data': npData, 'info': infoList}

