from os.path import isabs
from os.path import dirname
import argparse

import ROOT as rt
from array import array

def CopyDir(source, target):
    '''Recurse through the input root file'''
    # Loop over all entries of this directory
    keys = source.GetListOfKeys()
    nextKey = rt.TIter(keys)
    
    # Loop over the keys by getting the output of the iterator
    # If we need to reset the iterator call nextKey.Begin()
    for key in nextKey:
        # If it is a directory we should copy it and recurse inside
        # If it is a tree copy it via cloning it
        # If it is something else (object) Read it into a new object
        
        # Get class name of the key and see if it inherits from TDirectory or TTree
        className = key.GetClassName()
        dClass = rt.gROOT.GetClass(className)
        
        if dClass.InheritsFrom('TDirectory'):
            source.cd(key.GetName())
            target.mkdir(key.GetName())
            target.cd(key.GetName())
            CopyDir(source.GetDirectory(key.GetName()),target.GetDirectory(key.GetName()))
            target.cd()
        elif dClass.InheritsFrom('TTree'):
            tree = source.Get(key.GetName())
            target.cd()
            newTree = tree.CloneTree(-1)
            newTree.Write()
        else:
            source.cd()
            obj = key.ReadObj()
            target.cd()
            obj.Write()
            del obj
            
        target.SaveSelf(True)
        target.cd()
            


def CopyDir_Inject(source, target, injection, newFlag=False):
    '''Recurse through the input root file'''
    # Loop over all entries of this directory
    if newFlag is False:
        keys = source.GetListOfKeys()
        nextKey = rt.TIter(keys)
        
        # Get list of things to hunt for this round
        
        # Loop over the keys by getting the output of the iterator
        # If we need to reset the iterator call nextKey.Begin()
        for key in nextKey:
            # If it is a directory we should copy it and recurse inside
            # If it is a tree copy it via cloning it
            # If it is something else (object) Read it into a new object
            
            # Get class name of the key and see if it inherits from TDirectory or TTree
            
            # BRAD: Something is not right with the recursion and the popping of the dictionary.
            # Examine this and you probably will find your bug.
            className = key.GetClassName()
            dClass = rt.gROOT.GetClass(className)
            if dClass.InheritsFrom('TDirectory'):
                source.cd(key.GetName())
                target.mkdir(key.GetName())
                target.cd(key.GetName())
                
                if 'TDirectory' in injection:
                    dirjection = injection['TDirectory']
                    if key.GetName() in dirjection:
                        keyjection = dirjection[key.GetName()]
                        CopyDir_Inject(source.GetDirectory(key.GetName()),target.GetDirectory(key.GetName()), keyjection)
                        dirjection.pop(key.GetName())
                    else:
                        # the current key is not in dirjection
                        CopyDir_Inject(source.GetDirectory(key.GetName()),target.GetDirectory(key.GetName()), {})
                else:
                    # There is no TDirectory 
                    CopyDir_Inject(source.GetDirectory(key.GetName()),target.GetDirectory(key.GetName()), {})
                target.cd()
            elif dClass.InheritsFrom('TTree'):
                tree = source.Get(key.GetName())
                target.cd()
                if 'TTree' in injection:
                    treejection = injection['TTree']
                    if key.GetName() in treejection:
                        newTree = tree.CloneTree(0)
                        keyjection = treejection[key.GetName()]
                        # Need to inject new branches to the current key tree
                        dloc = {}
                        tbranch = {}
                        for branch in keyjection:
                            dloc[branch] = array('d', [0.])
                            tbranch[branch] = newTree.Branch(branch, dloc[branch], branch + '/D')
                        nEvents = tree.GetEntries()
                        for event in range(nEvents):
                            #print('The event is {0}'.format(event))
                            tree.GetEntry(event)
                            for branch in keyjection:
                                dloc[branch][0] = 0.0
                                dloc[branch][0] = keyjection[branch][event]
                                #tbranch[branch].Fill()
                            newTree.Fill()
                        treejection.pop(key.GetName())
                    else:
                        # current key is not in the list of things to inject into
                        newTree = tree.CloneTree(-1)
                    if not treejection:
                        injection.pop('TTree')
                else:
                    newTree = tree.CloneTree(-1)
                newTree.Write()
            else:
                source.cd()
                obj = key.ReadObj()
                target.cd()
                obj.Write()
                if 'TObject' in injection:
                    # We have some objects to write
                    objection = injection['TObject']
                    for objects in objection:
                        obj = objection[objects].Clone()
                        target.cd()
                        obj.Write()
                        del obj
                
            target.SaveSelf(True)
            target.cd()
        # Here we've looped over all keys now. If injection is not empty this represents NEW stuff to make
    if injection:
        # Injection is not empty so let's see what we got
        for key in injection:
            if key is 'TDirectory':
                dirjection = injection['TDirectory']
                for dirs in dirjection:
                    target.mkdir(dirs)
                    target.cd(dirs)
                    CopyDir_Inject(dirjection[dirs],target.GetDirectory(dirs), dirjection[dirs], True)
            elif key is 'TTree':
                # Build a new tree and then fill it's branches
                # Recall the dict should be treejection['Tree1']['branch1'] = values
                target.cd()
                treejection = injection['TTree']
                for tree in treejection:
                    branchjection = treejection[tree]
                    newTree = rt.TTree(tree,tree)
                    dloc = {}
                    for branch in branchjection:
                        nEntries = branchjection[branch].size
                        dloc[branch] = array('d', [0.0])
                        newTree.Branch(branch, dloc[branch], branch + '/D')
                    for event in range(nEntries):
                        for branch in branchjection:
                            dloc[branch][0] = branchjection[branch][event]
                        newTree.Fill()
                    newTree.Write()
                    del newTree
            elif key is 'TObject':
                # TObject
                target.cd()
                objection = injection['TObject']
                for objects in objection:
                    obj = objection[objects].Clone()
                    target.cd()
                    obj.Write()
                    del obj
            target.SaveSelf(True)
            target.cd()

def CopyFile(inFile, outFile):
    '''This function takes two arguments: the name of the input file to copy and the name of the file to output'''
    
    # Create the new root file to output
    
    oROOT = rt.TFile(outFile, 'recreate')
    
    iROOT = rt.TFile.Open(inFile)
    
    if not iROOT or iROOT.IsZombie():
        print('Cannot copy file {0}'.format(inFile))
    oROOT.cd()
    CopyDir(iROOT,oROOT)
    del iROOT
    oROOT.cd()
    del oROOT
    
    print('Created new file {0} from {1}'.format(outFile, inFile))
    

def CopyFile_Inject(inFile, outFile, injection):
    '''This function takes two arguments: the name of the input file to copy and the name of the file to output.
    Injection should be a dictionary whose primary keys indicate what type of object to start the insertion in (e.g., TDirectory, TTree, TObject)
    with a subdictionary indicating the particular name to inject something in, which itself is a dictionary of the name of the new thing with the values to insert.
    An example is given:
    injection = {'TTree': {'Tree1': {'NewBranch1': values, 'NewBranch2': values2}, 'Tree3': {'NewBranchA': values3}}}
    NOTES: If you are injecting a TObject, it must be a TObject class already. Branch values inside of TTrees can be numpy arrays
    '''

    # Create the new root file to output
    
    oROOT = rt.TFile(outFile, 'recreate')
    
    iROOT = rt.TFile.Open(inFile)
    
    if not iROOT or iROOT.IsZombie():
        print('Cannot copy file {0}'.format(inFile))
    oROOT.cd()
    CopyDir_Inject(iROOT,oROOT, injection)
    del iROOT
    oROOT.cd()
    del oROOT
    
    print('Created new file {0} from {1}'.format(outFile, inFile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', help='Specify the full path of the input file')
    parser.add_argument('-o', '--outputFile', help='Specify output file. If not a full path, it will be output in the same directory as the input file')    
    args = parser.parse_args()
    
    inFile = args.inputFile
    outFile = args.outputFile
    if not isabs(outFile):
        outFile = dirname(inFile) + '/' + outFile
    
    CopyFile(args.inputFile, outFile)