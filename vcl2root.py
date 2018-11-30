from os.path import isabs
from os.path import dirname
import argparse
import struct
import ROOT as rt
from array import array
import csv
# Let's try to parse the binary vcl file format. It has some junk header stuff until byte 6144
# This may differ in later versions of vcl...a hex editor can help you here


def bytes_from_file(filename, chunksize, offset=0):
    with open(filename, mode='rb') as f:
        f.seek(offset)
        chunk = f.read(chunksize)
        print(chunk)
        if chunk:
            return struct.unpack('<' + 's'*32 ,chunk)
    return None


def all_bytes_from_file(filename):
    with open(filename, mode='rb') as f:
        return f.read()
    return None


def write_to_root(outFile, trees, branches, byteFile, offset, dataType, dataSize, endian):
    '''Function to write byte file to a root file'''
    print('Writing ROOT file...')
    nCol = len(headers)
    bSize = len(byteFile)
    nRow = int( (bSize - offset)/nCol/dataSize )

    if isinstance(trees, str):
        trees = [trees]
    if isinstance(branches,str):
        branches = [branches]

    # First create and open the root file
    tFile = rt.TFile(outFile, "RECREATE")

    # Now create dir with info inside it...change these as you see fit
    tDir = rt.TDirectoryFile("FileInfo", "Information about the file")
    tDir.Add(rt.TNamed('ProgName', 'vcl2root.py'))
    tDir.Add(rt.TNamed('ProgVersion', '0.5.0'))
    tDir.Add(rt.TNamed('CreatedOn', 'vcl2root.py'))
    tDir.Add(rt.TNamed('CreatedOnUnixTime', 'vcl2root.py'))
    tDir.Add(rt.TNamed('CreatedOnServer', 'vcl2root.py'))
    tDir.Add(rt.TNamed('CreatedByUser', 'vcl2root.py'))
    tDir.Add(rt.TNamed('CSVFile', 'vcl2root.py'))
    tDir.Add(rt.TNamed('WindowsFile', 'vcl2root.py'))
    
    # Write the directory to the file
    tDir.Write()
    del tDir
    # Next we create and fill each tree one at a time
    for tree in trees:
        tTree = rt.TTree(tree, tree)
        # Next we loop over branches and create an address for them
        dloc = {}
        for branch in branches:
            dloc[branch] = array('d', [0.0])
            bName = rt.TString(branch + '/D')
            tTree.Branch(branch, dloc[branch], bName.Data() )
        # Now loop over the events and fill the branches up.
        # The binary file makes it more natural to loop over events per branch
        print('There are {0} events'.format(nRow))
        for event in range(nRow):
            for branch in branches:
                #print('Branch is {0}'.format(branch))
                dloc[branch][0] = struct.unpack(endian + dataType, byteFile[offset:offset+dataSize])[0]
                offset = offset + dataSize
            tTree.Fill()
        tTree.Write()
        del tTree
    tFile.cd()
    del tFile
    return None


def write_to_csv(outFile, headers, byteFile, offset, dataType, dataSize, endian):
    print('Writing csv file')
    nCol = len(headers)
    bSize = len(byteFile)
    data = []
    dataDict = {}
    # Write this all to a csv file
    with open(outFile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        while offset < bSize:
            data = []
            for col in range(nCol):
                data.append( struct.unpack(endian + dataType, byteFile[offset:offset+dataSize])[0] )
                dataDict[headers[col]] = data[col]
                offset = offset + dataSize
            writer.writerow(dataDict)
    return None


def get_header(byte_file, header_offset):
    '''Takes a bunch of bytes and parses them for header information'''
    headers = []
    offset = header_offset
    while offset < data_offset:
        line = byteFile[offset:offset+headerSize]
        if line[0:1] == b'\xff':
            offset = offset + headerSize
            continue
        line = str(line, 'utf-8')
        line = line.strip('\x00')
        # tidy up for ROOT and CSV
        if line is not '':
            # Turn E/P Cal into EP Cal
            line = line.replace('/','')
            # For units replace spaces with _
            line = line.replace(' t(s)', '_t(s)')
            line = line.replace(' T(K)', '_T(K)')
            line = line.replace(' R(Ohm)', '_R(Ohm)')
            # Now replace spaces with no spaces and remove parenthesis
            line = line.replace(' ', '')
            line = line.replace('(', '_')
            line = line.replace(')', '')
            # Add temperature unit where needed
            if line in ['InputWaterTemp', 'OutputWaterTemp', 'HeliumTemp', 'OilTemp']:
                line = line + '_C'
            headers.append( line )
        # Next adjust offset
        offset = offset + headerSize
    return headers
    

def vclparser():
    '''Main function that will parse a specified VCL file into something else'''
    # Now the hard part. How do we know how many bytes to read in?
    # It turns out yes, the VCL's have a pretty generic structure
    # Different generations *may* have different offsets or lines...change each as needed!!!
    # From the start of the file, the headers start at byte 6144 and are 32 bytes each.
    # The header information ends with a bunch of 0xFF bytes
    # The data starts always at byte offset 12288
    # The data is stored as doubles (so 8 bytes) and in little-endian style (Windows)
    # There are always a full line with nColumns = nHeaders that are not whitespace
    header_offset = 6144
    headerSize = 32
    data_offset = 12288
    dataType = 'd'
    dataSize = 8
    # Windows is little-endian: <
    endian = '<'
    # Read the entire binary file to memory
    byteFile = all_bytes_from_file(inFile)
    data = get_header(byteFile, header_offset)
    # Get the headers starting at offset. They are 32 bytes each and stop at byte 12288
    offset = header_offset
    headers = []
    
    # Now we have a list of headers. Starting at data_offset we have little-endian encoded doubles.
    # There will be len(headers) of these before the next line. Let's see.
    nRow = (len(byteFile) - data_offset)/len(headers)/dataSize
    if outType == 'root':
        trees = 'FridgeLogs'
        write_to_root(outFile, trees, headers, byteFile, data_offset, dataType, dataSize, endian)
    if outType == 'csv':
        write_to_csv(outFile, headers, byteFile, data_offset, dataType, dataSize, endian)
    if outType == 'both':
        write_to_csv(outFile + '.csv', headers, byteFile, data_offset, dataType, dataSize, endian)
        trees = 'FridgeLogs'
        write_to_root(outFile + '.root', trees, headers, byteFile, data_offset, dataType, dataSize, endian)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--inputVCLFile', help='Specify the full path of the VCL file you wish to convert')
    parser.add_argument('-o', '--outputFile', help='Specify output file. If not a full path, it will be output in the same directory as the input file')
    parser.add_argument('-t', '--outputType', default='csv', help='Specify the output file type, csv (default) or root')
    args = parser.parse_args()
    inFile = args.inputVCLFile
    outFile = args.outputFile
    outType = args.outputType
    if outType not in ['csv', 'root', 'both']:
        print('Output file type must be csv or root, or "both". Setting to both')
        outType = 'both'
    if outType != 'both':
        print('outType is {0}'.format(outType))
        outFile = outFile + '.' + outType
    if not isabs(outFile):
        outFile = dirname(inFile) + '/' + outFile
    print('The output file type is {0} and the output file is {1}'.format(outType, outFile))
    vcldata = vclparser()
    print('All done!')