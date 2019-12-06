'''Input arguments container for pyIV module'''


class InputArguments:
    '''Class to store input arguments for use with py_iv'''
    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.inputPath = ''
        self.outputPath = ''
        self.run = 0
        self.biasChannel = 0
        self.dataChannel = 0
        self.makeROOT = False
        self.readROOT = False
        self.readTESROOT = False
        self.plotTES = False
        self.newFormat = False
        self.pidLog = ''
        self.numberOfWindows = 1
        self.slewRate = 1
        self.squid = ''
        self.tzOffset = 0
        self.thermometer = 'EP'

    def set_from_args(self, args):
        '''Set InputArguments properties from argparse properties'''
        self.set_from_dict(vars(args))

    def set_from_dict(self, dictionary):
        '''Set InputArguments properties from dictionary properties'''
        for key, value in dictionary.items():
            if getattr(self, key, None) is not None:
                setattr(self, key, value)
            else:
                print('The requested argument attribute {} is not a defined property of class InputArguments'.format(key))
