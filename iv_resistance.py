class TESResistance:
    
    def __init__(self):
        self.left = Resistance()
        self.right = Resistance()
        self.parasitic = Resistance()
        self.sc = Resistance()
    def __repr__(self):
        '''Return string representation'''
        s = 'Left Branch:\n' + str(self.left)
        s += '\n' + 'Right Branch:\n' + str(self.right)
        s += '\n' + 'Parasitic:\n' + str(self.parasitic)
        s += '\n' + 'Superconducting:\n' + str(self.sc)
        return(s)


class Resistance:
    
    def __init__(self):
        self.value = None
        self.rms = None
    def set_values(self, value=None, rms=None):
        self.value = value
        self.rms = rms
    def __repr__(self):
        '''Return string representation'''
        s = '\t' + 'Value:\t' + str(self.value)
        s += '\n' + '\t' + 'RMS:\t' + str(self.rms)
        return(s)
