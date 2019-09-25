'''Class to store TES Resistance values from the left, right normal branches, the SC branch, and the parasitic resistance'''
import inspect

class TESResistance:
    '''Base class to store the different type of Resistance objects applicable to the IV scan'''

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
        return s

    def get_dict(self):
        '''Return a dictionary version of the class properties'''
        # The self.__dict__ could work or var(self) but let's be sure
        property_dict = {}
        names = dir(self)
        for name in names:
            if not name.startswith('__'):
                # Only get and keep a name if it is not a method or function
                # and if it has a form of _name check if name exists...if it does skip
                if not (name.startswith('_') and name[1:] in names):
                    value = getattr(self, name)
                    if not inspect.ismethod(value) and not inspect.isfunction(value):
                        property_dict[name] = value
        # Next handle left, right, sc and parasitic since they are Resistance objects
        for name in ['left', 'right', 'sc', 'parasitic']:
            property_dict[name] = property_dict[name].get_dict()
        return property_dict


class Resistance:
    '''Class to store the values of the TES resistances as well as the error on these values'''

    def __init__(self):
        self.value = None
        self.rms = None

    def set_values(self, value=None, rms=None):
        '''Method to set the values for the Resistance properties'''
        self.value = value
        self.rms = rms

    def __repr__(self):
        '''Return string representation'''
        s = '\t' + 'Value:\t' + str(self.value)
        s += '\n' + '\t' + 'RMS:\t' + str(self.rms)
        return s

    def get_dict(self):
        '''Return a dictionary version of the class properties'''
        # The self.__dict__ could work or var(self) but let's be sure
        property_dict = {}
        names = dir(self)
        for name in names:
            if not name.startswith('__'):
                # Only get and keep a name if it is not a method or function
                # and if it has a form of _name check if name exists...if it does skip
                if not (name.startswith('_') and name[1:] in names):
                    value = getattr(self, name)
                    if not inspect.ismethod(value) and not inspect.isfunction(value):
                        property_dict[name] = value
        return property_dict
