'''Class for storing IV fit results for left, right and sc branches'''

import inspect


class FitParameters:
    '''FitParameters is a general class for storing left, right normal branch fit results
    and SC branch fit results.
    '''
    def __init__(self, fit_type=None):

        self.left = FitResult()
        self.right = FitResult()
        self.sc = FitResult()
        self.normal = FitResult()

    def __repr__(self):
        '''return string representation.'''
        s = 'Left Branch:\n' + str(self.left)
        s = s + '\n' + 'Right Branch:\n' + str(self.right)
        s = s + '\n' + 'Superconducting:\n' + str(self.sc)
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
        # Next handle left, right, sc since they are FitResult objects
        for name in ['left', 'right', 'sc']:
            property_dict[name] = property_dict[name].get_dict()
        return property_dict


class FitResult:
    '''FitResult is a general class for storing the actual result and the error on the results'''
    def __init__(self):
        self.result = None
        self.error = None

    def set_values(self, result=None, error=None):
        '''Set the values for the properties of the FitResult'''
        self.result = result
        self.error = error

    def __repr__(self):
        ''''return string representation.'''
        s = '\t' + 'Result:\t' + str(self.result)
        s = s + '\n' + '\t' + 'Error:\t' + str(self.error)
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
