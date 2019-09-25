'''Errors for pyTES'''

class ArrayIsUnsortedException(Exception):
    '''Exception to raise if an array is unsorted'''


class InvalidChannelNumberException(Exception):
    '''Exception to raise if a channel number is invalid'''


class InvalidObjectTypeException(Exception):
    '''Exception to raise if the object passed into a function
    is not the correct type
    '''


class RequiredValueNotSetException(Exception):
    '''Exception to raise if a required value is missing'''