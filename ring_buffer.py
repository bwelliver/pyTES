'''A simple module that defines a Ring Buffer object with some useful methods.
The RingBuffer is a FIFO style and fills from left to right, so that oldest
entries are at the end of the buffer. Requires numpy.'''
import numpy as np
from numba.experimental import jitclass
from numba import int32, float32

spec = [('size_max', int32), ('_data', float32[:]), ('size', int32)]


@jitclass(spec)
class RingBuffer:
    '''Main RingBuffer class object. Initialization parameters can be specified
    as the max size, default storage values and storage types. The default value
    is 0.0 and type is float.'''
    def __init__(self, size_max, default_value=0, dtype=np.float32):
        '''initialization'''
        self.size_max = size_max
        self._data = np.full(size_max, default_value, dtype=np.float32)
        # self._data = np.empty(size_max, dtype=dtype)
        # self._data.fill(default_value)
        self.size = 0

    def append(self, value):
        '''append an element'''
        self._data = np.roll(self._data, 1)
        self._data[0] = value
        if self.size != self.size_max:
            self.size += 1
        # if self.size == self.size_max:
        #    self.__class__ = RingBufferFull

    def get_sum(self):
        '''sum of the current values'''
        return np.sum(self._data)

    def get_mean(self):
        '''mean of the current values.'''
        return np.mean(self._data)

    def get_nanmean(self):
        '''mean of the current values ignoring NaN values'''
        return np.nanmean(self._data)

    def get_med(self):
        '''median of the current values.'''
        return np.median(self._data)

    def get_std(self):
        '''std of the current values'''
        return np.std(self._data)

    def argmax(self):
        '''return index of first occurence of max value'''
        return np.argmax(self._data)

    def get_all(self):
        '''return a list of elements from the newest to the oldest
        (left to right'''
        return self._data

    def get_partial(self, min_index=0, max_index=-1):
        '''Get all data to a specified point'''
        return self.get_all()[min_index:max_index]

    def get_size(self):
        '''Returns the size of the RingBuffer'''
        return np.size(self._data)

    def insert_array(self, data, flipped=True):
        '''Adds contents of array to the buffer via append'''
        if flipped is True:
            for value in np.flip(data, 0):
                self.append(value)
        else:
            for value in data:
                self.append(value)

    def __getitem__(self, key):
        '''get element'''
        return self._data[key]

#    def __repr__(self):
#        '''return string representation'''
#        string_rep = self._data.__repr__()
#        string_rep = string_rep + '\t' + str(self.size)
#        string_rep = string_rep + '\t' + self.get_all()[::-1].__repr__()
#        string_rep = string_rep + '\t' + self.get_partial()[::-1].__repr__()
#        return string_rep


#class RingBufferFull(RingBuffer):
#    '''Sub-class to handle appending data when a RingBuffer is full'''
#    def append(self, value):
#        '''append an element when buffer is full'''
#        self._data = np.roll(self._data, 1)
#        self._data[0] = value
