class SQUIDParameters:
    '''Simple object class to store SQUID parameters'''
    
    def __init__(self, channel):
        self.squid = self.get_squid_parameters(channel)
        return None
    
    def get_squid_parameters(self, channel):
        '''Based on the channel number obtain SQUID parameters'''
        if channel == 2:
            self.serial = 'S0121'
            self.Li = 6e-9
            self.Mi = 1/26.062
            self.Mfb = 1/33.488
            self.Rfb = 10000.0
            self.Rsh = 21.0e-3
            self.Rbias = 10000.0
            self.Cbias = 100e-12
        elif channel == 3:
            self.serial = 'S0094'
            self.Li = 6e-9
            self.Mi = 1/23.99
            self.Mfb = 1/32.9
            self.Rfb = 10000.0
            self.Rsh = 22.8e-3
            self.Rbias = 10000.0
            self.Cbias = 100e-12
        else:
            raise InvalidChannelNumberException('Requested channel: {} is invalid. Please select 2 or 3'.format(channel))
        # Compute auxillary SQUID parameters based on ratios
        self.M = -self.Mi/self.Mfb
        self.Lfb = (self.M**2)*self.Li
        return None

