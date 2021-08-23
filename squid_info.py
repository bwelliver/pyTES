class InvalidChannelNumberException(Exception):
    '''Exception to raise if a channel number is invalid'''


class SQUIDParameters:
    '''Simple object class to store SQUID parameters'''

    def __init__(self, serial):
        self.squid = self.get_squid_parameters(serial)
        return None

    def get_squid_parameters(self, serial):
        '''Based on the serial number obtain SQUID parameters'''
        if serial == 'S0121':
            print("getting s121 parameters")
            self.serial = 'S0121'
            self.Li = 6e-9
            self.Mi = 1/26.062
            self.Mfb = 1/33.33730134 #1/33.16710559014 #1/33.33730134  #1/33.51162 #1/33.96774594  # 1/33.5132196
            self.Rfb = 10000.0
            self.Rsh = 21.0e-3
            self.Rbias = 10000.0
            self.Cbias = 100e-12
        elif serial == 'S0082':
            self.serial = 'S0082'
            self.Li = 6e-9
            self.Mi = 1/24.65
            self.Mfb = 1/32.62
            self.Rfb = 3e3
            self.Rsh = 20e-3  #20.8e-3  # 20e-3
            self.Rbias = 10000.0
            self.Cbias = 100e-12
        elif serial == 'S0094':
            self.serial = 'S0094'
            self.Li = 6e-9
            self.Mi = 1/23.993
            self.Mfb = 1/33.01006542
            self.Rfb = 3e3
            self.Rsh = 22.9e-3
            self.Rbias = 10000.0
            self.Cbias = 100e-12
        elif serial == 'S0204':
            self.serial = 'S0204'
            self.Li = 6e-9
            self.Mi = 1/25.135
            self.Mfb = 1/33.8403378  #1/32.62  #1/33.343662  #1/33.88826885424  #1/33.21197780928  #1/33.343662 #1/33.08377 #1/33.21197780928  #1/33.08377 #1/33.343662 #1/(33.5028)
            self.Rfb = 10e3
            self.Rsh = 20.2e-3
            self.Rbias = 10000.0
            self.Cbias = 100e-12
        elif serial == 'S0206':
            self.serial = 'S0206'
            self.Li = 6e-9
            self.Mi = 1/26.467
            self.Mfb = 1/33.40749307824  #1/33.39468924  #1/33.18313142268 #1/33.48441744234  #1/33.39468924 #1/33.48441744234 #1/33.69269472 #1/33.855534
            self.Rfb = 10e3
            self.Rsh = 21e-3
            self.Rbias = 10000  #394.092459 # Adjust 10kOhm bias for attenuator at digitizer with factor of 25.374756 10000.0
            self.Cbias = 100e-12
        else:
            raise InvalidChannelNumberException('Requested serial: {} is invalid. Please select S0094, S0121, S0204, S0206.'.format(serial))
        # Compute auxillary SQUID parameters based on ratios
        self.M = -self.Mi/self.Mfb
        self.Lfb = (self.M**2)*self.Li
        return None
