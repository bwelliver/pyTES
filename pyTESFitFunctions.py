import numpy as np
from numpy import square as pow2

def lin_sq(x, m, b):
    '''Get the output for a linear response to an input'''
    y = m*x + b
    return y


def quad_sq(x, a, b, c):
    '''Get the output for a quadratic response to an input'''
    y = a*x*x + b*x + c
    return y


def exp_tc(T, C, D, B, A):
    '''Alternative R vs T
    Here we have
    C -> Rn
    D -> Rp
    -B/A -> Tc
    In the old fit we hade (T-Tc)/Tw -> T/Tw - Tc/Tw
    We have here A*T + B --> A = 1/Tw and B = -Tc/Tw
    '''
    R = (C*np.exp(A*T + B)) / (1 + np.exp(A*T + B)) + D
    return R


def tanh_tc(T, Rn, Rp, Tc, Tw):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance
    Rp is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data
    Usually the following is true:
        When T >> Tc: R = Rn + Rp
        When T << Tc: R = Rp
        When T = Tc: R = Rn/2 + Rp
    But Rp is already subtracted from our data so it should be 0ish

    '''
    #R = (Rn/2.0)*(tanh((T - Tc)/Tw) + 1) + Rp
    R = ((Rn - Rp)/2)*(np.tanh((T - Tc)/Tw) + 1) + Rp
    return R


def tanh_tc2(T, Rn, Tc, Tw):
    '''Get resistance values from fitting T to a tanh
    Rn is the normal resistance
    Rp is the superconducting resistance (parasitic)
    Tc is the critical temperature
    Tw is the width of the transition
    T is the actual temperature data
    Usually the following is true:
        When T >> Tc: R = Rn + Rp
        When T << Tc: R = Rp
        When T = Tc: R = Rn/2 + Rp
    But Rp is already subtracted from our data so it should be 0ish

    '''

    R = (Rn/2.0)*(np.tanh((T - Tc)/Tw) + 1)
    return R



def nll_error(params, P, P_rms, T, func):
    '''A fit for whatever function with y-errors'''
    k, n, Ttes = params
    if k <= 0 or n <= 0 or Ttes <= 0:
        return np.inf
    else:
        model = func(T, k, n, Ttes)
        lnl = nsum(((P - model)/P_rms)**2)
        return lnl


def tes_power_polynomial(T, *args):
    '''General TES power equation
    P = k*(T^n - Tb^n) - Pp
    '''
    n, k, Ttes, Pp = args
    P = k*(np.power(Ttes, n) - np.power(T, n)) - Pp
    #P = k*(power(Ttes, n) - power(T, n))
    return P


def tes_power_polynomial5(T, *args):
    '''General TES power equation assuming n = 5
    P = k*(T^n - Tb^n)
    '''
    k, Pp = args
    P = k*(np.power(61.5e-3, 5) - np.power(T, 5)) - Pp
    #P = k*(power(Ttes, n) - power(T, n))
    return P