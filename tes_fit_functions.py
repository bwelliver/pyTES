import numpy as np
from numpy import square as pow2
from scipy.special import erf

def lin_sq(x, m, b):
    '''Get the output for a linear response to an input'''
    y = m*x + b
    return y


def quad_sq(x, a, b, c):
    '''Get the output for a quadratic response to an input'''
    y = a*x*x + b*x + c
    return y


def exp_tc_paper(T, *args):
    '''Alternative R vs T used for the Tc paper
    Here we have
    C -> Rn
    D -> Rp
    -B/A -> Tc
    In the old fit we hade (T-Tc)/Tw -> T/Tw - Tc/Tw
    We have here A*T + B --> A = 1/Tw and B = -Tc/Tw
    '''
    C, D, B, A = args
    R = (C*np.exp(A*T + B)) / (1 + np.exp(A*T + B)) + D
    return R


def dexp_tc(T, *args):
    '''Functional form of dR/dT for exp_tc model.
    Here we have
    C -> Rn
    D -> Rp
    -B/A -> Tc
    In the old fit we hade (T-Tc)/Tw -> T/Tw - Tc/Tw
    We have here A*T + B --> A = 1/Tw and B = -Tc/Tw
    '''
    Rn, Rp, Tc, Tw = args
    # denom = np.power(np.exp(Tc/Tw) + np.exp(T/Tw), 2)
    # denom = np.power(np.exp(Tc/Tw)*(1 + np.exp((T - Tc)/Tw)),2)
    # dRdT = Rn*np.exp((T + Tc)/Tw)/(Tw*denom)
    # The above gets overflows in the exponential due to the scale between T and Tw.
    # Luckily there is an alternative closed form involving cosh
    dR = (Rn/(2*Tw)) * 1/(np.cosh((Tc-T)/Tw) + 1)
    #dR = Rn / (2*Tw*np.cosh((Tc - T)/Tw) + 2*Tw)
    return dR


def exp_tc(T, *args):
    '''Alternative R vs T used for the Tc paper
    Here we have
    C -> Rn
    D -> Rp
    -B/A -> Tc
    In the old fit we hade (T-Tc)/Tw -> T/Tw - Tc/Tw
    We have here A*T + B --> A = 1/Tw and B = -Tc/Tw
    '''
    Rn, Rp, Tc, Tw = args
    R = Rn*(np.exp((T-Tc)/Tw)/(1 + np.exp((T-Tc)/Tw))) + Rp
    return R


def dtanh_tc(T, *args):
    """Return the 1st derivative of the model tanh function with respect to T.

    The first derivative if a tanh is a sech^2 where sech = 1/cosh
    This gives dR/dT smoothly
    """
    Rn, Rp, Tc, Tw = args
    dR = (Rn/(2*Tw)) * 1/np.power(np.cosh((Tc-T)/Tw), 2)
    return dR


def tanh_tc(T, *args):
    """Resistance values from fitting T to a tanh
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
    """
    Rn, Rp, Tc, Tw = args
    R = (Rn/2.0)*(np.tanh((T - Tc)/Tw) + 1) + Rp
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

def tes_power_polynomial_args(T, k, n, Ttes, Pp):
    '''General TES power equation
    P = k*(T^n - Tb^n) - Pp
    '''
    # k, n, Ttes, Pp = args
    P = k*(np.power(Ttes, n) - np.power(T, n)) - Pp
    return P

def tes_power_polynomial(T, *args):
    '''General TES power equation
    P = k*(T^n - Tb^n) - Pp
    '''
    k, n, Ttes, Pp = args
    P = k*(np.power(Ttes, n) - np.power(T, n)) - Pp
    return P


def tes_power_polynomial_fixed(fixedArgs):
    '''General TES power equation fit with optional fixed args
    P = k*(T^n - Tb^n)
    '''
    k = fixedArgs.get('k', None)
    n = fixedArgs.get('n', None)
    Ttes = fixedArgs.get('Ttes', None)
    Pp = fixedArgs.get('Pp', None)
    fixedArgs = [k, n, Ttes, Pp]

    def tes_power_polynomial(T, *args):
        args = list(args)
        newargs = (args.pop(0) if item is None else item for item in fixedArgs)
        k, n, Ttes, Pp = newargs
        P = k*(np.power(Ttes, n) - np.power(T, n)) - Pp
        return P
    return tes_power_polynomial
