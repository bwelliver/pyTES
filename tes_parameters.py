'''Module to return TES specific quantities given input i-v data'''


def compute_iTES(vout, r_fb, m):
    '''Compute the TES current from iv data
    iTES = vOut/(M*Rfb)
    '''
    iTES = vout/(m*r_fb)
    return iTES


def compute_rTES(ibias, ites, r_sh, r_p):
    '''Compute rTES directly from currents
    # 1. ish = ibias - iTES
    # 2. ites*rp + ites*rtes = ish*rsh
    # 3. rtes = (ish*rsh - ites*rp)/ites
    # 4. rtes = (ibias*rsh - ites*rsh - ites*rp)/ites
    # 5. rtes = (ibias/ites)*rsh - rsh - rp
    Note this implies Rtes = Rsh*(ibias/ites) - Rl, and we note also
    that Rsh*iBias ~ vTh
    '''
    rtes = (ibias/ites)*r_sh - r_sh - r_p
    return rtes


def compute_vTES(ibias, ites, r_sh, r_p):
    '''Computation of vTES
    This can be performed either via vTES = iTES*rTES
    or through noting that iSh*rSh = iTES*(rTES+Rp)
    This gives:
        (iBias - iTES)*Rsh = iTES*rTES + iTES*Rp
        (iBias - iTES)*Rsh - iTES*Rp = vTES
        iBias*Rsh - iTES*(Rl) = vTES
    If we use the rTES equation in the first place:
        vTES = iTES*(iBias/iTES)*Rsh - iTES*Rsh - iTES*Rp
        vTES = iBias*Rsh - iTES*(Rl) so the two are equivalent
    '''
    vtes = ibias*r_sh - ites*(r_sh + r_p)
    return vtes


def compute_pTES(ites, vtes):
    '''Computes the power through the TES
    Simpy put this is iTES*vTES or:
        pTES = iTES*iBias*Rsh - iTES^2 * (Rl)
    '''
    ptes = ites*vtes
    return ptes
