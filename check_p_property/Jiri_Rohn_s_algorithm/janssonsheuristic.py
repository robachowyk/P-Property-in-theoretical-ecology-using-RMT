import absvaleqn
import beeckcond
import is_member
import normform
import numpy as np
import pmatrix
import qzmatrix
import regising
import regsuffcondqz
import scipy
import scipy.optimize
import sgn
import singdetdesc
import vec2mat

def janssonsheuristic(Aci):
    n = Aci.shape[0]
    bc = np.ones(n)
    g = np.min(np.abs(Aci @ bc))
    for i in range(n):
        for j in range(n):
            bp = bc.copy()
            bp[j] = -bp[j]
            if np.min(np.abs(Aci @ bp)) > g:
                g = np.min(np.abs(Aci @ bp))
                bc = bp.copy()
    for i in range(n):
        for j in range(n):
            if j!=i:
                bp = bc.copy()
                bp[i] = -bp[i]
                bp[j] = -bp[j]
                if np.min(np.abs(Aci @ bp)) > g:
                    g = np.min(np.abs(Aci @ bp))
                    bc = bp.copy()
    return bc