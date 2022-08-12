import absvaleqn
import beeckcond
import is_member
import janssonsheuristic
import normform
import numpy as np
import pmatrix
import regising
import regsuffcondqz
import scipy
import scipy.optimize
import sgn
import singdetdesc
import vec2mat

def qzmatrix(Ac,Delta,z):
    n = Ac.shape[0]
    I = np.eye(n)
    Q = np.zeros((n,n))
    S = []
    for i in range(n):
        x, S = absvaleqn.absvaleqn(Ac.conj().T, -np.diag(z.flatten()) @ Delta.conj().T, I[:,i])
        if len(S)!=0:
            Q = []
            S = S.conj().T
            return Q, S
        Q[i,:] = x.copy().conj().T
    return Q, S
