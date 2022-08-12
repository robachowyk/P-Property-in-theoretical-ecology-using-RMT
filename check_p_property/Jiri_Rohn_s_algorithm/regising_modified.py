import absvaleqn
import beeckcond
import is_member
import janssonsheuristic
import normform
import numpy as np
import pmatrix_modified
import qzmatrix
import regsuffcondqz
import scipy
import scipy.optimize
import sgn
import singdetdesc
import vec2mat

import os
import sys
path = os.path.join("..","..", "simulations","utils") 
sys.path.append(path)
from general_utils import *

def regising(Ac,Delta):
    """
         dd=0.0:   singularity of the midpoint matrix Ac, 
         dd=0.1:   singularity via diagonal condition,
         dd=0.2:   singularity via steepest determinant descent,
         dd=0.3:   singularity as a by-product of 0.7,
         dd=0.4:   singularity via the main algorithm,
         dd=0.5:   regularity via Beeck's condition,
         dd=0.6:   regularity via symmetrization,
         dd=0.7:   regularity via two Qz-matrices,
         dd=0.8:   regularity via the main algorithm.
         
         dd=0.991: regularity via positive definiteness,
         dd=0.992: regularity via singular values.
         These two last criteria has been added by Kayané Robach based on:
         'A Manual of Results on Interval Linear Problems' from Jiri Rohn.
    """

    n = Ac.shape[0]
    dd = 0
    if np.linalg.matrix_rank(Ac) < n:
        S = Ac.copy()
        S = normform.normform(Ac,Delta,S)
        dd = 0.0
        return S, dd
    Aci = np.linalg.inv(Ac)
    D = Delta @ np.abs(Aci)
    dg = np.max(np.diag(D))
    J = np.argmax(np.diag(D))
    j = np.min(J)
    if dg >= 1:
        x = Aci[:,j]
        S = vec2mat.vec2mat(Ac,Delta,x)
        S = normform.normform(Ac,Delta,S)
        dd = 0.1
        return S, dd
    if beeckcond.beeckcond(Ac,Delta)<1:
        S = []
        dd = 0.5
        return S, dd
    S = singdetdesc.singdetdesc(Ac,Delta,0)
    if len(S)!=0:
        S = normform.normform(Ac,Delta,S)
        dd = 0.2
        return S, dd
    AA = Ac.conj().T @ Ac
    DD = Delta.conj().T @ Delta
    if (np.linalg.matrix_rank(AA) == n) and (beeckcond.beeckcond(AA,DD) < 1):
        S = []
        dd = 0.6
        return S, dd
    if (np.linalg.eigvals(AA - np.linalg.norm(DD) * np.eye(n)) > 0).all():
        dd = 0.6
        S = []
        return S, dd
    if np.max(np.linalg.eigvals(DD)) < np.min(np.linalg.eigvals(AA)):
        dd = 0.6
        S = []
        return S, dd
    if (np.linalg.eigvals(DD - AA) > 0).all():
        dd = 0.6
        S = [0, 0]
        return S, dd
    if np.max(np.linalg.eigvals(AA)) <= np.min(np.linalg.eigvals(DD)):
        dd = 0.6
        S = [0, 0]
        return S, dd
    rs, S = regsuffcondqz.regsuffcondqz(Ac,Delta)
    if rs == 0:
        S = normform.normform(Ac,Delta,S)
        dd = 0.3
        return S, dd
    if rs == 1:
        dd = 0.7
        return S, dd
    bc = janssonsheuristic.janssonsheuristic(Aci)
    eps = np.finfo(np.float64).eps
    ep = np.max([n,10]) * np.max([np.linalg.norm(Ac-Delta,ord=1), np.linalg.norm(Ac+Delta,ord=1), np.linalg.norm(bc,ord=1)]) * eps
    Z = [(sgn.sgn(Aci @ bc)).conj().T]
    D = []
    while len(Z)!=0:
        z = Z[-1].copy()
        Z = Z[:-1]
        D.append(z)
        Q, S = qzmatrix.qzmatrix(Ac,Delta,z)
        if len(S)!=0:
            S = normform.normform(Ac,Delta,S)
            dd = 0.4
            return S, dd
        xut = Q @ bc
        if (xut[z == 1] >= -ep).all():
            Q, S = qzmatrix.qzmatrix(Ac,Delta,-z)
            if len(S)!=0:
                S = normform.normform(Ac,Delta,S)
                dd = 0.4
                return S, dd
            xlt = Q @ bc
            if (xlt<=xut).all():
                for j in range(n):
                    zt = z.copy()
                    zt[j] = -zt[j]
                    if (xlt[j] * xut[j] <= ep) and (not( is_member.is_member(zt, Z) )) and (not( is_member.is_member(zt, D) )):
                        Z.append(zt)
            if len(Z) + len(D) > 20 * n**2:
                raise RuntimeError("In regising: program run has been stopped after reaching prescribed number of iterations")
    S = []
    dd = 0.8
    return S, dd