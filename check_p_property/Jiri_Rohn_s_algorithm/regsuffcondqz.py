import absvaleqn
import beeckcond
import is_member
import janssonsheuristic
import normform
import numpy as np
import pmatrix
import qzmatrix
import regising
import scipy
import scipy.optimize
import sgn
import singdetdesc
import vec2mat

def regsuffcondqz(Ac,Delta):
    n = Ac.shape[0]
    I = np.eye(n)
    e = np.ones((n,1))
    o = np.zeros((n,1))
    eps = np.finfo(np.float64).eps
    ep = np.max([n,10]) * np.max([np.linalg.norm(Ac-Delta,ord=1), np.linalg.norm(Ac+Delta,ord=1)])*eps
    Aci = np.linalg.inv(Ac)
    Q1, S = qzmatrix.qzmatrix(Ac,Delta,-e)
    if len(S)!=0:
        rs=0
        return rs, S
    Q2, S = qzmatrix.qzmatrix(Ac,Delta,e)
    if len(S)!=0:
        rs=0
        return rs, S
    A1 = np.concatenate((-Q1, e), axis=1)
    A2 = np.concatenate((-Aci, e), axis=1)
    A3 = np.concatenate((I, o), axis=1)
    A4 = np.concatenate((-I, o), axis=1)
    A = np.concatenate((A1, A2, A3, A4), axis=0)
    b = np.concatenate((o, o, e, e), axis=0)
    c = np.append(o, 1)
    x = scipy.optimize.linprog(-c, A_ub=A, b_ub=b, method="simplex").x
    if np.isinf(x[0]):
        rs = -1
        S = []
        return rs, S
    if x[-1] > ep:
        rs = 1
        S = []
        return rs, S
    rs = -1
    S = []
    return rs, S
