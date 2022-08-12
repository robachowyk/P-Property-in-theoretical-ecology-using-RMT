import absvaleqn
import beeckcond
import is_member
import janssonsheuristic
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

def vec2mat(Ac,Delta,x):
    n = len(x)
    eps = np.finfo(np.float64).eps
    ep = np.max([n, 100]) * np.max([np.linalg.norm(Ac-Delta,ord=np.inf), np.linalg.norm(Ac+Delta,ord=np.inf), np.linalg.norm(x,ord=np.inf)]) * eps
    ct = Ac @ x 
    jm = Delta @ np.abs(x)
    #     y = np.zeros(n)
    #     z = y.copy()
    #     for i in range(n):
    #         if jm[i] > ep:
    #             y[i] = ct[i]/jm[i]
    #         else:
    #             y[i] = 1
    #         if x[i] >= 0: 
    #             z[i] = 1
    #         else: 
    #             z[i] =- 1
    y = np.ones(n)
    y[jm > ep] = ct / jm
    z = sgn.sgn(x)
    S = Ac - np.diag(y.flatten()) @ Delta @ np.diag(z.flatten())
    return S
