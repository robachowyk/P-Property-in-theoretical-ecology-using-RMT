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
import vec2mat

def singdetdesc(Ac,Delta, t):
    n = Ac.shape[0]
    Ad = Ac - Delta
    Ah = Ac + Delta
    if np.linalg.matrix_rank(Ac) < n:
        S = Ac.copy()
        return S
    if np.linalg.matrix_rank(Ad) < n:
        S = Ad.copy()
        return S
    if np.linalg.matrix_rank(Ah) < n:
        S = Ah.copy()
        return S
    if t==1:
        A = Ah.copy()
    else:
        A = Ac.copy()
    C = np.linalg.inv(A)
    beta = 0.5
    p= np.zeros(n)
    while beta < 0.95:
        Zd = (C.conj().T >= 0)
        Zh = (C.conj().T < 0)
        B = np.multiply(Zd, Ad) + np.multiply(Zh, Ah)
        beta = np.min(np.diag(B@C))
        J = np.argmin(np.diag(B@C))
        k = np.min(J)
        if np.abs(beta) >= 0.95:
            S = []
            return S
        if (1e-10 < beta) and (beta < 0.95):
            D = B-A
            C = C - (1./beta) * np.outer(C[:,k],(D[k,:] @ C))
            A[k,:] = B[k,:]
        if (0 < beta) and (beta <= 1e-10):
            S = []
            return S
        if beta <= 0:
            for i in range(n):
                p[i] = B[k,:i] @ C[:i,k] + A[k,i:] @ C[i:,k]
            if (p > 0).all():
                S = []
                return S
            m = np.where(p <= 0)[0][0]
            if np.abs(C[m,k]) <= 1e-10:
                S=[]
                return S
            A[k,:m-1] = B[k,:m-1]
            A[k,m] = -(B[k,:m-1] @ C[:m-1,k] + A[k,m:] @ C[m:,k]) / C[m,k]
            S = A.copy()
            return S 
    S = []
    return S