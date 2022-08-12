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
import vec2mat

def absvaleqn(A, B, b):
    n = len(b)
    I = np.eye(n)
    eps = np.finfo(np.float64).eps
    ep = n * (np.max([np.linalg.norm(A, ord=np.inf), np.linalg.norm(B, ord=np.inf), np.linalg.norm(b, ord=np.inf)])) * eps
    x = []
    S = []
    iterr = 0
    if np.linalg.matrix_rank(A) < n:
        S = A.copy()
        return x, S
    x = np.linalg.solve(A, b)
    z = sgn.sgn(x)
    if np.linalg.matrix_rank(A + B @ np.diag(z.flatten())) < n:
        S = A + B @ np.diag(z.flatten())
        x = []
        return x, S
    x = np.linalg.solve(A + B @ np.diag(z.flatten()), b)
    C = - np.linalg.inv(A + B @ np.diag(z.flatten())) @ B
    X = np.zeros((n,n))
    r = np.zeros(n)
    while (z*x < -ep).any():
        k = np.where(z.flatten()*x.flatten() < -ep)[0][0]
        iterr += 1
        if 1 + 2*z[k]*C[k,k] <= 0:
            S = A + B @ (np.diag(z.flatten())+(1/C[k,k])*np.outer(I[:,k],I[k,:]))
            x = []
            return x, S
        if ( (k < n-1) and (r[k] > r[k:]).all() ) or ( (k == n-1) and (r[k] > 0) ):
            x = x - X[:,k]
            z = sgn.sgn(x)
            ct = A @ x
            jm = np.abs(B) @ np.abs(x)
            #             y = np.zeros(n)
            #             for i in range(n):
            #                 if jm[i] > ep:
            #                     y[i] = ct[i]/jm[i]
            #                 else:
            #                     y[i] = 1
            y = np.ones(n)
            y[jm > ep] = ct / jm
            S = A - np.diag(y.flatten()) @ np.abs(B) @ np.diag(z.flatten())
            x = []
            return x, S
        X[:,k] = x.copy()
        r[k] = iterr
        z[k] = -z[k]
        alpha = 2 * z[k] / (1-2*z[k]*C[k,k])
        x = x + alpha * x[k] * C[:,k]
        C = C + alpha * np.outer(C[:,k],C[k,:])
    return x, S
