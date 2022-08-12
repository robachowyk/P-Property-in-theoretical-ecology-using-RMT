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
import singdetdesc
import vec2mat

def sgn(x):
    n = len(x)
    #     z = np.zeros(n)
    #     for j in range(n):
    #         if x[j] >= 0:
    #             z[j] = 1
    #         else:
    #             z[j] = -1
    z = np.ones(n)
    z[x < 0] = -1
    return z
