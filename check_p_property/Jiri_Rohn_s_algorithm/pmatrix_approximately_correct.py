import absvaleqn
import beeckcond
import is_member
import janssonsheuristic
import normform
import numpy as np
import qzmatrix
import regising_approximately_correct
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

def pmatrix(A):
    n = A.shape[0]
    I = np.eye(n)
    assert np.linalg.matrix_rank(A - I) == n, "[input matrix] - [identity matrix] is singular, checking p-property is not possible with this algorithm"
    B = np.linalg.inv(A-I) @ (A+I)
    S, dd = regising_approximately_correct.regising(B, I)
    return len(S)==0