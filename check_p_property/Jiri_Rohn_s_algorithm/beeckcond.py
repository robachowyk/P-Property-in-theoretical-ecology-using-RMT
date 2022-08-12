import absvaleqn
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

def beeckcond(Ac,Delta):
    return np.max(np.abs(np.linalg.eigvals(np.abs(np.linalg.inv(Ac)) @ Delta)))