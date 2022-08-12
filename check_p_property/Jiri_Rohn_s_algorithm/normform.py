import absvaleqn
import beeckcond
import is_member
import janssonsheuristic
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

def normform(Ac,Delta,S):
    Snf = singdetdesc.singdetdesc(Ac,Delta,1)
    if len(Snf)!=0:
        S = Snf.copy()
    return S