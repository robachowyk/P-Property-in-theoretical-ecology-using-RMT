import absvaleqn
import beeckcond
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

def is_member(array, list_of_arrays):
    return (len(list_of_arrays)!=0) and (np.any(np.all(array == list_of_arrays, axis=1)))