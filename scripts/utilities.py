import os
import numpy as np

def get_dir():
    return os.getcwd()[:-8]

def combine_left_right(vl,bl,vr,br):
    b = np.clip(bl+br, 0.,1.)
    bl = np.repeat(np.expand_dims(bl,axis=2),axis=2,repeats=3)
    br = np.repeat(np.expand_dims(br,axis=2),axis=2,repeats=3)
    v_ = np.ones(vl.shape)*255.*(1-bl) + bl*vl
    v = v_*(1-br) + br*vr

    return v,b