import math as m
import pandas as pd
import numpy as np


def flatten_namevalue_pairs(pairs, separator):
    message = ''

    for name, value, format in pairs:
        message += '{0}: {1}{2}'.format(
            name,
            ('{0:'+format+'}').format(value),
            separator)
            
    return message[:-1*len(separator)]

def shift_array(ar, n):
    e = np.empty_like(ar)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = ar[:-n]
    else:
        e[n:] = np.nan
        e[:n] = ar[-n:]
    return e
