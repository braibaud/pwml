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


