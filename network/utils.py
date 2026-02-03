import numpy as np

def relu(x):
    calculated_value = (x + np.abs(x)) / 2
    return max(0,calculated_value)