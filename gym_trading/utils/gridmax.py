from itertools import filterfalse, permutations
import numpy as np

def gridmax(input_shape, level):
    n_partitions = 2 ** level
    base = [0.0]*(input_shape - 1)
    for i in range(1, n_partitions+1):
        if (not (i & (i-1) == 0)) and i != 1:
            continue
        base += [1.0/i]*i
        sub_i = 1.0/i
        while sub_i < 1.0:
            base.append(1.0-sub_i)
            sub_i += 1.0/i
    possible_dists = np.array(list(filterfalse(lambda x: sum(x) != 1.0, permutations(base, input_shape))))
    possible_dists = np.unique(possible_dists.round(decimals=5), axis=0)
    return possible_dists