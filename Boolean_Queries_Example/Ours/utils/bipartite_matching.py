import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from math import log
import pandas as pd


def normalize_frequencies(frequencies):
    total_sum = sum(frequencies)
    if total_sum == 0:
        return frequencies  # If the sum is zero, no normalization is done
    return [f / total_sum for f in frequencies]

def bipartite_matching(aux_labels, aux_fre, obs_labels, obs_fre):
    assert len(aux_labels) == len(aux_fre) 
    assert len(obs_labels) == len(obs_fre) 


    orig_obs_labels = obs_labels.copy()  
    orig_len = len(obs_labels)
    max_len = max(len(obs_labels), len(aux_labels))

    # Padding to equalize the lengths
    if len(obs_labels) < max_len:
        pad_len = max_len - len(obs_labels)
        obs_labels += [f'PAD_OBS_{i}' for i in range(pad_len)]
        obs_fre += [0.0] * pad_len

    if len(aux_labels) < max_len:
        pad_len = max_len - len(aux_labels)
        aux_labels += [f'PAD_AUX_{i}' for i in range(pad_len)]
        aux_fre += [sys.float_info.max] * pad_len


    N = max_len
    matrix = []
    for i in range(N):
        row = []
        for j in range(len(aux_labels)):
            try:
                value = obs_fre[i] * log(aux_fre[j])
            except ValueError:
                value = -1 * sys.float_info.max
            if np.isnan(value):
                value = -1 * sys.float_info.max
            row.append(value)
        matrix.append(row)

    matrix = np.array(matrix)
    rows, cols = linear_sum_assignment(cost_matrix=matrix, maximize=True)
    match_dict = {}
    for i, j in zip(rows, cols):
        if obs_labels[i] in orig_obs_labels:
            match_dict[obs_labels[i]] = aux_labels[j]

    # Return the list of (obs_label, aux_label) pairs as a 2-tuple list
    return [(label, match_dict.get(label, "_PAD_")) for label in orig_obs_labels]
