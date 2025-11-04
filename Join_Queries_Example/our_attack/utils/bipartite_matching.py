import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from math import log

def normalize_frequencies(frequencies):
    total_sum = sum(frequencies)
    if total_sum == 0:
        return frequencies  # If the sum is zero, no normalization is done
    return [f / total_sum for f in frequencies]

def bipartite_matching(aux_labels, aux_fre1, aux_fre2, obs_labels, obs_fre1, obs_fre2,type):
    assert len(aux_labels) == len(aux_fre1) == len(aux_fre2)
    assert len(obs_labels) == len(obs_fre1) == len(obs_fre2)

    # Normalize frequencies
    aux_fre1 = normalize_frequencies(aux_fre1)
    aux_fre2 = normalize_frequencies(aux_fre2)
    obs_fre1 = normalize_frequencies(obs_fre1)
    obs_fre2 = normalize_frequencies(obs_fre2)

    orig_obs_labels = obs_labels.copy()  
    orig_len = len(obs_labels)
    max_len = max(len(obs_labels), len(aux_labels))

    # Padding to equalize the lengths
    if len(obs_labels) < max_len:
        pad_len = max_len - len(obs_labels)
        obs_labels += [f'PAD_OBS_{i}' for i in range(pad_len)]
        obs_fre1 += [0.0] * pad_len
        obs_fre2 += [0.0] * pad_len

    if len(aux_labels) < max_len:
        pad_len = max_len - len(aux_labels)
        aux_labels += [f'PAD_AUX_{i}' for i in range(pad_len)]
        aux_fre1 += [sys.float_info.max] * pad_len
        aux_fre2 += [sys.float_info.max] * pad_len

    N = max_len
    matrix = []
    for i in range(N):
        row = []
        for j in range(len(aux_labels)):
            try:
                if type==0:
                    value = obs_fre2[i] * log(aux_fre2[j])
                else:
                    value = obs_fre1[i] * log(aux_fre1[j]) + obs_fre2[i] * log(aux_fre2[j])
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
