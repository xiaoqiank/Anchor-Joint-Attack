#!/usr/bin/env python3
import os
import sys
from math import log
import numpy as np
from ot import sinkhorn # requires POT (Python Optimal Transport)
# from ot import emd  # optional if you want exact EMD; sinkhorn is regularized
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Sinkhorn did not converge*")


def normalize_to_prob(vec):
    a = np.array(vec, dtype=float)
    s = a.sum()
    if s <= 0:
        return None  # signal invalid distribution
    return a / s

def emd_joint_matching(obs_labels, obs_freq, aux_labels, aux_freq, reg):
    """
    Use Sinkhorn OT to compute transport matrix T between obs_dist and aux_dist.
    Return list of tuples: (obs_label, matched_aux_label, confidence)
    Confidence in [0,1], computed as difference between top and second mass in row (or row top if only one).
    """
    # Basic checks
    if len(obs_labels) == 0 or len(aux_labels) == 0:
        return []

    # Build cost matrix (here simple absolute difference of frequencies)
    # Note: cost matrix should reflect "distance" between supports; using abs diff is simple but you can customize.
    N_obs = len(obs_labels)
    N_aux = len(aux_labels)
    M = np.zeros((N_obs, N_aux), dtype=float)
    for i in range(N_obs):
        for j in range(N_aux):
            try:
                M[i, j] = abs(float(obs_freq[i]) - float(aux_freq[j]))
            except Exception:
                M[i, j] = 1.0

    # Normalize distributions to probabilities
    obs_dist = normalize_to_prob(obs_freq)
    aux_dist = normalize_to_prob(aux_freq)
    if obs_dist is None or aux_dist is None:
        # Can't form valid distributions
        return []

    # Sinkhorn expects arrays a (sum=1), b (sum=1), cost matrix M, reg
    # Ensure shapes are correct
    obs_dist = obs_dist.astype(float)
    aux_dist = aux_dist.astype(float)

    # Compute transport plan (regularized OT)
    try:
        T = sinkhorn(obs_dist, aux_dist, M, reg)  # shape (N_obs, N_aux)
    except Exception as e:
        print(f"[ERROR] sinkhorn failed: {e}")
        return []

    # For each obs row, pick the best aux index and compute confidence
    results = []
    for i in range(N_obs):
        row = T[i]
        # If row all zeros (no transport mass), fallback to argmax on aux_dist
        if np.allclose(row, 0.0):
            j = int(np.argmax(aux_dist))
            confidence = 0.0
        else:
            # sort indices by mass descending
            sorted_idx = np.argsort(row)[::-1]
            j = int(sorted_idx[0])
            top = row[j]
            if len(sorted_idx) > 1:
                second = row[sorted_idx[1]]
                diff = top - second
                # normalize confidence to [0,1] by dividing by top if top>0 (so diff/top in [0,1])
                confidence = diff#float(diff / top) if top > 0 else 0.0
                # clamp
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 1.0  # only one non-zero
        results.append((obs_labels[i], aux_labels[j], confidence))

    return results


