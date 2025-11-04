
import os
import sys
from GetAnchors import GetAnchors 
from XColRecover import XColRecover
from collections import defaultdict

def resolve_conflicts(triplets, co_tau):
    grouped = defaultdict(list)
    for obs, aux, score in triplets:
        grouped[obs].append((aux, score))

    resolved = []
    for obs, candidates in grouped.items():
        # Extract candidates with confidence > co_tau and not equal to 1
        valid = [(aux, s) for aux, s in candidates if s > co_tau and s != 1.0]
        if valid:
            best_aux, best_score = max(valid, key=lambda x: x[1])
        else:
            # If no valid items, keep the one with the highest confidence
            best_aux, best_score = max(candidates, key=lambda x: x[1])
        resolved.append((obs, best_aux, best_score))
    return resolved


def keep_max_score(triplets):
    """
    For each obs_label, only keep the match with the highest confidence in aux_label.
    If an aux_label is mapped by multiple obs_labels, only keep the one with the highest confidence.
    Input: triplets is a list of (obs_label, aux_label, score) tuples
    Output: a list of deduplicated tuples
    """
    best = {}
    for a, b, s in triplets:
        if (a, b) not in best or s > best[(a, b)][2]:
            best[(a, b)] = (a, b, s)

    final = {}
    for (a, b), triple in best.items():
        if b not in final or triple[2] > final[b][2]:
            final[b] = triple
    return list(final.values())


def read_freq_file(filepath):
    """
    Read a frequency file (tab-separated) and return three columns: label, fre1, fre2.
    Each line format: label \t fre1 \t fre2
    """
    labels, f1, f2 = [], [], []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                labels.append(parts[0])
                f1.append(float(parts[1]))
                f2.append(float(parts[2]))
    return labels, f1, f2