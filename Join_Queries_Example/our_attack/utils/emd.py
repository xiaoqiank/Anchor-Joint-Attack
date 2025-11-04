import sys
from math import log
import numpy as np
from ot import emd
from ot import sinkhorn
from scipy.optimize import linear_sum_assignment
import os
from datetime import datetime

def normalize_frequencies(frequencies):
    total_sum = sum(frequencies)
    if total_sum == 0:
        return frequencies  # If the sum is zero, no normalization is done
    return [f / total_sum for f in frequencies]

def emd_joint_matching(aux_labels, aux_fre1, aux_fre2, obs_labels, obs_fre1, obs_fre2, year=None,reg=0.001):
    assert len(aux_labels) == len(aux_fre1) == len(aux_fre2)
    assert len(obs_labels) == len(obs_fre1) == len(obs_fre2)
    N_obs = len(obs_labels)
    N_aux = len(aux_labels)

    # Normalize frequencies
    aux_fre1 = normalize_frequencies(aux_fre1)
    aux_fre2 = normalize_frequencies(aux_fre2)
    obs_fre1 = normalize_frequencies(obs_fre1)
    obs_fre2 = normalize_frequencies(obs_fre2)

    cost_matrix = np.zeros((N_obs, N_aux))
    for i in range(N_obs):
        for j in range(N_aux):
            try:
                value = abs(obs_fre1[i] - aux_fre1[j]) + abs(obs_fre2[i] - aux_fre2[j])
            except ValueError:
                value = 1
            cost_matrix[i, j] = value  # EMD minimizes the cost, so this is correct

    # Create probability distributions for the observation and auxiliary data
    obs_dist = np.array([f1 + f2 for f1, f2 in zip(obs_fre1, obs_fre2)])
    aux_dist = np.array([f1 + f2 for f1, f2 in zip(aux_fre1, aux_fre2)])
    obs_dist = obs_dist / obs_dist.sum()
    aux_dist = aux_dist / aux_dist.sum()

    # Apply Sinkhorn algorithm to find the optimal transport plan
    T = sinkhorn(obs_dist, aux_dist, cost_matrix, reg=reg)
    #T = emd(obs_dist, aux_dist, cost_matrix)  # You could use EMD instead of Sinkhorn


    mapping = {}
    confidence_rows = []

    for i, obs_label in enumerate(obs_labels):
        row = T[i]
        nonzero_indices = np.nonzero(row)[0]
        if len(nonzero_indices) <= 1:
            confidence = 1.0
            j = nonzero_indices[0] if len(nonzero_indices) == 1 else 0
        else:
            sorted_indices = np.argsort(row)[::-1]
            j = sorted_indices[0]
            second_j = sorted_indices[1]
            if row[j] == row[second_j]:
                confidence = 1.0
            else:
                diff = row[j] - row[second_j]
                if diff == 0.0:
                    print(f"[DEBUG] obs_label = {obs_label} → T row = {T[i]}")
                    print(f"obs_label: {obs_label} confidence is 0, reason analysis:")
                    print(f"row: {row}")
                    print(f"sorted_indices: {sorted_indices}")
                    print(f"j = {j} (aux_label = {aux_labels[j]}), second_j = {second_j} (aux_label = {aux_labels[second_j]})")
                    print(f"row[j] = {row[j]}, row[second_j] = {row[second_j]}")
                    print(f"full T row: {T[i]}")
                confidence = diff if diff > 1e-6 else 1.0
        mapping[obs_label] = aux_labels[j]
        confidence_rows.append((obs_label, aux_labels[j], confidence))

    # Return a list of tuples with the mappings and their confidence values
    return confidence_rows

def emd_wrapper_matching(obs_labels, obs_fre1, obs_fre2, aux_labels, aux_fre1, aux_fre2, year=None,reg=0.001):
    confidence_rows = emd_joint_matching(
        aux_labels=aux_labels,
        aux_fre1=aux_fre1,
        aux_fre2=aux_fre2,
        obs_labels=obs_labels,
        obs_fre1=obs_fre1,
        obs_fre2=obs_fre2,
        year=year,
        reg=reg
    )
    return confidence_rows

def read_freq_file(path):
    labels, fre1, fre2 = [], [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                labels.append(parts[0])
                fre1.append(float(parts[1]))
                fre2.append(float(parts[2]))
    return labels, fre1, fre2

def main():
    base_path_aux = "dataset/frequency_weighted/crimes(ca)-crashes-beat"
    base_path_obs = "dataset/frequency_weighted/crimes(ca)-crashes-beat-20%"
    aux_year = 2018
    obs_years = [2019, 2020, 2021, 2022]

    aux_path = os.path.join(base_path_aux, f"Beat_freq_{aux_year}.txt")
    aux_labels, aux_fre1, aux_fre2 = read_freq_file(aux_path)

    for year in obs_years:
        obs_path = os.path.join(base_path_obs, f"Beat_freq_{year}.txt")
        obs_labels, obs_fre1, obs_fre2 = read_freq_file(obs_path)

        confidence_rows = emd_wrapper_matching(obs_labels, obs_fre1, obs_fre2, aux_labels, aux_fre1, aux_fre2, year=year,reg=0.001)
        correct = 0
        for obs_label, aux_label, _ in confidence_rows:
            #print(f"{obs_label} --> {aux_label}")
            if obs_label == aux_label:
                correct += 1
        acc = 100.0 * correct / len(confidence_rows) if confidence_rows else 0.0
        print(f"{acc:.2f}% ({correct}/{len(confidence_rows)})")

if __name__ == "__main__":
    main()
