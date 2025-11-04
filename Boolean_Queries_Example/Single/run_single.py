#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
from math import log
import numpy as np
from ot import sinkhorn  # requires POT (Python Optimal Transport)
# from ot import emd     # optional if you want exact EMD; sinkhorn is regularized
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Sinkhorn did not converge*")


def normalize_frequencies(frequencies):
    total_sum = sum(frequencies)
    if total_sum == 0:
        return frequencies  # If the sum is zero, no normalization is done
    return [f / total_sum for f in frequencies]


def normalize_to_prob(vec):
    """
    Normalize a non-negative vector into a probability distribution;
    if the sum <= 0, returns None.
    """
    a = np.array(vec, dtype=float).reshape(-1)
    s = a.sum()
    if s <= 0:
        return None  # signal invalid distribution
    return a / s


def save(output_path, target_column, aux_year, obs_year, strategy, accuracy, row_recovery, count):
    """
    Function to append a result to a CSV file.
    If the file doesn't exist, it creates it and writes the header.
    Headers: column name, auxiliary year, target year, algorithm, accuracy, row recovery, ratio
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_exists = os.path.exists(output_path)
    need_header = (not file_exists) or os.path.getsize(output_path) == 0

    # Add row recovery column
    headers = ["Column Name", "Auxiliary Year", "Target Year", "Algorithm", "Accuracy", "Row Recovery", "Ratio"]

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(headers)
        writer.writerow([target_column, int(aux_year), int(obs_year), str(strategy), f"{float(accuracy):.6f}", f"{float(row_recovery):.6f}", count])


def emd_joint_matching(obs_labels, obs_freq, aux_labels, aux_freq, reg):
    """
    Use Sinkhorn (regularized EMD) to compute the transportation between the frequency distributions of obs and aux,
    and provide the top-1 matching for each obs_label with a simple confidence score.

    Returns: List[ (obs_label, aux_label, confidence) ]
    """
    if len(obs_labels) == 0 or len(aux_labels) == 0:
        return []

    # Convert to numpy arrays and check length consistency
    obs_freq = np.asarray(obs_freq, dtype=float).reshape(-1)
    aux_freq = np.asarray(aux_freq, dtype=float).reshape(-1)
    if len(obs_labels) != len(obs_freq) or len(aux_labels) != len(aux_freq):
        return []

    N_obs = len(obs_labels)
    N_aux = len(aux_labels)

    obs_prob = normalize_to_prob(obs_freq)
    aux_prob = normalize_to_prob(aux_freq)
    if obs_prob is None or aux_prob is None:
        return []

    # Cost matrix: L1 probability difference (absolute scalar)
    M = np.zeros((N_obs, N_aux), dtype=float)
    for i in range(N_obs):
        for j in range(N_aux):
            M[i, j] = abs(float(obs_prob[i]) - float(aux_prob[j]))

    try:
        # Sinkhorn requires a, b to be probability histograms (sum to 1)
        T = sinkhorn(obs_prob, aux_prob, M, reg)  # shape: (N_obs, N_aux)
    except Exception as e:
        print(f"[ERROR] Sinkhorn failed: {e}")
        return []

    results = []
    for i in range(N_obs):
        row = T[i]
        if np.allclose(row, 0.0):
            # Degeneration case: the row is all zeros, fallback to selecting the largest aux_prob entry
            j = int(np.argmax(aux_prob))
            confidence = 0.0
        else:
            # Take the column with the largest transport value as the match
            sorted_idx = np.argsort(row)[::-1]
            j = int(sorted_idx[0])
            top = float(row[j])
            if len(sorted_idx) > 1:
                second = float(row[sorted_idx[1]])
                diff = top - second
                # Clip the difference to [0,1] as "simple confidence"
                confidence = max(0.0, min(1.0, diff))
            else:
                # Only one non-zero, consider the confidence as 1
                confidence = 1.0

        results.append((obs_labels[i], aux_labels[j], confidence))

    return results


def bipartite_matching(aux_labels, aux_fre, obs_labels, obs_fre):
    """
    Weighted bipartite graph matching (Hungarian algorithm, maximize=True).
    Note: Here, the matrix is a "score" rather than a "cost," so use maximize=True.
    """
    # Ensure everything is in lists to avoid ndarray broadcasting issues
    aux_labels = list(aux_labels)
    obs_labels = list(obs_labels)
    aux_fre = list(np.asarray(aux_fre, dtype=float).reshape(-1))
    obs_fre = list(np.asarray(obs_fre, dtype=float).reshape(-1))

    assert len(aux_labels) == len(aux_fre)
    assert len(obs_labels) == len(obs_fre)

    orig_obs_labels = obs_labels.copy()
    max_len = max(len(obs_labels), len(aux_labels))

    # Padding to equalize the lengths
    if len(obs_labels) < max_len:
        pad_len = max_len - len(obs_labels)
        obs_labels.extend([f'PAD_OBS_{i}' for i in range(pad_len)])
        obs_fre.extend([0.0] * pad_len)

    if len(aux_labels) < max_len:
        pad_len = max_len - len(aux_labels)
        aux_labels.extend([f'PAD_AUX_{i}' for i in range(pad_len)])
        # Assign a very large value to PAD columns to make their log value large
        aux_fre.extend([sys.float_info.max] * pad_len)

    N = max_len
    matrix = []
    for i in range(N):
        row = []
        for j in range(len(aux_labels)):
            u = aux_fre[j]
            v = obs_fre[i]
            if u <= 0:
                value = -1 * sys.float_info.max  # Avoid log(0) or negative values
            else:
                value = v * log(u)
            if np.isnan(value) or np.isinf(value):
                value = -1 * sys.float_info.max
            row.append(value)
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)
    rows, cols = linear_sum_assignment(cost_matrix=matrix, maximize=True)

    match_dict = {}
    for i, j in zip(rows, cols):
        if obs_labels[i] in orig_obs_labels:
            match_dict[obs_labels[i]] = aux_labels[j]

    # Return the (obs_label, aux_label) list aligned with the original obs_labels
    return [(label, match_dict.get(label, "_PAD_")) for label in orig_obs_labels]


def read_freq_file(path):
    """
    Read a frequency file, automatically detecting the delimiter (comma or tab),
    and attempts to find two columns containing 'value' and 'freq' in their names.
    Returns: (labels: List[str], freqs: np.ndarray[float])
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]

        col_value, col_freq = None, None
        for c in df.columns:
            if col_value is None and "value" in c:
                col_value = c
            if col_freq is None and ("freq" in c or "frequency" in c):
                col_freq = c

        if col_value is None or col_freq is None:
            raise ValueError(f"Columns (Value, Frequency) not found in {path}")

        labels = df[col_value].astype(str).fillna("").tolist()
        freqs = pd.to_numeric(df[col_freq], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return labels, freqs

    except Exception as e:
        print(f"Failed to read file {path}: {e}")
        return [], np.array([], dtype=float)


def append_accuracy(result_path, column_name, aux_year, obs_year, correct, total):
    """
    Append accuracy results to a file.
    If the file doesn't exist, it writes the header:
    column_name,aux_year,obs_year,acc,ratio
    where acc is a decimal in [0,1], and ratio is 'correct/total'.
    """
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    header = "column_name,aux_year,obs_year,acc,ratio\n"
    line = f"{column_name},{aux_year},{obs_year},{(correct/total if total else 0.0):.6f},{correct}/{total}\n"

    write_header = not os.path.exists(result_path) or os.path.getsize(result_path) == 0
    with open(result_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(line)

# Original function: def single(target_column, obs_year, aux_year, input_dir, strategy, obs_file, aux_file, reg):
def single(target_column, input_dir, strategy, obs_file, obs_year, aux_file, aux_year, reg):
    """
    Returns (accuracy, row_recovery, count)

    row_recovery: sum of obs frequencies (from obs_file) for obs_labels that were
    correctly matched (obs_label == aux_label), divided by total sum of obs frequencies.
    """
    accuracy = 0.0
    count = ""
    row_recovery = 0.0

    # Read frequency files
    aux_labels, aux_freq = read_freq_file(aux_file)
    obs_labels, obs_freq = read_freq_file(obs_file)

    # Ensure obs_labels and obs_freq align
    obs_freq_list = list(obs_freq) if obs_freq is not None else []

    correct = 0
    total = 0

    if strategy == "emd":
        confidence_rows = emd_joint_matching(
            obs_labels=obs_labels,
            obs_freq=obs_freq,
            aux_labels=aux_labels,
            aux_freq=aux_freq,
            reg=reg
        )
        correct = sum(1 for o, a, _ in confidence_rows if o == a)
        total = len(confidence_rows)

        # Compute row-level sums for correctly-matched obs labels
        if total > 0 and len(obs_labels) == len(obs_freq_list):
            # Build a map from label to freq (sum frequencies for duplicate labels)
            label_to_freq = {}
            for lbl, fr in zip(obs_labels, obs_freq_list):
                label_to_freq.setdefault(lbl, 0.0)
                try:
                    label_to_freq[lbl] += float(fr)
                except Exception:
                    pass

            sum_all = sum(label_to_freq.values())
            sum_correct = 0.0
            for o, a, _ in confidence_rows:
                if o == a:
                    sum_correct += label_to_freq.get(o, 0.0)

            row_recovery = (sum_correct / sum_all) if sum_all > 0 else 0.0

        count = f"{correct}/{total}"
        accuracy = (correct / total) if total else 0.0

    elif strategy == "bi":
        val_match = bipartite_matching(
            obs_labels=obs_labels,
            obs_fre=obs_freq,
            aux_labels=aux_labels,
            aux_fre=aux_freq
        )
        correct = sum(1 for o, a in val_match if o == a)
        total = len(val_match)

        if total > 0 and len(obs_labels) == len(obs_freq_list):
            label_to_freq = {}
            for lbl, fr in zip(obs_labels, obs_freq_list):
                label_to_freq.setdefault(lbl, 0.0)
                try:
                    label_to_freq[lbl] += float(fr)
                except Exception:
                    pass

            sum_all = sum(label_to_freq.values())
            sum_correct = 0.0
            for o, a in val_match:
                if o == a:
                    sum_correct += label_to_freq.get(o, 0.0)

            row_recovery = (sum_correct / sum_all) if sum_all > 0 else 0.0

        count = f"{correct}/{total}"
        accuracy = (correct / total) if total else 0.0

    return accuracy, row_recovery, count


def main():
    input_dir = "/path/to/your/data/frequency"
    output_dir = "/path/to/your/output/results"
    reg = 0.0001
    aux_years = [i for i in range(2018, 2025)]
    obs_years = [j for j in range(2018, 2025)]
    attack_columns = ["Procedure", "Diagnosis"]
    strategies = ["emd", "bi"]

    for aux_year in aux_years:
        for i in range(0, 2):
            j = abs(i-1)
            target_column = attack_columns[i]
            aux_file = os.path.join(input_dir, f"{target_column}_freq_{aux_year}.txt")
            
            output_path = os.path.join(output_dir, f"{target_column}_aux{aux_year}.csv")
            for obs_year in obs_years:
                obs_file = os.path.join(input_dir, f"{target_column}_freq_{obs_year}.txt")
                for strategy in strategies:
                    accuracy, row_recovery, count = single(target_column, input_dir, strategy, obs_file, obs_year, aux_file, aux_year, reg)

                    save(output_path, target_column, aux_year, obs_year, strategy, accuracy, row_recovery, count)

    print("😀")


if __name__ == "__main__":
    main()
