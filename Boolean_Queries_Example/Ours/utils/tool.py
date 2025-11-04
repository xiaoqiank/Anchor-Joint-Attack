import os
import sys
from collections import defaultdict
import pandas as pd


def load_col4_vectors(path): # Get joint distribution file
    freq_map = {}
    df = pd.read_csv(path, dtype={0: str})
    for _, row in df.iterrows():
        try:
            key = str(row[0])
            val = float(row[3])
        except (ValueError, IndexError):
            continue
        freq_map.setdefault(key, []).append(val)
    for k in freq_map:
        freq_map[k].sort(reverse=True)
    return freq_map



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



def read_freq_file(path):
    """Read Value & Frequency columns (auto sep). Return labels (list[str]), freqs (list[float])"""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]

        col_value, col_freq = None, None
        for c in df.columns:
            if "value" in c:
                col_value = c
            elif "freq" in c:
                col_freq = c

        if col_value is None or col_freq is None:
            raise ValueError(f"Columns (Value, Frequency) not found in {path}")

        labels = df[col_value].astype(str).fillna("").tolist()
        freqs = pd.to_numeric(df[col_freq], errors="coerce").fillna(0.0).tolist()
        return labels, freqs
    except Exception as e:
        print(f"Failed to read file {path}: {e}")
        return [], []
