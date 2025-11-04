import sys
import os
import csv
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
from utils.emd import emd_wrapper_matching
import warnings
warnings.filterwarnings("ignore", message="Sinkhorn did not converge.*")


# New function: GetAnchors, to get the initial set of anchor points
def GetAnchors(year, aux_path, obs_path,cooccur_aux_path, cooccur_obs_path, tau, mu, output_csv,type,e,reg):
    # e: whether to calculate Euclidean distance
    val_match = run_attack_for_year(year, aux_path, obs_path, type, tau, reg)
    #print(val_match)
    if not val_match:
        print(f"[{year}] No match result, skip")
        return [], []

    obs_freq = load_col4_vectors(cooccur_obs_path)
    aux_freq = load_col4_vectors(cooccur_aux_path)

    results = []
    results1 = []
    for triple in val_match:
        # Triple structure: obs_label = triple[0], aux_label = triple[1], confidence = triple[2]
        o = str(triple[0])
        a = str(triple[1])
        c = float(triple[2])
        obs_v = obs_freq.get(o, [])
        aux_v = aux_freq.get(a, [])
        length = max(len(obs_v), len(aux_v))
        obs_v += [0] * (length - len(obs_v))
        aux_v += [0] * (length - len(aux_v))
        dist = np.linalg.norm(np.array(obs_v) - np.array(aux_v))
        if e==False or dist < mu:
            results.append(triple)
        results1.append((o, a, f"{c:.6f}", f"{dist:.6f}"))
    
    return results,results1
    # df_out = pd.DataFrame(results, columns=["obs_label", "matched_aux_label", "confidence", "euclidean_distance"])
    # df_out.to_csv(output_csv, index=False)
    # print(f"Anchor file written: {output_csv}")


def run_attack_for_year(year, aux_path, obs_path, type, tau, reg):
    # type: whether to use cross-column equality

    if not os.path.exists(aux_path):
        print(f"[{year}] Auxiliary file does not exist: {aux_path}")
        return
    if not os.path.exists(obs_path):
        print(f"[{year}] Observation file does not exist: {obs_path}")
        return
    
    aux_labels, aux_fre1, aux_fre2 = load_text_file(aux_path)
    obs_labels, obs_fre1, obs_fre2 = load_text_file(obs_path)

    # Find intersection of labels
    intersect_labels = sorted(set(aux_labels) & set(obs_labels))
    if not intersect_labels:
        print(f"[{year}] No intersection between auxiliary and observation data labels")
        return

    # Align aux data
    aux_index = {label: i for i, label in enumerate(aux_labels)}
    aux_fre1_aligned = [aux_fre1[aux_index[label]] for label in intersect_labels]
    aux_fre2_aligned = [aux_fre2[aux_index[label]] for label in intersect_labels]

    # Align obs data
    obs_index = {label: i for i, label in enumerate(obs_labels)}
    obs_fre1_aligned = [obs_fre1[obs_index[label]] for label in intersect_labels]
    obs_fre2_aligned = [obs_fre2[obs_index[label]] for label in intersect_labels]


    if type == 0:
        obs_fre1_aligned = [0.0] * len(obs_fre1_aligned)
        aux_fre1_aligned = [0.0] * len(aux_fre1_aligned)

    # Execute EMD attack, return triples (obs_label, pred_label, confidence)
    val_match_raw = emd_wrapper_matching(
        aux_labels=intersect_labels,
        aux_fre1=aux_fre1_aligned,
        aux_fre2=aux_fre2_aligned,
        obs_labels=intersect_labels,
        obs_fre1=obs_fre1_aligned,
        obs_fre2=obs_fre2_aligned,
        year=year,
        reg=reg
    )

    # Convert labels to str and confidence to float
    val_match = [(str(t[0]), str(t[1]), float(t[2])) for t in val_match_raw]

    # Sort by confidence in descending order and filter items with confidence >= τ
    #val_match = [t for t in val_match if t[2] >= tau]
    val_match = [t for t in val_match if t[2] >= tau and t[2]!=1.0]
    val_match.sort(key=lambda x: x[2], reverse=True)
    #print(val_match)
    return val_match


def load_text_file(filepath):
    labels, f1, f2 = [], [], []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            labels.append(str(parts[0]))
            f1.append(float(parts[1]))
            f2.append(float(parts[2]))
    return labels, f1, f2

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


