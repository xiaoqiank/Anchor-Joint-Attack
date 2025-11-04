import os
import csv
import numpy as np
import pandas as pd
import sys
# Prefer local sibling packages; if an external helper package exists in a
# repository-level folder `AdditionalExp/ourattack`, add it to sys.path so
# the script can import those helpers without embedding any user-specific
# absolute paths in the source file.
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
alt_helper_path = os.path.join(proj_root, 'AdditionalExp', 'ourattack')
if os.path.isdir(alt_helper_path):
    sys.path.insert(0, alt_helper_path)
from utils.emd import emd_joint_matching
import warnings
warnings.filterwarnings("ignore", message="Sinkhorn did not converge.*")



def GetAnchors(target_column, aux_year, year, aux_file, aux_co_file, obs_co_file, output_dir, reg, confidence_t, euclidean_t, target_label, dependent_label):    
    
    output_path = os.path.join(output_dir, f"record_aux{aux_year}")
    val_match = run_single_attack(year, aux_file, obs_co_file, target_label,dependent_label, reg)
    if not val_match:
        print(f"[{year}] No match result, skip")
        return [], []

    aux_freq = load_euclidean_vectors(aux_co_file)
    obs_freq = load_euclidean_sample_vectors(target_label, dependent_label, obs_co_file)

    results = []
    for triple in val_match:
        o = str(triple[0])
        a = str(triple[1])
        c = float(triple[2])
        obs_v = obs_freq.get(o, [])
        aux_v = aux_freq.get(a, [])

        min_len = min(len(obs_v), len(aux_v))
        if min_len == 0:
            dist = float("inf")
        else:
            obs_trunc = np.array(obs_v[:min_len])
            aux_trunc = np.array(aux_v[:min_len])
            dist = np.linalg.norm(obs_trunc - aux_trunc)
        results.append((o, a, f"{c:.6f}", f"{dist:.6f}"))


    df_out = pd.DataFrame(results, columns=["obs_label", "matched_aux_label", "confidence", "euclidean_distance"])
    df_out["confidence"] = pd.to_numeric(df_out["confidence"], errors="coerce").fillna(0.0)
    df_out["euclidean_distance"] = pd.to_numeric(df_out["euclidean_distance"], errors="coerce").fillna(float("inf"))
    df_out = df_out.sort_values(by=["confidence", "euclidean_distance"], ascending=[False, True]).reset_index(drop=True)
    output_r1 = os.path.join(output_path, f"output_{target_column}1/{target_column}_single_{year}.csv")
    # ensure directory exists and overwrite per run
    os.makedirs(os.path.dirname(output_r1) or '.', exist_ok=True)
    df_out.to_csv(output_r1, index=False, encoding="utf-8-sig", mode='w', header=True)
    results = [(row.obs_label, row.matched_aux_label, f"{row.confidence:.6f}", f"{row.euclidean_distance:.6f}") for row in df_out.itertuples()]

    output_fr1 = os.path.join(output_path, f"output_{target_column}1/{target_column}_anchor_{year}.csv")
    df_filtered = df_out[(df_out['confidence'] >= confidence_t) & (df_out['euclidean_distance'] <= euclidean_t)].copy()
    os.makedirs(os.path.dirname(output_fr1) or '.', exist_ok=True)
    df_filtered.to_csv(output_fr1, index=False, encoding="utf-8-sig", mode='w', header=True)
    results1 = [(row.obs_label, row.matched_aux_label, f"{row.confidence:.6f}", f"{row.euclidean_distance:.6f}") for row in df_filtered.itertuples()]
    #print("Results:", results1)
    return results, results1
    

# Auxiliary data is used as-is; target (observed) data may be modified/sampled
def run_single_attack(year, aux_file, obs_co_file, target_label, dependent_label, reg):
    if not os.path.exists(aux_file):
        print(f"[{year}] Auxiliary file does not exist: {aux_file}")
        return
    if not os.path.exists(obs_co_file):
        print(f"[{year}] Observation file does not exist: {obs_co_file}")
        return
    aux_labels, aux_fre = load_text_file(aux_file)
    obs_labels, obs_fre = load_sample_freq(target_label, dependent_label, obs_co_file)  # Only extract sampled frequency information

    val_match_raw = emd_joint_matching(
        aux_labels=aux_labels,
        aux_freq=aux_fre,
        obs_labels=obs_labels,
        obs_freq=obs_fre,
        reg=reg
    )

    val_match = [(str(t[0]), str(t[1]), float(t[2])) for t in val_match_raw]
    val_match.sort(key=lambda x: x[2], reverse=True)

    return val_match


def load_text_file(path):
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



def load_sample_freq(target_label, dependent_label, obs_co_file):
    target_set = set(target_label)
    dep_set = set(dependent_label)
    label_count = {}
    total_sum = 0.0

    # Only extract frequency information for sampled labels from the joint
    # observation file (obs_co_file). The obs_co_file is expected to be a
    # CSV with at least: target_label, dependent_label, count, ...
    with open(obs_co_file, "r", newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  

        for row in reader:
            if len(row) < 3:
                continue
            try:
                cnt = float(row[2])
            except ValueError:
                continue
            total_sum += cnt 

            t_label = row[0].strip()
            i_label = row[1].strip()

            if t_label in target_set and i_label in dep_set:
                label_count[t_label] = label_count.get(t_label, 0.0) + cnt

    if total_sum <= 0:
        return [], []

    label_freq_pairs = [(lab, cnt / total_sum) for lab, cnt in label_count.items()]
    label_freq_pairs.sort(key=lambda x: x[1], reverse=True)

    obs_labels = [lab for lab, _ in label_freq_pairs]
    obs_freqs = [freq for _, freq in label_freq_pairs]

    return obs_labels, obs_freqs


def load_euclidean_vectors(path): # Get joint distribution file
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



def load_euclidean_sample_vectors(target_label, dependent_label, obs_co_file):
    target_set = set(target_label)
    dep_set = set(dependent_label)
    freq_map = {}

    with open(obs_co_file, "r", newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  

        for row in reader:
            if len(row) < 4:
                continue
            try:
                t_label = row[0].strip()
                i_label = row[1].strip()
                val = float(row[3])
            except ValueError:
                continue

            if t_label in target_set and i_label in dep_set:
                freq_map.setdefault(t_label, []).append(val)

    for k in freq_map:
        freq_map[k].sort(reverse=True)
    return freq_map


