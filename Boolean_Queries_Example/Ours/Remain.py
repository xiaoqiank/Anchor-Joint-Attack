import os
import csv
import numpy as np
import pandas as pd
from utils.emd import emd_joint_matching
from utils.bipartite_matching import bipartite_matching
import warnings
warnings.filterwarnings("ignore", message="Sinkhorn did not converge.*")


def Remain(iter_cnt, column_name, aux_year, obs_year, remain_mathod, aux_file, obs_co_file, output_dir, reg, target_label, dependent_label):
    """Compute matches for remaining (unmatched) labels.

    Reads existing matched pairs if present and computes matches for the
    labels that remain unmatched using either an EMD-based joint matching
    or a bipartite matching fallback.
    """

    # Per-aux-year output folder that groups intermediate results
    output_path = os.path.join(output_dir, f"record_aux{aux_year}")
    aux_labels_all, aux_freq_all = load_text_file(aux_file)
    obs_labels_all, obs_freq_all = load_sample_freq(target_label, dependent_label, obs_co_file)



    # read current matching result (this iteration) if present
    matching_file = os.path.join(output_path, f"output_{column_name}{iter_cnt}/{column_name}_result_{obs_year}.csv")
    obs_match_labels = []
    aux_match_labels = []
    matched_Result = []

    if os.path.exists(matching_file):
        try:
            df_match = pd.read_csv(matching_file, dtype=str)
            if df_match.shape[1] >= 2:
                obs_match_labels = df_match.iloc[:, 0].astype(str).tolist()
                aux_match_labels = df_match.iloc[:, 1].astype(str).tolist()
                matched_Result = [(str(r[0]), str(r[1])) for r in df_match.iloc[:, :2].itertuples(index=False, name=None)]
        except Exception:
            # fallback to csv.reader for malformed files
            with open(matching_file, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                try:
                    first = next(reader)
                except StopIteration:
                    first = None
                # if header, skip non-data
                if first is not None and len(first) >= 2:
                    try:
                        _ = float(first[2])
                        matched_Result.append((first[0], first[1]))
                    except Exception:
                        pass
                for row in reader:
                    if len(row) < 2:
                        continue
                    matched_Result.append((row[0], row[1]))
                obs_match_labels = [r[0] for r in matched_Result]
                aux_match_labels = [r[1] for r in matched_Result]


    # Determine unmatched labels by set difference: candidates for the remain step
    obs_unmatched_labels = list(set(obs_labels_all) - set(obs_match_labels))
    aux_unmatched_labels = list(set(aux_labels_all) - set(aux_match_labels))

    # build frequency maps for quick lookup
    obs_freq_map = {label: freq for label, freq in zip(obs_labels_all, obs_freq_all)}
    aux_freq_map = {label: freq for label, freq in zip(aux_labels_all, aux_freq_all)}

    # Get frequencies for unmatched labels
    obs_unmatched_freq = [float(obs_freq_map.get(l, 0.0)) for l in obs_unmatched_labels]
    aux_unmatched_freq = [float(aux_freq_map.get(l, 0.0)) for l in aux_unmatched_labels]

  
    if remain_mathod=="emd":
        val_match_remain = emd_joint_matching(
            aux_labels=aux_unmatched_labels,
            aux_freq=aux_unmatched_freq,
            obs_labels=obs_unmatched_labels,
            obs_freq=obs_unmatched_freq,
            reg=reg
        )
    
    elif remain_mathod=="bimatch":
        # bipartite_matching signature: bipartite_matching(aux_labels, aux_fre, obs_labels, obs_fre)
        val_match_remain = bipartite_matching(
            aux_unmatched_labels,
            aux_unmatched_freq,
            obs_unmatched_labels,
            obs_unmatched_freq
        )
    
    # Normalize matches: emd_joint_matching returns (o,p,score) while bipartite_matching returns (o,p)
    val_match = []
    for t in val_match_remain:
        if not t:
            continue
        if len(t) >= 3:
            try:
                score = float(t[2])
            except Exception:
                score = 1.0
            val_match.append((str(t[0]), str(t[1]), score))
        elif len(t) == 2:
            # bipartite_matching returns pairs without confidence; assign default confidence 1.0
            val_match.append((str(t[0]), str(t[1]), 1.0))
        else:
            # unexpected shape, skip
            continue
    val_match.sort(key=lambda x: x[2], reverse=True)

    # write remaining matches to CSV
    remain_dir = os.path.join(output_path, f"output_{column_name}_remain")
    os.makedirs(remain_dir or '.', exist_ok=True)
    remain_path = os.path.join(remain_dir, f"{column_name}_remain_{obs_year}.csv")
    if val_match:
        df_remain = pd.DataFrame(val_match, columns=["obs_label", "matched_aux_label", "confidence"])
        df_remain.to_csv(remain_path, index=False, encoding='utf-8-sig', mode='w', header=True)
    else:
        # write empty file with header
        df_remain = pd.DataFrame(columns=["obs_label", "matched_aux_label", "confidence"])
        df_remain.to_csv(remain_path, index=False, encoding='utf-8-sig', mode='w', header=True)

    remain_Result = [(t[0], t[1]) for t in val_match]

    Results = matched_Result + remain_Result
    # write final combined results
    final_dir = os.path.join(output_path, "output_final_results")
    os.makedirs(final_dir or '.', exist_ok=True)
    final_path = os.path.join(final_dir, f"{column_name}_{obs_year}.csv")
    df_final = pd.DataFrame(Results, columns=["obs_label", "matched_aux_label"]) if Results else pd.DataFrame(columns=["obs_label", "matched_aux_label"])
    df_final.to_csv(final_path, index=False, encoding='utf-8-sig', mode='w', header=True)

    return Results, val_match

    
 

def load_sample_freq(target_label, dependent_label, obs_co_file):
    target_set = set(target_label)
    dep_set = set(dependent_label)
    label_count = {}
    total_sum = 0.0

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


def read_joint_result(column_name, obs_year, result_previous_path):
    anchor = []
    # If file does not exist, return empty list
    if not os.path.isfile(result_previous_path):
        return anchor

    with open(result_previous_path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            first = None

        # if first row looks like data (has >=4 cols and col[2] can be parsed as float) keep it
        if first is not None and len(first) >= 4:
            try:
                _ = float(first[2])
                anchor.append((first[0], first[1], float(first[2]), float(first[3])))
            except Exception:
                # first row may be a header or malformed, skip it
                pass

        for row in reader:
            if len(row) < 4:
                continue
            try:
                c = float(row[2])
                d = float(row[3])
            except Exception:
                continue
            anchor.append((row[0], row[1], c, d))

    return anchor


def score(final_Result,obs_year,target_column,output_path):
    save_path=os.path.join(output_path, f"output_final_results/{target_column}.txt")
    correct = 0
    total = len(final_Result)
    for obs_label, aux_label in final_Result:
        if obs_label == aux_label:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"{target_column}_{obs_year}:{accuracy:.4f} ({correct}/{total})")

    # Write summary accuracy to save_path (append mode): target_column, obs_year, accuracy
    return accuracy






        
