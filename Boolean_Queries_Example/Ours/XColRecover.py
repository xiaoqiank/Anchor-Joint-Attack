import sys
import os
import csv
import numpy as np
import pandas as pd
import shutil

# Prefer imports from the local `utils` package. Avoid embedding any
# absolute user-specific paths in source. If project-level helper packages
# are needed, they can be prepended to sys.path dynamically at runtime.
from utils.emd import emd_joint_matching

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Warning: numerical errors at iteration")
# Use joint distribution information to assist in the recovery of another related column



def XColRecover(iter_cnt, target_column_name, dependent_column_name, aux_year, obs_year, anchor, result_previous, reg, confidence_t, euclidean_t, input_path, output_path, target_label, dependent_label):
    obs_dependent_target_copath = os.path.join(input_path, f"{dependent_column_name}_{target_column_name}_{obs_year}.csv")
    aux_dependent_target_copath = os.path.join(input_path, f"{dependent_column_name}_{target_column_name}_{aux_year}.csv")

    df_obs = pd.read_csv(obs_dependent_target_copath, dtype=str)
    df_aux = pd.read_csv(aux_dependent_target_copath, dtype=str)
    all_match = []

    for triple in anchor:
        obs_label = str(triple[0])
        aux_label = str(triple[1])
        conf = float(triple[2])
        eucl = float(triple[3])

    # target column contains only sampled labels; filter to sampled set
    df_obs_sub = df_obs[(df_obs[df_obs.columns[0]] == obs_label) & (df_obs[df_obs.columns[1]].isin(target_label))]
        df_aux_sub = df_aux[df_aux[df_aux.columns[0]] == aux_label]

        if df_obs_sub.empty or df_aux_sub.empty:
            continue

        obs_labels = df_obs_sub.iloc[:, 1].astype(str).tolist()
        obs_freq = df_obs_sub.iloc[:, 3].astype(float).tolist()
        aux_labels = df_aux_sub.iloc[:, 1].astype(str).tolist()
        aux_freq = df_aux_sub.iloc[:, 3].astype(float).tolist()


        val_match = emd_joint_matching(
            obs_labels=obs_labels,
            obs_freq=obs_freq,
            aux_labels=aux_labels,
            aux_freq=aux_freq,
            reg=reg
        )

        for o, p, score in val_match:
            all_match.append((str(o), str(p), float(score)))

    if not all_match:
        # If there are no matches, copy the previous iteration's result CSV (if exists)
        #print(f"[{obs_year}] No match result, trying to copy previous results...")
        prev_path = os.path.join(output_path, f"output_{target_column_name}{iter_cnt-1}/{target_column_name}_result_{obs_year}.csv")
        new_path = os.path.join(output_path, f"output_{target_column_name}{iter_cnt}/{target_column_name}_result_{obs_year}.csv")
        if os.path.exists(prev_path):
            try:
                os.makedirs(os.path.dirname(new_path) or '.', exist_ok=True)
                shutil.copy2(prev_path, new_path)
                print(f"[{obs_year}] No match result; copied previous results from {prev_path} to {new_path}")
            except Exception as e:
                print(f"[{obs_year}] Warning: failed to copy previous result file: {e}")
        else:
            print(f"[{obs_year}] No match result and no previous file at {prev_path}, skip")
        return [], [], []

    # compute Euclidean distance between truncated vectors
    obs_target_dependent_copath = os.path.join(input_path, f"{target_column_name}_{dependent_column_name}_{obs_year}.csv")
    aux_target_dependent_copath = os.path.join(input_path, f"{target_column_name}_{dependent_column_name}_{aux_year}.csv")
  
    obs_freq = load_euclidean_sample_vectors(target_label, dependent_label, obs_target_dependent_copath)
    aux_freq = load_euclidean_vectors(aux_target_dependent_copath)

    results = []
    for triple in all_match:
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
        #print(results)

    df_out = pd.DataFrame(results, columns=["obs_label", "matched_aux_label", "confidence", "euclidean_distance"])
    df_out["confidence"] = pd.to_numeric(df_out["confidence"], errors="coerce").fillna(0.0)
    df_out["euclidean_distance"] = pd.to_numeric(df_out["euclidean_distance"], errors="coerce").fillna(float("inf"))
    df_out = df_out.sort_values(by=["confidence", "euclidean_distance"], ascending=[False, True]).reset_index(drop=True)
    output_r1 = os.path.join(output_path, f"output_{target_column_name}{iter_cnt}/{target_column_name}_batch_{obs_year}.csv")
    # ensure directory exists and overwrite per run
    os.makedirs(os.path.dirname(output_r1) or '.', exist_ok=True)
    df_out.to_csv(output_r1, index=False, encoding="utf-8-sig", mode='w', header=True)
    results = [(row.obs_label, row.matched_aux_label, f"{row.confidence:.6f}", f"{row.euclidean_distance:.6f}") for row in df_out.itertuples()]

    # Remove duplicates: group rows with the same obs_label, keep the row
    # with the highest confidence; if equal, prefer the smaller Euclidean
    # distance. Apply thresholds first.
    # Apply thresholds first
    df_thresholded = df_out[(df_out['confidence'] >= confidence_t) & (df_out['euclidean_distance'] <= euclidean_t)].copy()

    if not df_thresholded.empty:
        df_sorted = df_thresholded.sort_values(by=['obs_label', 'confidence', 'euclidean_distance'],
                                              ascending=[True, False, True]).copy()
        df_dedup = df_sorted.drop_duplicates(subset=['obs_label'], keep='first').reset_index(drop=True)
    else:
        df_dedup = df_thresholded.copy()
    # sort deduped results by confidence desc, euclidean_distance asc for tie-break, before saving
    if not df_dedup.empty:
        df_dedup = df_dedup.sort_values(by=['confidence', 'euclidean_distance'], ascending=[False, True]).reset_index(drop=True)

    output_r2 = os.path.join(output_path, f"output_{target_column_name}{iter_cnt}/{target_column_name}_filter1_{obs_year}.csv")
    os.makedirs(os.path.dirname(output_r2) or '.', exist_ok=True)
    df_dedup.to_csv(output_r2, index=False, encoding="utf-8-sig", mode='w', header=True)
    results1 = [(row.obs_label, row.matched_aux_label, f"{row.confidence:.6f}", f"{row.euclidean_distance:.6f}") for row in df_dedup.itertuples()]


    # second-stage filtering: compare anchor and results1, prefer rows with
    # higher confidence; tie-breaker is smaller Euclidean distance. Keep
    # distinct obs_label entries for final output.

    # Build DataFrames for anchor and results1, ensure numeric types for comparisons
    df_result_previous = pd.DataFrame(result_previous, columns=["obs_label", "matched_aux_label", "confidence", "euclidean_distance"]) 
    df_result_previous["confidence"] = pd.to_numeric(df_result_previous["confidence"], errors="coerce").fillna(0.0)
    df_result_previous["euclidean_distance"] = pd.to_numeric(df_result_previous["euclidean_distance"], errors="coerce").fillna(float("inf"))

    df_results1 = pd.DataFrame(results1, columns=["obs_label", "matched_aux_label", "confidence", "euclidean_distance"]) 
    df_results1["confidence"] = pd.to_numeric(df_results1["confidence"], errors="coerce").fillna(0.0)
    df_results1["euclidean_distance"] = pd.to_numeric(df_results1["euclidean_distance"], errors="coerce").fillna(float("inf"))

    # --- Merge previous and current result sets and deduplicate according to rules ---
    # concat previous results and current results1
    df_combined = pd.concat([df_result_previous, df_results1], ignore_index=True)
    # ensure numeric types
    df_combined['confidence'] = pd.to_numeric(df_combined['confidence'], errors='coerce').fillna(0.0)
    df_combined['euclidean_distance'] = pd.to_numeric(df_combined['euclidean_distance'], errors='coerce').fillna(float('inf'))

    # For duplicate obs_label keep the row with highest confidence, tie-breaker: smallest euclidean_distance
    if not df_combined.empty:
        df_combined_sorted = df_combined.sort_values(by=['obs_label', 'confidence', 'euclidean_distance'],
                                                     ascending=[True, False, True]).copy()
        df_merged = df_combined_sorted.drop_duplicates(subset=['obs_label'], keep='first').reset_index(drop=True)
        # final ordering: confidence desc, euclidean asc
        df_merged = df_merged.sort_values(by=['confidence', 'euclidean_distance'], ascending=[False, True]).reset_index(drop=True)
    else:
        df_merged = df_combined.copy()
    output_r3 = os.path.join(output_path, f"output_{target_column_name}{iter_cnt}/{target_column_name}_result_{obs_year}.csv")
    os.makedirs(os.path.dirname(output_r3) or '.', exist_ok=True)
    df_merged.to_csv(output_r3, index=False, encoding='utf-8-sig', mode='w', header=True)
    results2 = [(row.obs_label, row.matched_aux_label, f"{row.confidence:.6f}", f"{row.euclidean_distance:.6f}") for row in df_merged.itertuples()]

    # --- Generate the subset of rows present in results1 but not in result_previous
    # --- and save that subset as the new anchor file ---
    prev_labels = set(df_result_previous['obs_label'].tolist())
    df_res2_only = df_merged[~df_merged['obs_label'].isin(prev_labels)].copy()
    if not df_res2_only.empty:
        df_res2_only['confidence'] = pd.to_numeric(df_res2_only['confidence'], errors='coerce').fillna(0.0)
        df_res2_only['euclidean_distance'] = pd.to_numeric(df_res2_only['euclidean_distance'], errors='coerce').fillna(float('inf'))
        df_res2_only = df_res2_only.sort_values(by=['confidence', 'euclidean_distance'], ascending=[False, True]).reset_index(drop=True)
    output_r4 = os.path.join(output_path, f"output_{target_column_name}{iter_cnt}/{target_column_name}_anchor_{obs_year}.csv")
    os.makedirs(os.path.dirname(output_r4) or '.', exist_ok=True)
    df_res2_only.to_csv(output_r4, index=False, encoding='utf-8-sig', mode='w', header=True)

    # return combined results: results, results1, results2 (merged)
    return results, results1, results2


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


def read_previous(column_name, obs_year, result_previous_path):
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


if __name__ == "__main__":
    # Use project-relative defaults to avoid embedding personal absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(project_root, 'dataset', 'frequency')
    output_path = os.path.join(project_root, 'output')
    aux_year = 2018
    obs_years = [2019, 2020, 2021, 2022]
    attack_name = "ours"
    table = "sparcs"
    reg=0.0001

    iter_cnts = [i for i in range(2, 16)]

 
    for iter_cnt in iter_cnts:

        target_column_name="Diagnosis"
        dependent_column_name="Procedure"
        confidence_t = 0.0001  
        euclidean_t = 0.08    
       

        #inter=2 top-100: 0.0004, 0.06; all: 0.0001,0.04
        

        for obs_year in obs_years:
            iter_prev = iter_cnt - 1
            # take tuples of (dependent, target, confidence, euclidean)
            anchor_path = os.path.join(output_path, f"output_{dependent_column_name}{iter_prev}/{dependent_column_name}_anchor_{obs_year}.csv")
            anchor=read_previous(dependent_column_name, obs_year, anchor_path)
            #print("anchor length:", len(anchor))
            previous_path=os.path.join(output_path, f"output_{target_column_name}{iter_prev}/{target_column_name}_result_{obs_year}.csv")
            result_previous=read_previous(target_column_name, obs_year, previous_path)
            #print("result_previous length:", len(result_previous))
            XColRecover(iter_cnt,target_column_name, dependent_column_name, aux_year, obs_year, anchor, result_previous,reg, confidence_t, euclidean_t, input_path, output_path)
            


        
        target_column_name="Procedure"
        dependent_column_name="Diagnosis"
        confidence_t =0.0001 
        euclidean_t =0.08 
       

        for obs_year in obs_years:
            iter_prev = iter_cnt - 1
            # take tuples of (dependent, target, confidence, euclidean)
            anchor_path = os.path.join(output_path, f"output_{dependent_column_name}{iter_prev}/{dependent_column_name}_anchor_{obs_year}.csv")
            anchor=read_previous(dependent_column_name, obs_year, anchor_path)
            #print("anchor length:", len(anchor))
            previous_path=os.path.join(output_path, f"output_{target_column_name}{iter_prev}/{target_column_name}_result_{obs_year}.csv")
            result_previous=read_previous(target_column_name, obs_year, previous_path)
            #print("result_previous length:", len(result_previous))
            XColRecover(iter_cnt,target_column_name, dependent_column_name, aux_year, obs_year, anchor, result_previous,reg, confidence_t, euclidean_t, input_path, output_path)
        

