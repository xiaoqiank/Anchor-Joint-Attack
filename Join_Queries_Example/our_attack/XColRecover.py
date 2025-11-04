import sys
import os
import csv
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
from utils.emd import emd_wrapper_matching
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Warning: numerical errors at iteration")
 # Use joint distribution information to assist in the recovery of another related column
def XColRecover(year, results_iter, cooccur_obs_path, cooccur_aux_path, tau):

    df_obs = pd.read_csv(cooccur_obs_path, dtype=str)
    df_aux = pd.read_csv(cooccur_aux_path, dtype=str)
    all_results = []

    for triple in results_iter:
        obs_label = str(triple[0])
        aux_label = str(triple[1])
        conf = float(triple[2])
        #print(conf)

        # Get all joint frequency vectors corresponding to obs_label
        df_obs_sub = df_obs[df_obs[df_obs.columns[0]] == obs_label]
        df_aux_sub = df_aux[df_aux[df_aux.columns[0]] == aux_label]
        #print(df_obs_sub)
        if df_obs_sub.empty or df_aux_sub.empty:
            continue
        # The intermediate iteration uses single-column frequency
        obs_labels = df_obs_sub.iloc[:, 1].astype(str).tolist()
        obs_fre1 = df_obs_sub.iloc[:, 3].astype(float).tolist()
        obs_fre2 = [0.0] * len(obs_labels)

        aux_labels = df_aux_sub.iloc[:, 1].astype(str).tolist()
        aux_fre1 = df_aux_sub.iloc[:, 3].astype(float).tolist()
        aux_fre2 = [0.0] * len(aux_labels)

        emd_results = emd_wrapper_matching(
            obs_labels=obs_labels,
            obs_fre1=obs_fre1,
            obs_fre2=obs_fre2,
            aux_labels=aux_labels,
            aux_fre1=aux_fre1,
            aux_fre2=aux_fre2,
            year=year
        )

        for o, p, score in emd_results:
            if score>tau:
                all_results.append((str(o), str(p), float(score)))

    # Merge to ensure obs_label is unique
    df_all = pd.DataFrame(all_results, columns=["obs_label", "aux_label", "confidence"])
    df_all["count"] = 1

    df_grouped = df_all.groupby(["obs_label", "aux_label"]).agg({
        "confidence": "max",
        "count": "count"
    }).reset_index()

    # Find the optimal aux_label for each obs_label
    # ==== [MODIFIED] enhanced filtering logic for duplicates/conflicts ====
    # Enhanced filtering logic, preferentially keeping candidate mappings with confidence > tau and not equal to 1
    filtered_rows = []

    for obs_label, group in df_grouped.groupby("obs_label"):
        # Try to preferentially keep items with confidence > tau and not equal to 1
        preferred = group[(group["confidence"] > tau) & (group["confidence"] < 1.0)]
        if not preferred.empty:
            best_row = preferred.sort_values(["confidence", "count"], ascending=[False, False]).iloc[0]
        else:
            best_row = group.sort_values(["confidence", "count"], ascending=[False, False]).iloc[0]
        filtered_rows.append(best_row.to_dict())

    if not filtered_rows:
        # print("[WARNING] No eligible candidates, returning empty DataFrame")
        df_best = pd.DataFrame(columns=["obs_label", "aux_label", "confidence"])
    else:
        df_best = pd.DataFrame(filtered_rows)
    # ==== [END MODIFIED] ====

    # print("[DEBUG] df_best.columns:", df_best.columns.tolist())
    # print("[DEBUG] df_best preview:\n", df_best.head())

    required_cols = ["obs_label", "aux_label", "confidence"]
    if not all(col in df_best.columns for col in required_cols):
        print("[ERROR] df_best is missing required columns, current columns are:", df_best.columns.tolist())
        return []
    final_results = list(df_best[required_cols].itertuples(index=False, name=None))
    return final_results

