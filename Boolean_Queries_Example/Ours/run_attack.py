import sys
import os
import csv
import math
import random
import numpy as np

from ourattack_sample.GetAnchors import GetAnchors
from ourattack_sample.XColRecover import XColRecover
from ourattack_sample.Remain import Remain


def init_anchor(attack_columns, aux_years, obs_years, input_dir, output_dir, reg, confidence_t, euclidean_t, sample_labels):
    for aux_year in aux_years:
        for i in range(0, 2):
            j = abs(i - 1)
            target_column = attack_columns[i]
            dependent_column = attack_columns[j]
            aux_file = os.path.join(input_dir, f"{target_column}_freq_{aux_year}.txt")
            aux_co_file = os.path.join(input_dir, f"{target_column}_{dependent_column}_{aux_year}.csv")
            for obs_year in obs_years:
                target_label = sample_labels[target_column][obs_year]
                dependent_label = sample_labels[dependent_column][obs_year]
                obs_co_file = os.path.join(input_dir, f"{target_column}_{dependent_column}_{obs_year}.csv")
                GetAnchors(target_column, aux_year, obs_year, aux_file, aux_co_file, obs_co_file, output_dir, reg,
                           confidence_t, euclidean_t, target_label, dependent_label)


def joint_recover(attack_columns, aux_years, obs_years, iter_cnts, reg, confidence_t, euclidean_t, input_dir, output_dir, sample_labels):
    for aux_year in aux_years:
        output_path = os.path.join(output_dir, f"record_aux{aux_year}")
        for iter_cnt in iter_cnts:
            for i in range(0, 2):
                j = abs(i - 1)
                target_column = attack_columns[i]
                dependent_column = attack_columns[j]
                for obs_year in obs_years:
                    iter_prev = iter_cnt - 1
                    anchor_path = os.path.join(output_path, f"output_{dependent_column}{iter_prev}/{dependent_column}_anchor_{obs_year}.csv")
                    anchor = read_previous(dependent_column, obs_year, anchor_path)
                    previous_path = os.path.join(output_path, f"output_{target_column}{iter_prev}/{target_column}_result_{obs_year}.csv")
                    result_previous = read_previous(target_column, obs_year, previous_path)
                    target_label = sample_labels[target_column][obs_year]
                    dependent_label = sample_labels[dependent_column][obs_year]
                    XColRecover(iter_cnt, target_column, dependent_column, aux_year, obs_year, anchor, result_previous, reg,
                                confidence_t, euclidean_t, input_dir, output_path, target_label, dependent_label)


def remain(attack_columns, aux_years, obs_years, iter_cnts, reg, confidence_t, euclidean_t, input_dir, output_dir, remain_mathod, sample_labels):
    """Run Remain for each (aux_year, target_column, obs_year), call score and
    return a dict of accuracies lists per target column.
    """
    iter_cnt = iter_cnts[-1]
    accuracies = {col: {"value": [], "row": []} for col in attack_columns}

    for aux_year in aux_years:
        for i in range(0, 2):
            j = abs(i - 1)
            target_column = attack_columns[i]
            dependent_column = attack_columns[j]
            aux_file = os.path.join(input_dir, f"{target_column}_freq_{aux_year}.txt")
            for obs_year in obs_years:
                obs_co_file = os.path.join(input_dir, f"{target_column}_{dependent_column}_{obs_year}.csv")
                target_label = sample_labels[target_column][obs_year]
                dependent_label = sample_labels[dependent_column][obs_year]
                final_Result, remain_Result = Remain(iter_cnt, target_column, aux_year, obs_year, remain_mathod, aux_file, obs_co_file, output_dir, reg, target_label, dependent_label)
                value_acc, row_rec = score(final_Result, aux_year, obs_year, target_column, output_dir, obs_co_file, target_label, dependent_label)
                accuracies[target_column]["value"].append(value_acc)
                accuracies[target_column]["row"].append(row_rec)

    return accuracies


def score(final_Result, aux_year, obs_year, target_column, output_dir, obs_co_file, sampled_target_labels, sampled_dependent_labels):
    """Compute value-level accuracy and row-recovery rate.

    - value_accuracy: proportion of sampled labels whose matched label equals the obs label
    - row_recovery: sum of counts (3rd col in obs_co_file) for rows where
      (col1 in sampled_target_labels AND col2 in sampled_dependent_labels)
      and the obs label was matched correctly, divided by total sum of counts for
      rows satisfying (col1 in sampled_target_labels AND col2 in sampled_dependent_labels).
    Returns (value_accuracy, row_recovery)
    """
    save_dir = os.path.join(output_dir, f"record_aux{aux_year}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{target_column}_acc.txt")

    # value-level accuracy
    correct = 0
    total = len(final_Result)
    correct_labels = set()
    for obs_label, aux_label in final_Result:
        if obs_label == aux_label:
            correct += 1
            correct_labels.add(obs_label)
    value_accuracy = correct / total if total > 0 else 0.0

    # row-level recovery: sum counts from obs_co_file for sampled pairs
    sum_all = 0.0
    sum_result = 0.0
    target_set = set(sampled_target_labels or [])
    dep_set = set(sampled_dependent_labels or [])
    try:
        with open(obs_co_file, "r", newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    cnt = float(row[2])
                except Exception:
                    continue
                t_label = row[0].strip()
                i_label = row[1].strip()
                if t_label in target_set and i_label in dep_set:
                    sum_all += cnt
                    if t_label in correct_labels:
                        sum_result += cnt
    except FileNotFoundError:
        # if obs_co_file missing, row recovery stays 0
        pass

    row_recovery = (sum_result / sum_all) if sum_all > 0 else 0.0

    print(f"{target_column}_{obs_year}: value_acc={value_accuracy:.4f} ({correct}/{total}), row_recovery={row_recovery:.4f} ({sum_result:.1f}/{sum_all:.1f})")

    file_exists = os.path.exists(save_path)
    with open(save_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("target_column,aux_year,obs_year,value_accuracy,row_recovery\n")
        f.write(f"{target_column},{aux_year},{obs_year},{value_accuracy:.6f},{row_recovery:.6f}\n")

    return value_accuracy, row_recovery


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

def read_weighted_sample(data_path, sample_size):
    labels, weights = _read_label_weights_skip_header(data_path)
    n = len(labels)
    if n == 0:
        return []
    k = _resolve_k_ratio(sample_size, n)  
    if k <= 0:
        return []
    total_pos = sum(w for w in weights if w > 0)
    if total_pos <= 0:
        return random.sample(labels, k=min(k, n))
    w = np.array([max(0.0, float(x)) for x in weights], dtype=float)
    p = w / w.sum()
    picked = np.random.choice(labels, size=min(k, n), replace=False, p=p)
    return picked.tolist()

def read_random_sample(data_path, sample_size):
    labels = _read_labels_skip_header(data_path)
    n = len(labels)
    if n == 0:
        return []
    k = _resolve_k_ratio(sample_size, n)
    if k <= 0:
        return []
    return random.sample(labels, k=min(k, n))

def _read_labels_skip_header(path):
    labels = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:  
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if parts:
            labels.append(parts[0])
    return labels

def _read_label_weights_skip_header(path):
    labels, weights = [], []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:  
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        lab = parts[0]
        try:
            w = float(parts[1])
        except ValueError:
            continue
        labels.append(lab)
        weights.append(w)
    return labels, weights

def _resolve_k_ratio(sample_size, n):
    k = int(math.floor(n * float(sample_size)))

    if k > n:
        k = n
    if k < 0:
        k = 0
    return k

def sample(input_dir, sample_strategy, sample_size, sample_column, obs_years):
    sample_labels = {}
    for obs_year in obs_years:
        data_path = os.path.join(input_dir, f"{sample_column}_freq_{obs_year}.txt")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if sample_strategy == "weighted":
            sample_label = read_weighted_sample(data_path, sample_size)
        elif sample_strategy == "random":
            sample_label = read_random_sample(data_path, sample_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {sample_strategy}")
        sample_labels[obs_year] = sample_label
    return sample_labels

def main():
    # Parameter settings
    # dataset root is relative to current working directory
    input_dir = os.path.join("dataset", "frequency")  # relative path to frequency data
    remain_mathod = "emd"
    reg = 0.0001
    confidence_t = 0.0001
    euclidean_t = 0.08
    attack_columns = ["Diagnosis", "Procedure"]
    aux_years = [i for i in range(2018, 2025)]
    obs_years = [j for j in range(2018, 2025)]
    iter_cnts = [k for k in range(2, 16)]  
    sample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample_strategies = ["weighted", "random"]
    sample_columns = ["Diagnosis", "Procedure", "mix"]

    # base output folder (relative)
    base_output_root = os.path.join("outputs", "Attack_Test_results")

    for sample_strategy in sample_strategies:
        for sample_column in sample_columns:
            for sample_size in sample_sizes:
                # per-configuration output dir (relative)
                output_dir = os.path.join(base_output_root, f"samplestrategy_{sample_strategy}", f"samplecolumn_{sample_column}", f"samplesize_{sample_size}")
                os.makedirs(output_dir, exist_ok=True)

                runs = 10

                for run_idx in range(runs):
                    # build sampled labels fresh each run
                    if sample_column == attack_columns[0]:
                        sample_labels0 = sample(input_dir, sample_strategy, sample_size, attack_columns[0], obs_years)
                        sample_labels1 = sample(input_dir, sample_strategy, 1, attack_columns[1], obs_years)
                    elif sample_column == attack_columns[1]:
                        sample_labels0 = sample(input_dir, sample_strategy, 1, attack_columns[0], obs_years)
                        sample_labels1 = sample(input_dir, sample_strategy, sample_size, attack_columns[1], obs_years)
                    else:  # mix
                        sample_size0 = sample_size / 2
                        sample_size1 = sample_size - sample_size0
                        sample_labels0 = sample(input_dir, sample_strategy, sample_size0, attack_columns[0], obs_years)
                        sample_labels1 = sample(input_dir, sample_strategy, sample_size1, attack_columns[1], obs_years)

                    sample_labels_run = {attack_columns[0]: sample_labels0, attack_columns[1]: sample_labels1}

                    # per-run output dir
                    output_dir_run = os.path.join(output_dir, f"run_{run_idx}")
                    os.makedirs(output_dir_run, exist_ok=True)

                    init_anchor(attack_columns, aux_years, obs_years, input_dir, output_dir_run, reg, confidence_t, euclidean_t, sample_labels_run)
                    joint_recover(attack_columns, aux_years, obs_years, iter_cnts, reg, confidence_t, euclidean_t, input_dir, output_dir_run, sample_labels_run)
                    run_accur = remain(attack_columns, aux_years, obs_years, iter_cnts, reg, confidence_t, euclidean_t, input_dir, output_dir_run, remain_mathod, sample_labels_run)


                print("Finished parameter set:", sample_strategy, sample_size, sample_column)


if __name__ == "__main__":
    main()






