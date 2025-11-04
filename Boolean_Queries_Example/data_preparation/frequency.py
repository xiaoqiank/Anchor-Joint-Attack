#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script computes frequency information for SPARCS CCSR data (clean version with no missing values).

Input (per year):
  /path/to/dataset/sparcs_clean/sparcs_{year}.csv
    where year ∈ [2018..2024]

Output Directory:
  /path/to/output/frequency

Output (for each year):
  1) Single-column frequency TXT (tab-separated with headers "Value\tCount\tFrequency")
     - Diagnosis_freq_{year}.txt
     - Procedure_freq_{year}.txt
  2) Pair-column frequency CSV
     - Diagnosis_Procedure_{year}.csv
     - Procedure_Diagnosis_{year}.csv
"""

import os
import pandas as pd

# ===== Paths and Constants =====
IN_DIR  = "/path/to/dataset/sparcs_clean"  # Input data directory (obfuscated)
OUT_DIR = "/path/to/output/frequency"     # Output directory (obfuscated)
YEARS   = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

COL_DIAG = "CCSR Diagnosis Code"  # Diagnosis column name
COL_PROC = "CCSR Procedure Code"  # Procedure column name

os.makedirs(OUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

def write_freq_txt(series_counts: pd.Series, total_rows: int, out_txt_path: str):
    """
    Writes value_counts of a column to a tab-separated txt file.
    Header: Value\tCount\tFrequency
    Rows:  Value\tCount\tFrequency (Count/total_rows)
    Sorted by Count (descending), then Value (ascending).
    """
    sc = series_counts.rename("Count")
    if sc.index.name is None:
        sc.index.name = "Value"

    df_freq = sc.reset_index()

    # Rename the first column to 'Value' to avoid KeyError
    first_col = df_freq.columns[0]
    df_freq = df_freq.rename(columns={first_col: "Value"})

    # Calculate Frequency
    df_freq["Frequency"] = df_freq["Count"].astype(float) / float(total_rows)

    # Sort: by Count descending, Value ascending
    df_freq = df_freq.sort_values(by=["Count", "Value"], ascending=[False, True])

    # Write to txt file
    with open(out_txt_path, "w", encoding="utf-8", newline="") as f:
        f.write("Value\tCount\tFrequency\n")
        for _, row in df_freq.iterrows():
            f.write(f"{row['Value']}\t{int(row['Count'])}\t{row['Frequency']:.12f}\n")

def one_year(year: int):
    """Process a single year of data"""
    in_csv = os.path.join(IN_DIR, f"sparcs_{year}.csv")  # Input file path
    if not os.path.exists(in_csv):
        print(f"[WARN] Input file for {year} does not exist: {in_csv} (skipping)")
        return

    # Read the CSV file as string type, skip bad lines
    try:
        df = pd.read_csv(in_csv, dtype=str, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(in_csv, dtype=str, encoding="latin1", on_bad_lines="skip")

    # Basic validation: check for necessary columns
    for col in (COL_DIAG, COL_PROC):
        if col not in df.columns:
            raise ValueError(f"Missing necessary column: {col} in {year}")

    total = len(df)
    if total == 0:
        print(f"[INFO] No rows in {year}, but empty result files will still be generated.")

    # ===== 1) Single-column frequency: TXT =====
    diag_counts = df[COL_DIAG].value_counts(dropna=False)
    out_diag_txt = os.path.join(OUT_DIR, f"Diagnosis_freq_{year}.txt")
    write_freq_txt(diag_counts, total, out_diag_txt)

    proc_counts = df[COL_PROC].value_counts(dropna=False)
    out_proc_txt = os.path.join(OUT_DIR, f"Procedure_freq_{year}.txt")
    write_freq_txt(proc_counts, total, out_proc_txt)

    # ===== 2) Pair-column frequency: CSV =====
    # 2.1 Diagnosis_Procedure_{year}.csv
    joint_dp = (
        df.groupby([COL_DIAG, COL_PROC])
          .size()
          .reset_index(name="JointCount")
    )
    diag_totals = diag_counts.rename("DiagTotal").to_frame()
    dp = joint_dp.merge(diag_totals, left_on=COL_DIAG, right_index=True, how="left")
    dp["CondFreq"] = dp["JointCount"] / dp["DiagTotal"].astype(float)
    dp = dp.rename(columns={COL_DIAG: "Diagnosis", COL_PROC: "Procedure"})
    dp = dp[["Diagnosis", "Procedure", "JointCount", "CondFreq"]]
    dp = dp.sort_values(by=["Diagnosis", "CondFreq", "Procedure"],
                        ascending=[True, False, True])
    out_dp_csv = os.path.join(OUT_DIR, f"Diagnosis_Procedure_{year}.csv")
    dp.to_csv(out_dp_csv, index=False, encoding="utf-8")

    # 2.2 Procedure_Diagnosis_{year}.csv
    joint_pd = (
        df.groupby([COL_PROC, COL_DIAG])
          .size()
          .reset_index(name="JointCount")
    )
    proc_totals = proc_counts.rename("ProcTotal").to_frame()
    pd2 = joint_pd.merge(proc_totals, left_on=COL_PROC, right_index=True, how="left")
    pd2["CondFreq"] = pd2["JointCount"] / pd2["ProcTotal"].astype(float)
    pd2 = pd2.rename(columns={COL_PROC: "Procedure", COL_DIAG: "Diagnosis"})
    pd2 = pd2[["Procedure", "Diagnosis", "JointCount", "CondFreq"]]
    pd2 = pd2.sort_values(by=["Procedure", "CondFreq", "Diagnosis"],
                          ascending=[True, False, True])
    out_pd_csv = os.path.join(OUT_DIR, f"Procedure_Diagnosis_{year}.csv")
    pd2.to_csv(out_pd_csv, index=False, encoding="utf-8")

    print(f"[OK] {year}: total={total} → "
          f"{os.path.basename(out_diag_txt)}, "
          f"{os.path.basename(out_proc_txt)}, "
          f"{os.path.basename(out_dp_csv)}, "
          f"{os.path.basename(out_pd_csv)}")

def main():
    for y in YEARS:
        one_year(y)
    print(f"\n All completed. Output directory: {OUT_DIR}")

if __name__ == "__main__":
    main()
