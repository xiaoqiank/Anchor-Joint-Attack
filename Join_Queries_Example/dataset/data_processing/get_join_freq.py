import os, duckdb
import pandas as pd
import numpy as np


def get_sorted_freq(con, table, column, year):
    query = f'''
    SELECT CAST("{column}" AS TEXT) AS ca
    FROM {table}
    WHERE year = {year} AND "{column}" IS NOT NULL
    '''
    df = con.execute(query).fetchdf()
    df['ca'] = df['ca'].str.strip()  # Remove spaces and other abnormal characters
    freq = df['ca'].value_counts().sort_index()
    return freq.index.tolist(), freq.values / freq.values.sum()


def gensi_ct_freq(con, table1, table2, t1_col, t2_col, years, output_dir):
    for year in years:
        print(f"\n[Year {year}] Processing...")

        labels_t1, freq_t1 = get_sorted_freq(con, table1, t1_col, year)
        labels_t2, freq_t2 = get_sorted_freq(con, table2, t2_col, year)

        freq_dict1 = dict(zip(labels_t1, freq_t1))
        freq_dict2 = dict(zip(labels_t2, freq_t2))

        common_labels = sorted(set(labels_t1).intersection(labels_t2))

        print(f"  {t1_col} labels: {len(labels_t1)}, {t2_col} labels: {len(labels_t2)}")
        print(f"  Common labels: {len(common_labels)}")
        if len(common_labels) > 0:
            print(f"  Sample common labels: {common_labels[:5]}")

        output_file = os.path.join(output_dir, f"{t2_col}_freq_{year}.txt")
        with open(output_file, "w") as f:
            for l in common_labels:
                f1 = freq_dict1.get(l, 0)
                f2 = freq_dict2.get(l, 0)
                f.write(f"{l}\t{f1}\t{f2}\n")

        print(f"  Output written to: {output_file}")


if __name__ == "__main__":
    # Join equi-class leakage
    db_path = "dataset/chicago_data.db"
    table1 = "rideshares"
    table2 = "crimes"
    t1_col = "Dropoff Community Area"
    t2_col = "Community Area"
    years = [2018, 2019, 2020, 2021, 2022]
    output_dir = "dataset/frequency/crimes(di)-rideshares-dropoff"

    con = duckdb.connect(db_path, read_only=True)
    os.makedirs(output_dir, exist_ok=True)

    gensi_ct_freq(con, table1, table2, t1_col, t2_col, years, output_dir)

    con.close()
