import os
import duckdb
import pandas as pd
import numpy as np
import random

def generate_single_column_freq(con, table, column, year, output_dir, filter_col=None, filter_values=None):
    """Generates and saves the frequency of values for a single column, with optional filtering."""
    query = f'SELECT "{column}" AS col FROM {table} WHERE year = {year} AND "{column}" IS NOT NULL'
    if filter_col and filter_values:
        values_str = ", ".join(["'" + str(v).replace("'", "''") + "'" if isinstance(v, str) else str(v) for v in filter_values])
        query += f' AND "{filter_col}" IN ({values_str})'

    df = con.execute(query).fetchdf()
    if df.empty:
        print(f"  [WARN] No data for single-column frequency for '{column}' in {year} with given filters.")
        return

    counts = df['col'].value_counts().sort_index()
    total = counts.sum()
    
    output_filename = os.path.join(output_dir, f"{column}_freq_{year}.txt")
    with open(output_filename, "w") as f:
        for value, count in counts.items():
            freq = count / total
            f.write(f"{value}\t{count}\t{freq}\n")
    print(f"  Generated single-column frequency file: {output_filename}")


def generate_co_occurrence_freq(con, table, main_col, other_col, year, output_dir, filter_col=None, filter_values=None):
    """Generates and saves the co-occurrence frequency between two columns, with optional filtering."""
    if not (filter_col and filter_values):
        print(f"  [WARN] Co-occurrence generation requires a filter column and values.")
        return

    values_str = ", ".join(["'" + str(v).replace("'", "''") + "'" if isinstance(v, str) else str(v) for v in filter_values])
    query = f'''
    SELECT "{main_col}" as col1, "{other_col}" as col2, COUNT(*) as cnt
    FROM {table}
    WHERE year = {year} AND "{filter_col}" IN ({values_str}) AND "{main_col}" IS NOT NULL AND "{other_col}" IS NOT NULL
    GROUP BY col1, col2
    ORDER BY col1, col2
    '''
    df = con.execute(query).fetchdf()
    if df.empty:
        print(f"  [WARN] No co-occurrence data found for '{main_col}' and '{other_col}' in {year} with given filters.")
        return

    df = df[df['cnt'] > 2]
    if df.empty:
        print(f"  [INFO] No co-occurrence data with count > 2 for year {year} and given sample.")
        return

    df['freq'] = df.groupby('col1')['cnt'].transform(lambda x: x / x.sum()).round(6)
    df = df[['col1', 'col2', 'cnt', 'freq']]
    
    out_path = os.path.join(output_dir, f"{main_col}_{other_col}_co_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Generated co-occurrence frequency file: {out_path}")


def get_freq_distribution(con, table, column, year, filter_col=None, filter_values=None):
    """Helper to get frequency distribution for a column with optional filtering."""
    query = f'SELECT CAST("{column}" AS TEXT) AS col FROM {table} WHERE year = {year} AND "{column}" IS NOT NULL'
    if filter_col and filter_values:
        values_str = ", ".join(["'" + str(v).replace("'", "''") + "'" if isinstance(v, str) else str(v) for v in filter_values])
        query += f' AND "{filter_col}" IN ({values_str})'
    
    df = con.execute(query).fetchdf()
    if df.empty:
        return [], np.array([])
        
    df['col'] = df['col'].str.strip()
    freq = df['col'].value_counts().sort_index()
    return freq.index.tolist(), freq.values / freq.values.sum()

def generate_join_freq(con, t1_config, t2_config, year, output_dir):
    """Generates a frequency file by joining two table configurations."""
    labels_t1, freq_t1 = get_freq_distribution(con, **t1_config, year=year)
    labels_t2, freq_t2 = get_freq_distribution(con, **t2_config, year=year)

    if not labels_t1 or not labels_t2:
        print(f"  [WARN] Not enough data for join frequency between {t1_config['table']} and {t2_config['table']} for year {year}.")
        return

    freq_dict1 = dict(zip(labels_t1, freq_t1))
    freq_dict2 = dict(zip(labels_t2, freq_t2))
    
    common_labels = sorted(set(labels_t1).intersection(labels_t2))

    output_file = os.path.join(output_dir, f"{t2_config['column']}_freq_{year}.txt")
    with open(output_file, "w") as f:
        for l in common_labels:
            f1 = freq_dict1.get(l, 0)
            f2 = freq_dict2.get(l, 0)
            f.write(f"{l}\t{f1}\t{f2}\n")
    print(f"  Generated join frequency file: {output_file}")


def main():
    # --- Configuration ---
    db_path = "dataset/chicago_data.db"
    years = [2018, 2019, 2020, 2021, 2022]
    sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    base_output_dir = "dataset/frequency_weighted"

    table1 = "crimes"
    table2 = "taxi"
    
    where_col = "District"
    join_col1 = "Community Area"
    join_col2 = "Pickup Community Area"

    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    for year in years:
        print(f"\n[Processing Year: {year}]")
        
        # Get district distribution for weighted sampling
        districts, probs = get_freq_distribution(con, table=table1, column=where_col, year=year)

        if not districts:
            print(f"  No districts found in {year}. Skipping.")
            continue
        
        for rate in sample_rates:
            sample_size = int(len(districts) * rate)
            # Weighted sampling: districts with higher probability are more likely to be chosen
            sampled_districts = np.random.choice(districts, size=sample_size, replace=False, p=probs).tolist()
            
            rate_str = f"crimes(di)-taxi-pickup-{int(rate*100)}%"
            output_dir_for_rate = os.path.join(base_output_dir, rate_str)
            os.makedirs(output_dir_for_rate, exist_ok=True)
            
            print(f"\n  [Rate: {int(rate*100)}% ({sample_size} districts) | Output: {output_dir_for_rate}]")

            # 1. District_freq_{year}.txt (from sampled crimes)
            generate_single_column_freq(con, table1, where_col, year, output_dir_for_rate, 
                                        filter_col=where_col, filter_values=sampled_districts)

            # 2. District_Community Area_co_{year}.csv (from sampled crimes)
            generate_co_occurrence_freq(con, table1, where_col, join_col1, year, output_dir_for_rate, 
                                        filter_col=where_col, filter_values=sampled_districts)
                                        
            # 3. Community Area_District_co_{year}.csv (from sampled crimes)
            generate_co_occurrence_freq(con, table1, join_col1, where_col, year, output_dir_for_rate, 
                                        filter_col=where_col, filter_values=sampled_districts)
            
            # 4. Community Area_freq_{year}.txt (join between all rideshares and sampled crimes)
            rideshares_config = {
                "table": table2,
                "column": join_col2
            }
            crimes_config_sampled = {
                "table": table1,
                "column": join_col1,
                "filter_col": where_col,
                "filter_values": sampled_districts
            }
            generate_join_freq(con, rideshares_config, crimes_config_sampled, year, output_dir_for_rate)

    con.close()
    print("\n--- Script finished successfully! ---")

if __name__ == "__main__":
    main()
