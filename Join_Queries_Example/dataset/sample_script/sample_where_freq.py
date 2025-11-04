# -*- coding: utf-8 -*-
# gensi_where_freq.py
import os, duckdb
import pandas as pd

def read_freq_file_and_get_zero_freq_values(file_path):
    """Read the frequency file and get the value of the first column of the row where the third column is 0"""
    zero_freq_values = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3 and float(parts[2]) == 0:
                        zero_freq_values.append(parts[0])
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")
    return zero_freq_values

def gensi_freq_with_exclusion(con, table, join_name, col, year, excluded_community_areas, output_file):
    """Generate the frequency distribution of the District column after excluding specific Community Areas"""
    # Build exclusion conditions
    exclusion_condition = ""
    if excluded_community_areas:
        excluded_values = "', '".join(excluded_community_areas)
        exclusion_condition = f"AND \"{join_name}\" NOT IN ('{excluded_values}')"
    
    query = f'''
    SELECT "{col}" AS col
    FROM {table}
    WHERE year = {year} AND "{col}" IS NOT NULL {exclusion_condition}
    '''
    
    try:
        df = con.execute(query).fetchdf()
        counts = df['col'].value_counts().sort_index()
        total = counts.sum()

        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            for value, count in counts.items():
                freq = count / total if total > 0 else 0
                f.write(f"{value}\t{count}\t{freq}\n")
    except Exception as e:
        print(f"Error processing query for {output_file}: {e}")

def main():
    print("Starting script execution...")
    
    # Define variables
    base_dir = os.path.join("..", "..")
    sample_columns = [1, 2]
    table_name1 = "crimes"
    table_name2_list = ["crashes"] 
    where_name = "ca"  
    where_Name = "Community Area"
    join_name_list = ["Beat"]  #["dropoff", "pickup"] Note that you need to define join_name1="Community Area"
    years = [2019, 2020, 2021, 2022]
    sample_types = ["random", "weighted"]
    #sample_sizes = [4, 8, 12, 16, 20, 24, 27, 31, 35, 39,46, 54, 62, 70] #ca indeed 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 60%, 70%, 80%, 85%, 90%
    sample_sizes = [8, 16, 24, 32, 40, 47, 55, 63, 71, 79, 95, 111, 126, 134, 142]  # beat

    
    
    print(f"Base directory: {base_dir}")
    print(f"Sample columns: {sample_columns}")
    print(f"Years: {years}")
    print(f"Sample sizes: {sample_sizes}")
    
    # Database connection
    db_path = os.path.join("..", "..", "..", "database_duckdb", "chicago_data.db")
    try:
        con = duckdb.connect(db_path, read_only=True)
        print(f"Successfully connected to database: {db_path}")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    processed_count = 0
    
    # Iterate through all combinations
    for sample_column in sample_columns:
        for table_name2 in table_name2_list:
            for join_name in join_name_list:
                for sample_type in sample_types:
                    for year in years:
                        for sample_size in sample_sizes:
                            # Construct input file path
                            input_file_path = os.path.join(
                                base_dir,
                                f"sample-{sample_column}-column",
                                f"{table_name1}({where_name})-{table_name2}-{join_name}-{sample_type}",
                                f"{join_name}_freq_{year}_sample_{sample_size}.txt"
                            )
                            
                            # Construct output file path
                            output_file_path = os.path.join(
                                base_dir,
                                f"sample-{sample_column}-column",
                                f"{table_name1}({where_name})-{table_name2}-{join_name}-{sample_type}",
                                f"{where_Name}_freq_{year}_sample_{sample_size}.txt"
                            )
                            
                            # Check if the input file exists
                            if not os.path.exists(input_file_path):
                                continue
                            
                            # Read the input file and get the Community Area value where the third column is 0
                            list_0 = read_freq_file_and_get_zero_freq_values(input_file_path)
                            
                            # Generate the frequency distribution of the District column, excluding the Community Area in list_0
                            print(f"Processing: {os.path.basename(input_file_path)}")
                            if list_0:
                                print(f"Excluded {join_name}: {list_0}")

                            gensi_freq_with_exclusion(con, table_name1, join_name, where_Name, year, list_0, output_file_path)
                            processed_count += 1
                            print(f"Completed: {os.path.basename(output_file_path)}\n")
    
    con.close()
    print(f"All files have been processed! A total of {processed_count} files have been processed.")

if __name__ == "__main__":
    main()

