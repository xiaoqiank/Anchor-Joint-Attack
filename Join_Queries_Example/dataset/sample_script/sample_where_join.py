import os, duckdb
import pandas as pd
import numpy as np

'''
Generate a script to count the joint frequency distribution of two columns in a table and output it to a file in the specified directory.
Processing logic:
1. Traverse all variable combinations
2. Read the corresponding frequency file, extract the value of the first column of the row where the third column is 0 as an exclusion list
3. Exclude these values in the database query to generate a joint statistical distribution of the remaining data
4. Output the formatted result file
'''

def read_exclude_list(file_path):
    """
    Read the frequency file and get the value of the first column of the row where the third column is 0
    """
    exclude_list = []
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3 and parts[2] == '0':
                        exclude_list.append(parts[0])
        print(f"Read {len(exclude_list)} excluded values from file {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return exclude_list

def generate_joint_freq_with_exclusion(db_path, table_name1, where_col, join_col, year, exclude_list, output_path):
    """
    Generate joint frequency distribution after excluding specified values
    """
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        # Build exclusion conditions
        exclude_condition = ""
        if exclude_list:
            exclude_values = "', '".join(str(val) for val in exclude_list)
            exclude_condition = f"AND \"{join_col}\" NOT IN ('{exclude_values}')"
        
        # SQL query
        query = f'''
        SELECT "{where_col}" as col1, "{join_col}" as col2, COUNT(*) as cnt
        FROM {table_name1}
        WHERE year = {year} AND "{where_col}" IS NOT NULL AND "{join_col}" IS NOT NULL
        {exclude_condition}
        GROUP BY col1, col2
        ORDER BY col1, col2
        '''
        
        print(f"Executing query, excluding {len(exclude_list)} values, year: {year}")
        df = con.execute(query).fetchdf()
        
        if len(df) > 0:
            # Outlier handling: delete rows with cnt<=2 as outliers or noise values
            original_count = len(df)
            df = df[df['cnt'] > 2]
            filtered_count = len(df)
            print(f"Outlier filtering: original record count {original_count}, filtered record count {filtered_count}, deleted {original_count - filtered_count} abnormal records")
            
            # Calculate conditional probability: for each col1 value, calculate freq = cnt / sum(cnt for same col1)
            df['freq'] = df.groupby('col1')['cnt'].transform(lambda x: (x / x.sum()).round(6))
            df = df[['col1', 'col2', 'cnt', 'freq']]
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Output to CSV file
            df.to_csv(output_path, index=False)
            print(f"Output file: {output_path}, record count: {len(df)}")
        else:
            print(f"Query result is empty, skip output: {output_path}")
            
        con.close()
        
    except Exception as e:
        print(f"Error processing database query: {e}")

def main():
    # Configure parameters
    base_dir = os.path.join("..", "..")
    db_path = os.path.join("..", "..", "..", "database_duckdb", "chicago_data.db")
    
    # Variable definition
    sample_columns = [1, 2]
    table_name1 = "crimes"
    table_name2_list = ["crashes"] 
    where_name = "ca"  
    where_Name = "Community Area"
    join_name1 = "Beat"
    join_name2_list = ["Beat"] #["dropoff", "pickup"] 
    years = [2019, 2020, 2021, 2022] 
    sample_types = ["random", "weighted"]
    sample_sizes = [8, 16, 24, 32, 40, 47, 55, 63, 71, 79, 95, 111, 126, 134, 142]


    
    # Iterate through all combinations
    for sample_col in sample_columns:
        for table_name2 in table_name2_list:
            for join_name2 in join_name2_list:
                for sample_type in sample_types:
                    for year in years:
                        for sample_size in sample_sizes:
                            # Build input file path
                            input_dir = f"sample-{sample_col}-column/{table_name1}({where_name})-{table_name2}-{join_name2}-{sample_type}"
                            input_file = f"{join_name1}_freq_{year}_sample_{sample_size}.txt"
                            input_path = os.path.join(base_dir, input_dir, input_file)
                            
                            # Read exclusion list
                            exclude_list = read_exclude_list(input_path)
                            
                            # Build output file path
                            output_file = f"{where_Name}_{join_name1}_co_{year}_sample_{sample_size}.csv"
                            output_path = os.path.join(base_dir, input_dir, output_file)
                            
                            # Generate joint frequency distribution
                            print(f"\nProcessing: {input_dir}/{input_file}")
                            generate_joint_freq_with_exclusion(
                                db_path, table_name1, where_Name, join_name1, 
                                year, exclude_list, output_path
                            )



def genco_st_freq(db_path,table_name,where_name,join_name,years,output_dir):
    """
    The original function is reserved for backup
    """
    con = duckdb.connect(db_path, read_only=True)
    for year in years:
        query = f'''
        SELECT "{where_name}" as col1, "{join_name}" as col2, COUNT(*) as cnt
        FROM {table_name}
        WHERE year = {year} AND "{where_name}" IS NOT NULL AND "{join_name}" IS NOT NULL
        GROUP BY col1, col2
        ORDER BY col1, col2
        '''
        df = con.execute(query).fetchdf()
        df = df[df['cnt'] > 2]
        df['freq'] = df.groupby('col1')['cnt'].transform(lambda x: x / x.sum()).round(6)
        df = df[['col1', 'col2', 'cnt', 'freq']]
        out_path = f"{output_dir}/{where_name}_{join_name}_co_{year}.csv"
        df.to_csv(out_path, index=False)
    con.close()


if __name__ == "__main__":
    main()