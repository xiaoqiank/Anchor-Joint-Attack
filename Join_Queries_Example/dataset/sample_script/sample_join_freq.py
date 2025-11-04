import os
import pandas as pd
import numpy as np

def process_frequency_files():
    """
    Process frequency files, convert frequencies to frequencies and save
    """
    # Define the value range of all variables
    base_dir = os.path.join("..", "..")
    sample_columns = [1, 2]
    table_name1 = "crimes"
    table_name2_list = ["crashes"] #["rideshares", "taxi"]
    where_name = "ca"  #"di"
    join_name_list = ["Beat"] #["dropoff", "pickup"] Note that you need to define join_name1="Community Area"
    years = [2019, 2020, 2021, 2022]
    sample_types = ["random", "weighted"]
    #sample_sizes = [4, 8, 12, 16, 20, 24, 27, 31, 35,] #ca 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 60%, 70%, 80%, 85%, 90%
    sample_sizes = [8, 16, 24, 32, 40, 47, 55, 63, 71, 79, 95, 111, 126, 134, 142]  # beat
    
    total_files = 0
    processed_files = 0
    error_files = 0
    
    # Iterate through all parameter combinations
    for sample_col in sample_columns:
        for table_name2 in table_name2_list:
            for join_name in join_name_list:
                for sample_type in sample_types:
                    for year in years:
                        for sample_size in sample_sizes:
                            # Build input file path
                            input_dir = f"sample-{sample_col}-column/{table_name1}({where_name})-{table_name2}-{join_name}-{sample_type}"
                            input_filename = f"{join_name}_freq_{year}_sample_{sample_size}.txt"
                            input_path = os.path.join(base_dir, input_dir, input_filename)
                            
                            # Build output file path
                            output_filename = f"{join_name}_freq1_{year}_sample_{sample_size}.txt"
                            output_path = os.path.join(base_dir, input_dir, output_filename)
                            
                            total_files += 1
                            
                            # Check if the input file exists
                            if not os.path.exists(input_path):
                                print(f"File does not exist: {input_path}")
                                error_files += 1
                                continue
                            
                            try:
                                # Read data
                                df = pd.read_csv(input_path, sep='\t', header=None, names=['attribute', 'freq1', 'freq2'])
                                
                                # Delete rows where the third column is 0
                                df_filtered = df[df['freq2'] != 0].copy()
                                
                                if len(df_filtered) == 0:
                                    print(f"Warning: After deleting rows where the third column is 0, the file is empty: {input_path}")
                                    error_files += 1
                                    continue
                                
                                # Normalization: convert frequency to frequency
                                sum_freq1 = df_filtered['freq1'].sum()
                                sum_freq2 = df_filtered['freq2'].sum()
                                
                                if sum_freq1 == 0 or sum_freq2 == 0:
                                    print(f"Warning: The sum of frequencies is 0: {input_path}")
                                    error_files += 1
                                    continue
                                
                                df_filtered['freq1'] = df_filtered['freq1'] / sum_freq1
                                df_filtered['freq2'] = df_filtered['freq2'] / sum_freq2
                                
                                # Make sure the output directory exists
                                output_dir = os.path.dirname(output_path)
                                os.makedirs(output_dir, exist_ok=True)
                                
                                # Save the results
                                df_filtered.to_csv(output_path, sep='\t', header=False, index=False, 
                                                 float_format='%.10f')
                                
                                processed_files += 1
                                if processed_files % 50 == 0:
                                    print(f"Processed {processed_files}/{total_files} files")
                                    
                            except Exception as e:
                                print(f"Error processing file {input_path}: {str(e)}")
                                error_files += 1
                                continue
    
    print(f"\nProcessing complete:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Error files: {error_files}")

if __name__ == "__main__":
    process_frequency_files()