import os
import shutil
import random
import numpy as np
import pandas as pd


def weighted_sample_by_value(data, column_idx, sample_count):
    """
    Weighted sampling by value, the smaller the value, the more likely it is to be selected.
    """
    values = data[:, column_idx].astype(float)
    # Calculate weights: the smaller the value, the greater the weight (use reciprocal to avoid division by zero)
    weights = 1.0 / (values + 1e-6)
    # Normalize weights
    weights = weights / weights.sum()
    
    # Select indices based on weights
    indices = np.random.choice(len(data), size=min(sample_count, len(data)), 
                              replace=False, p=weights)
    return indices


def random_sample(data, sample_count):
    """
    Random sampling.
    """
    indices = np.random.choice(len(data), size=min(sample_count, len(data)), 
                              replace=False)
    return indices


def sample_single_column(input_file, output_file, column_idx, sample_count, strategy):
    """
    Sample a single column.
    column_idx: 1 for the second column, 2 for the third column
    """
    # Read data
    data = np.loadtxt(input_file, delimiter='\t')
    
    # Copy data
    sampled_data = data.copy()
    
    # Select rows to be zeroed according to the policy
    if strategy == 'random':
        indices = random_sample(data, sample_count)
    elif strategy == 'weighted':
        indices = weighted_sample_by_value(data, column_idx, sample_count)
    
    # Set the specified column of the selected row to zero
    sampled_data[indices, column_idx] = 0
    
    # Save the results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, sampled_data, delimiter='\t', fmt='%d')


def sample_two_columns(input_file, output_file, sample_count, strategy):
    """
    Sample two columns at the same time, allocate the number of samples equally, and not on the same row.
    """
    # Read data
    data = np.loadtxt(input_file, delimiter='\t')
    
    # Copy data
    sampled_data = data.copy()
    
    # Calculate the number of samples for each column
    col2_count = sample_count // 2
    col3_count = sample_count - col2_count
    
    # Select the rows to be zeroed for the second column
    if strategy == 'random':
        col2_indices = random_sample(data, col2_count)
    elif strategy == 'weighted':
        col2_indices = weighted_sample_by_value(data, 1, col2_count)
    
    # Select the rows to be zeroed for the third column (excluding the rows that have been selected by the second column)
    available_indices = list(set(range(len(data))) - set(col2_indices))
    
    if len(available_indices) >= col3_count:
        if strategy == 'random':
            col3_indices = np.random.choice(available_indices, size=col3_count, replace=False)
        elif strategy == 'weighted':
            # Recalculate weights for available rows
            available_data = data[available_indices]
            values = available_data[:, 2].astype(float)
            weights = 1.0 / (values + 1e-6)
            weights = weights / weights.sum()
            selected_available_indices = np.random.choice(len(available_indices), 
                                                        size=col3_count, 
                                                        replace=False, p=weights)
            col3_indices = [available_indices[i] for i in selected_available_indices]
    else:
        col3_indices = available_indices
    
    # Set to zero
    sampled_data[col2_indices, 1] = 0  # Set the second column to zero
    sampled_data[col3_indices, 2] = 0   # Set the third column to zero
    
    # Save the results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, sampled_data, delimiter='\t', fmt='%d')


def main():
    # Base path
    base_input_path = os.path.join("..", "..", "full")
    base_output_path = os.path.join("..", "..")
    
    # Data directory
    datasets = [
        # "crimes(di)-taxi-pickup",
        # "crimes(di)-rideshares-dropoff",
        # "crimes(di)-rideshares-pickup", 
        # "crimes(di)-taxi-dropoff",
        "crimes(ca)-crashes-Beat"
    ]
    
    # Year
    years = [2019, 2020, 2021, 2022]
    
    # Number of samples
    #sample_counts = [4, 8, 12, 16, 20, 24, 27, 31, 35, 39,46, 54, 62, 70] #ca indeed 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 60%, 70%, 80%, 85%, 90%
    sample_counts = [8, 16, 24, 32, 40, 47, 55, 63, 71, 79, 95, 111, 126, 134, 142]  # beat
    # Sampling strategy
    strategies = ['random', 'weighted']
    # column_name="Community Area"
    column_name="Beat"

    print("Start processing single-column sampling...")
    print(f"datasets: {datasets}")
    print(f"years: {years}")
    print(f"sample_counts: {sample_counts}")
    print(f"strategies: {strategies}")
    print(f"column_name: {column_name}")

    # Single column sampling (third column)
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        for strategy in strategies:
            print(f"  strategy: {strategy}")
            for sample_count in sample_counts:
                output_dir = f"{base_output_path}/sample-1-column/{dataset}-{strategy}"
                
                for year in years:
                    input_file = f"{base_input_path}/{dataset}/{column_name}_freq_{year}.txt"
                    output_file = f"{output_dir}/{column_name}_freq_{year}_sample_{sample_count}.txt"
                    
                    if os.path.exists(input_file):
                        try:
                            sample_single_column(input_file, output_file, 2, sample_count, strategy)  # The index of the third column is 2
                            print(f"    Done: {os.path.basename(output_file)}")
                        except Exception as e:
                            print(f"    Error: {e}")
                    else:
                        print(f"    File does not exist: {input_file}")
    
    print("\nStart processing two-column sampling...")
    
    # Two-column sampling (second and third columns)
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        for strategy in strategies:
            print(f"  strategy: {strategy}")
            for sample_count in sample_counts:
                output_dir = f"{base_output_path}/sample-2-column/{dataset}-{strategy}"
                
                for year in years:
                    input_file = f"{base_input_path}/{dataset}/{column_name}_freq_{year}.txt"
                    output_file = f"{output_dir}/{column_name}_freq_{year}_sample_{sample_count}.txt"

                    if os.path.exists(input_file):
                        try:
                            sample_two_columns(input_file, output_file, sample_count, strategy)
                            print(f"    Done: {os.path.basename(output_file)}")
                        except Exception as e:
                            print(f"    Error: {e}")
                    else:
                        print(f"    File does not exist: {input_file}")
    
    print("\nAll sampling tasks are complete!")


if __name__ == "__main__":
    # Set random seed to ensure reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()

