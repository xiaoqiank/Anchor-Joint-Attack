"""
Batch script to run experiments with all parameter combinations
Iterate through the following parameters：
- sample_type: weighted, random
- sample_size: 2, 4, 6, 8, 10, 12, 14
- sample_column: 1, 2
- table_name2: pickup, dropoff

Support parallel execution to improve running speed
"""

import subprocess
import sys
import time
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading

# Global lock for protecting log file writes
log_lock = threading.Lock()

def run_experiment(params, result_dir, log_file_path=None):
    """Run a single experiment configuration"""
    exp_name = f"{params['table_name2']}_{params['column1_name']}_{params['sample_type']}_size{params['sample_size']}_col{params['sample_column']}"
    output_file = f"results_{exp_name}.csv"
    
    base_data_dir = "dataset/sample-for-STE/"
    aux_file_pattern = f"full/{params['table_name1']}({params['where_clause']})-{params['table_name2']}-{params['column1_name']}/{params['column2_name']}_freq_{params['aux_year']}.txt"
    aux_file_path = base_data_dir + aux_file_pattern
        
    # Target dataset path (may have multiple years)
    for year in params['years']:
        obs_file_pattern = f"sample-{params['sample_column']}-column/{params['table_name1']}({params['where_clause']})-{params['table_name2']}-{params['column1_name']}-{params['sample_type']}/{params['column2_name']}_freq_{year}_sample_{params['sample_size']}.txt"
        obs_file_path = base_data_dir + obs_file_pattern
        print(f"[Thread {threading.current_thread().name}] Target dataset path ({year}): {obs_file_path}")
    
    # Construct command line arguments
    cmd = [
        'python3', 'test_sample_cross_driver.py',
        '--table_name1', params['table_name1'],
        '--table_name2', params['table_name2'],
        '--column1_name', params['column1_name'],
        '--column2_name', params['column2_name'],
        '--sample_size', str(params['sample_size']),
        '--sample_column', str(params['sample_column']),
        '--sample_type', params['sample_type'],
        '--where_clause', params['where_clause'],
        '--aux_year', str(params['aux_year']),
        '--output_file', output_file,
        '--result_dir', result_dir
    ]
    
    # Add years parameter
    cmd.extend(['--years'] + [str(year) for year in params['years']])
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # No timeout limit
        duration = time.time() - start_time
        success = result.returncode == 0
        if log_file_path:
            with log_lock:
                with open(log_file_path, 'a', encoding='utf-8') as log:
                    log.write(f"\nExperiment: {exp_name}\n")
                    log.write(f"Parameters: {params}\n")
                    log.write(f"Auxiliary dataset: {aux_file_path}\n")
                    for year in params['years']:
                        obs_file_pattern = f"sample-{params['sample_column']}-column/{params['table_name1']}({params['where_clause']})-{params['table_name2']}-{params['column1_name']}-{params['sample_type']}/{params['column2_name']}_freq_{year}_sample_{params['sample_size']}.txt"
                        obs_file_path = base_data_dir + obs_file_pattern
                        log.write(f"Target dataset ({year}): {obs_file_path}\n")
                    
                    log.write(f"Status: {'Success' if success else 'Failure'} (Duration: {duration:.2f}s)\n")
                    if not success and result.stderr:
                        log.write(f"Error: {result.stderr}\n")
                    log.write("=" * 40 + "\n")
        
        return {
            'exp_name': exp_name,
            'success': success,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'params': params
        }
        
    except Exception as e:
        duration = time.time() - start_time
        if log_file_path:
            with log_lock:
                with open(log_file_path, 'a', encoding='utf-8') as log:
                    log.write(f"\nExperiment: {exp_name}\n")
                    log.write(f"Parameters: {params}\n")
                    base_data_dir = "dataset/sample-for-STE/"
                    aux_file_pattern = f"full/{params['table_name1']}({params['where_clause']})-{params['table_name2']}-{params['column1_name']}/{params['column2_name']}_freq_{params['aux_year']}.txt"
                    aux_file_path = base_data_dir + aux_file_pattern
                    log.write(f"Auxiliary dataset: {aux_file_path}\n")
                    for year in params['years']:
                        obs_file_pattern = f"sample-{params['sample_column']}-column/{params['table_name1']}({params['where_clause']})-{params['table_name2']}-{params['column1_name']}-{params['sample_type']}/{params['column2_name']}_freq_{year}_sample_{params['sample_size']}.txt"
                        obs_file_path = base_data_dir + obs_file_pattern
                        log.write(f"Target dataset ({year}): {obs_file_path}\n")
                    
                    log.write(f"Status: Exception (Duration: {duration:.2f}s)\n")
                    log.write(f"Exception: {str(e)}\n")
                    log.write("=" * 40 + "\n")
        
        return {
            'exp_name': exp_name,
            'success': False,
            'duration': duration,
            'stdout': "",
            'stderr': str(e),
            'params': params
        }

def main():
    required_files = ['sample_cross_driver.py', 'greedy.py', 'split.py', 'genetic.py']
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f" Missing required files: {missing_files}")
        return 1
    else:
        print(" All required files exist")
    
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    max_workers = 1  
    

    #parameters
    base_params = {
        'table_name1': 'crimes',
        'column2_name': 'Community Area',
        'where_clause': 'di',
        'years': [2019, 2020, 2021, 2022],
        'aux_year': 2018
    }
    table_configs = [
        {'table_name2': 'taxi', 'column1_name': 'pickup'},
        {'table_name2': 'taxi', 'column1_name': 'dropoff'},
        {'table_name2': 'rideshares', 'column1_name': 'pickup'},
        {'table_name2': 'rideshares', 'column1_name': 'dropoff'}
    ]
    
    sample_types = ['weighted', 'random']
    sample_sizes = [4, 8, 12, 16, 20, 24, 27, 31, 35, 39, 46, 54, 62, 70]
    sample_columns = [1, 2]

   # New parameter configuration - based on provided file paths
    # base_params = {
    #     'table_name1': 'crimes',
    #     'column2_name': 'Beat',
    #     'where_clause': 'ca',
    #     'years': [2019,2020,2021,2022],
    #     'aux_year': 2018
    # }
    #sample_sizes = [8, 16, 24, 32, 40, 47, 55, 63, 71, 79, 95, 111, 126, 134, 142]
    # table_configs = [
    #     {'table_name2': 'crashes', 'column1_name': 'Beat'}
    # ]




    # Generate all parameter combinations
    all_experiments = []
    for table_config in table_configs:
        for sample_type in sample_types:
            for sample_size in sample_sizes:
                for sample_column in sample_columns:
                    params = {**base_params, **table_config}
                    params.update({
                        'sample_type': sample_type,
                        'sample_size': sample_size,
                        'sample_column': sample_column
                    })
                    all_experiments.append(params)
    
    total_experiments = len(all_experiments)

    result_dir = "output_result"
    os.makedirs(result_dir, exist_ok=True)
    n
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_run_dir = os.path.join(result_dir, f"batch_run_{timestamp}")
    os.makedirs(batch_run_dir, exist_ok=True)
    log_file = os.path.join(batch_run_dir, "batch_run.log")
    
    # Initialize log file
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Batch experiment run log (parallel version)\n")
        log.write(f"Start time: {datetime.now()}\n")
        log.write(f"Total experiments: {total_experiments}\n")
        log.write(f"Parallel worker processes: {max_workers}\n")
        log.write("=" * 80 + "\n")
    
    print(f"  Starting parallel execution of experiments...")
    print(f" Progress will be updated in real time...")
    
    successful_runs = 0
    failed_runs = 0
    completed = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(run_experiment, params, batch_run_dir, log_file): params 
            for params in all_experiments
        }
        

        for future in as_completed(future_to_params):
            completed += 1
            result = future.result()
            
            if result['success']:
                successful_runs += 1
                print(f"[{completed}/{total_experiments}] {result['exp_name']} (Duration: {result['duration']:.2f}s)")
            else:
                failed_runs += 1
                print(f"[{completed}/{total_experiments}] {result['exp_name']} (Duration: {result['duration']:.2f}s)")
                if result['stderr']:
                    print(f"Error: {result['stderr'][:100]}...")
            
            # Display progress
            progress = (completed / total_experiments) * 100
            elapsed_time = time.time() - start_time
            avg_time_per_task = elapsed_time / completed
            estimated_remaining = avg_time_per_task * (total_experiments - completed) / max_workers  # Consider parallel factor
            
            print(f"Progress: {progress:.1f}% | Success: {successful_runs} | Failure: {failed_runs} | "
                  f"Estimated remaining: {estimated_remaining/60:.1f} minutes")
            print("-" * 50)
    total_time = time.time() - start_time

    
    result_files = [f for f in os.listdir(batch_run_dir) if f.startswith('results_') and f.endswith('.csv')]
    if result_files:
        print(f"{len(result_files)} result CSV files were generated in {batch_run_dir}/")
        print(" Result file list:")
        for i, file in enumerate(result_files[:10], 1):  
            file_path = os.path.join(batch_run_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  {i}. {file} ({file_size} bytes)")
        if len(result_files) > 10:
            print(f"  ... and {len(result_files) - 10} more files")
    else:
        print("Warning: No result CSV files found")

    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"\nBatch run complete!\n")
        log.write(f"End time: {datetime.now()}\n")
        log.write(f"Total duration: {total_time/60:.1f} minutes\n")
        log.write(f"Success: {successful_runs}/{total_experiments}\n")
        log.write(f"Failure: {failed_runs}/{total_experiments}\n")
        log.write(f"Parallel worker processes: {max_workers}\n")
        log.write("=" * 80 + "\n")
    
    if failed_runs > 0:
        print(f"\n  Warning: {failed_runs} experiments failed, please check the log file for details.")
        return 1
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nUser interrupted the batch run")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during the batch run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
