# -*- coding: utf-8 -*-
import sys
import math
import time
import os
from pandas import read_csv
from greedy import greedy
from split import split  
from genetic import genetic  


eps = 0.001
# Output file will be dynamically set at runtime
output_file = None

def set_output_file(filepath):
    """Set output file path"""
    global output_file
    try:
        if output_file is not None:
            output_file.close()
        
        # Check if file already exists, if not, write header line
        file_exists = os.path.exists(filepath)
        output_file = open(filepath, 'a', encoding='utf-8')  # Append mode
        
        if not file_exists:
            # Write CSV header line, add sample_type column
            output_file.write('year,category,algorithm,sample_type,u1,u2,u3,n1,n2,m,N,rscore_percent,vscore_percent,runtime_seconds\n')
            output_file.flush()
        
        return output_file
    except Exception as e:
        print(f"Error: Unable to set output file {filepath}: {e}")
        return None

# returns the auxiliary distribution and the histogram for the observed data
def hists_from_files(exp_params):

    # Parse experiment parameters
    (exp_name, file_path_params) = exp_params
    (year, category_aus, category_obs) = exp_name
    
    # Read preprocessed txt files
    # Assume file_path_params contains two txt file paths: aux_file_path and obs_file_path
    if isinstance(file_path_params, dict):
        aux_file_path = file_path_params.get('aux_file', '')
        obs_file_path = file_path_params.get('obs_file', '')
    else:
        # If it's old format, try to construct file paths
        print("Warning: Using fallback file path construction")
        aux_file_path = f"dataset/sample-for-STE/aux_{category_aus}_{year}.txt"
        obs_file_path = f"dataset/sample-for-STE/obs_{category_obs}_{year}.txt"
    
    print(f"Reading auxiliary data from: {aux_file_path}")
    print(f"Reading observed data from: {obs_file_path}")
    
    # Read auxiliary data files
    aux_data = []
    try:
        with open(aux_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    parts = line.split('\t')  # Assume tab-separated
                    if len(parts) >= 3:
                        label = parts[0].strip()
                        freq1 = int(parts[1].strip())
                        freq2 = int(parts[2].strip())
                        aux_data.append((label, freq1, freq2))
    except FileNotFoundError:
        print(f"Error: Auxiliary file not found: {aux_file_path}")
        raise
    except Exception as e:
        print(f"Error reading auxiliary file: {e}")
        raise
    
    # Read observation data files
    obs_data = []
    try:
        with open(obs_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    parts = line.split('\t')  # Assume tab-separated
                    if len(parts) >= 3:
                        label = parts[0].strip()
                        freq1 = int(parts[1].strip())
                        freq2 = int(parts[2].strip())
                        obs_data.append((label, freq1, freq2))
    except FileNotFoundError:
        print(f"Error: Observed file not found: {obs_file_path}")
        raise
    except Exception as e:
        print(f"Error reading observed file: {e}")
        raise
    
    # Construct auxiliary distributions (need to normalize to probability distributions)
    total_aux1 = sum(freq1 for _, freq1, _ in aux_data)
    total_aux2 = sum(freq2 for _, _, freq2 in aux_data)
    
    aux_dist1 = []
    aux_dist2 = []
    
    for label, freq1, freq2 in aux_data:
        if total_aux1 > 0:
            aux_dist1.append((freq1 / total_aux1, label))  # (probability, label)
        if total_aux2 > 0:
            aux_dist2.append((freq2 / total_aux2, label))  # (probability, label)
    
    # Construct observation histograms (keep count format, including labels with frequency 0)
    obs_hist1 = [(freq1, label) for label, freq1, _ in obs_data]  # (count, label) - include 0 frequency
    obs_hist2 = [(freq2, label) for label, _, freq2 in obs_data]  # (count, label) - include 0 frequency
    
    print(f"Loaded {len(aux_dist1)} auxiliary distribution 1 entries")
    print(f"Loaded {len(aux_dist2)} auxiliary distribution 2 entries")
    print(f"Loaded {len(obs_hist1)} observed histogram 1 entries")
    print(f"Loaded {len(obs_hist2)} observed histogram 2 entries")
    
    return aux_dist1, aux_dist2, obs_hist1, obs_hist2


# assignment is a list of indices to guess
def get_scores(correct_vals,guessed_vals,obs_freqs):
    # change r-score to be the fraction of rows correctly
    # identified over the ones observed
    assert(len(correct_vals) == len(guessed_vals))
    assert(len(obs_freqs) == len(guessed_vals))
    rscore = 0
    vscore = 0
    M = len(guessed_vals)
    obs_rows = 0
    for i in range(M):
        f1, f2 = obs_freqs[i]
        obs_rows = obs_rows + f1 + f2
        if correct_vals[i] == guessed_vals[i]:
            rscore = rscore + f1 + f2
            vscore += 1
    
    # Prevent division by zero error
    if obs_rows == 0:
        rscore = 0
    else:
        rscore = rscore / obs_rows
    
    if M == 0:
        vscore = 0
    else:
        vscore = vscore / M
    
    return rscore, vscore

# runs is number of runs for random experiments and 0 for deterministic ones
def write_scores(exp_params,alg,sample_type,u1,u2,u3,n1,n2,m,N,rscore,vscore,run_time=None):
    # Fix parameter unpacking issue: now exp_params is in (exp_name, file_path_params) format
    (exp_name, _) = exp_params  # ignore file_path_params
    (year, category_aus, category_obs) = exp_name
    if run_time is not None:
        data = [year,category_obs,alg,sample_type,u1,u2,u3,n1,n2,m,N,100*rscore,100*vscore,run_time]
    else:
        data = [year,category_obs,alg,sample_type,u1,u2,u3,n1,n2,m,N,100*rscore,100*vscore]
    line = ','.join([str(d) for d in data])
    if output_file is not None:
        output_file.write(line + '\n')
        output_file.flush()
    else:
        print(line)



def run_attacks(exp_params, sample_type):
    '''
    Step 3: Generate statistics based on hists_from_files(exp_params)
    '''
    
    aux1, aux2, obs1, obs2 = hists_from_files(exp_params)
    '''
    Step 5: Remove commented code, no need to pad 0s since I've already done padding in data files
    '''
    obs1 = sorted(obs1,key=lambda x: x[0]) # make sure we have the sorted invariant
    N = len(aux1)
    # create list of frequencies
    obs_freqs = []
    correct_vals = []
    u1 = 0
    u2 = 0
    u3 = 0
    n1 = 0
    n2 = 0
    m = 0
    for (f1,v) in obs1:
        for (f2,v2) in obs2:
            if v == v2:
                if f1 > 0 and f2 > 0:
                    correct_vals.append(v)
                    obs_freqs.append((f1,f2))
                    n1 += 1
                    n2 += 1
                    m += 1
                elif f1 > 0 and f2 == 0:
                    u1 += f1
                    n1 += 1
                elif f2 > 0 and f1 == 0:
                    u2 += f2
                    n2 += 1
                else:
                    u3 += 1
                    
    print("u1:", u1, "u2:", u2, "u3:", u3, "n1:", n1, "n2:", n2, "m:", m, "N:", N)



    # test greedy - resume greedy algorithm execution
    start_time = time.time()
    f,U1,U2 = greedy(aux1,aux2,obs_freqs,u1,u2,eps)
    greedy_time = time.time() - start_time
    rscr, vscr = get_scores(correct_vals,f,obs_freqs)
    write_scores(exp_params,"greedy",sample_type,u1,u2,u3,n1,n2,m,N,rscr,vscr,greedy_time)

    # test genetic - Commented out other algorithms, only execute greedy
    start_time = time.time()
    f,U1,U2 = genetic(aux1,aux2,obs_freqs,u1,u2,eps)
    genetic_time = time.time() - start_time
    rscr, vscr = get_scores(correct_vals,f,obs_freqs)
    write_scores(exp_params,"genetic",sample_type,u1,u2,u3,n1,n2,m,N,rscr,vscr,genetic_time)

    # test split - Commented out other algorithms, only execute greedy
    start_time = time.time()
    f,U1,U2 = split(aux1,aux2,obs_freqs,u1,u2,eps)
    split_time = time.time() - start_time
    rscr, vscr = get_scores(correct_vals,f,obs_freqs)
    write_scores(exp_params,"split",sample_type,u1,u2,u3,n1,n2,m,N,rscr,vscr,split_time)

if __name__ == "__main__":
    import argparse
    import os
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run cross-column driver test experiment')
    parser.add_argument('--table_name1', default='crimes')
    parser.add_argument('--table_name2', default='taxi') 
    parser.add_argument('--column1_name', default='dropoff')
    parser.add_argument('--column2_name', default='Community Area')
    parser.add_argument('--sample_size', type=int, default=2)
    parser.add_argument('--sample_column', type=int, default=1)
    parser.add_argument('--sample_type', default='weighted', choices=['weighted', 'random'])
    parser.add_argument('--where_clause', default='di')
    parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022])
    parser.add_argument('--aux_year', type=int, default=2018)
    parser.add_argument('--output_file', help='Specify output CSV file path')
    parser.add_argument('--result_dir', help='Result folder path')
    args = parser.parse_args()
    
    # Check if command line arguments are provided (other than program name itself)
    import sys
    has_custom_args = len(sys.argv) > 1
    
    if not has_custom_args:
        print("Running experiment with default configuration...")
        print("Tip: You can use --help to view all available parameters")
        print()
    
    '''
    Construct file path based on input parameters:
    Parent directory: dataset/sample-for-STE/
    Sub-level file path: sample-{sample_column}-column/{table_name1}({where_clause})-{table_name2}-{column1_name}-{sample_type}/{column2_name}_freq_{year}_sample_{sample_size}.txt
    '''
    
    # Define experiment configuration parameters
    base_dir = "dataset/sample-for-STE/"
    
    # Set output file
    if args.output_file:
        # If output file is specified, use the specified file
        output_file_path = args.output_file
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            output_file_path = os.path.join(args.result_dir, os.path.basename(args.output_file))
        set_output_file(output_file_path)
    else:
        # Use default filename based on sample size
        default_filename = f"results_sample_size_{args.sample_size}.csv"
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            output_file_path = os.path.join(args.result_dir, default_filename)
        else:
            output_file_path = default_filename
        set_output_file(output_file_path)
    
    chicago_exps = []
    valid_exps = 0
    
    # Construct experiment parameters based on input parameters
    for year in args.years:
        category_obs = f"{args.column1_name}-{args.table_name1}-{args.table_name2}"
        
        # Auxiliary dataset path - use unsampled data from full directory
        aux_file_pattern = f"full/{args.table_name1}({args.where_clause})-{args.table_name2}-{args.column1_name}/{args.column2_name}_freq_{args.aux_year}.txt"
        aux_file_path = base_dir + aux_file_pattern
        
        # Target dataset path - use sampled data from sample directory
        obs_file_pattern = f"sample-{args.sample_column}-column/{args.table_name1}({args.where_clause})-{args.table_name2}-{args.column1_name}-{args.sample_type}/{args.column2_name}_freq_{year}_sample_{args.sample_size}.txt"
        obs_file_path = base_dir + obs_file_pattern
        
        # Check if files exist
        aux_exists = os.path.exists(aux_file_path)
        obs_exists = os.path.exists(obs_file_path)
        
        if aux_exists and obs_exists:
            # Construct experiment parameters
            category_aus = f"{args.column1_name}-{args.table_name1}-{args.table_name2}"  # auxiliary dataset category
            exp_name = (year, category_aus, category_obs)
            file_paths = {
                'aux_file': aux_file_path,
                'obs_file': obs_file_path
            }
            chicago_exps.append((exp_name, file_paths))
            valid_exps += 1
            print(f"✓ Found valid experiment files - year {year}")
        else:
            print(f"✗ Skip year {year} - files missing:")
            if not aux_exists:
                print(f"    Auxiliary data file does not exist: {aux_file_path}")
            if not obs_exists:
                print(f"    Observation data file does not exist: {obs_file_path}")
    
    print(f"\nFound total of {valid_exps} valid experiments")
    
    # If no experiment files found, exit
    if not chicago_exps:
        print("Error: No valid experiment files found")
        print("\nSuggest checking the following:")
        print("1. Confirm data files exist at specified paths")
        print("2. Check if filename format is correct")
        print("3. Try using different parameter combinations")
        exit(1)

    for exp in chicago_exps:
        print("Running experiment:")
        print("Experiment: {}".format(exp[0]))
        start_time = time.time()
        try:
            run_attacks(exp, args.sample_type)
            total_time = time.time() - start_time
            print("Total time for experiment {}: {:.4f} seconds".format(exp[0], total_time))
        except Exception as e:
            total_time = time.time() - start_time
            print("Error running experiment {}: {}".format(exp, e))
            print("Time before error: {:.4f} seconds".format(total_time))
            import traceback; traceback.print_exc()
        print("-" * 50)
