sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack.run_attack import step1, setup_logger

def log_experiment_to_run_log(experiment_name, params, dataset_paths, status="Success", result_file=None):
    """Log experiment information to run.log file"""
    run_log_path = os.path.join("output_result", "run.log")
        os.makedirs(os.path.dirname(run_log_path), exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.exists(run_log_path):
        with open(run_log_path, 'w', encoding='utf-8') as f:
            f.write("Experiment Run Log - Attack Framework\n")
            f.write(f"Creation Time: {current_time}\n")
            f.write(f"Log File: {os.path.abspath(run_log_path)}\n")
            f.write("================================================================================\n\n")
    
    # Add new experiment record
    with open(run_log_path, 'a', encoding='utf-8') as f:
        f.write(f"Experiment Record: {experiment_name}\n")
        f.write(f"Experiment Time: {current_time}\n")
        f.write("Experiment Parameters:\n")
        for key, value in params.items():
            f.write(f"  - {key}: {value}\n")
        f.write("\nDataset Paths:\n")
        for key, value in dataset_paths.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nExperiment Status: {status}\n")
        if result_file:
            f.write(f"Result File: {result_file}\n")
        f.write("\n" + "="*80 + "\n\n")

def run_batch_experiments():
    """Run batch attack experiments and save results to CSV"""
    
    # Set up batch experiment logger
    logger = setup_logger(log_dir="logs/batch_logs")

    obs_years = [2019, 2020, 2021, 2022]
    aux_year = 2018
    attack_name = "ours"
    table1 = "crimes"
    where_name="Community Area"
    join_name="Beat"
    where_tau=0.005
    where_mu=0.05
    join_tau=0.00003
    join_mu=0.05
    co_tau=0.05
    T_max=4
    reg=0.0001
 
    datasets_config = {
        # "crimes(di)-rideshares-dropoff": {"where_name": "District", "join_name": "Community Area"},
        # "crimes(di)-rideshares-pickup": {"where_name": "District", "join_name": "Community Area"},
        # "crimes(di)-taxi-dropoff": {"where_name": "District", "join_name": "Community Area"},
        # "crimes(di)-taxi-pickup": {"where_name": "District", "join_name": "Community Area"},
        "crimes(ca)-crashes-Beat": {"where_name": "Community Area", "join_name": "Beat"},
    }
    sample_types = ["random", "weighted"]
    sample_sizes = [8, 16, 24, 32, 40, 47, 55, 63, 71, 79, 95, 111, 126, 134, 142]
    #sample_sizes = [4, 8, 12, 16, 20, 24, 27, 31, 35, 39, 46, 54, 62, 70] # Second round, number of missing values for 60%, 70%, 80%, 85%, 90%
    sample_columns = [1, 2]

    logger.info("Starting batch attack experiments")
    logger.info("Batch experiment parameter configuration:")
    logger.info(f"Observation years: {obs_years}")
    logger.info(f"Auxiliary year: {aux_year}")
    logger.info(f"where_tau: {where_tau}, where_mu: {where_mu}")
    logger.info(f"join_tau: {join_tau}, join_mu: {join_mu}")
    logger.info(f"co_tau: {co_tau}, T_max: {T_max}, reg: {reg}")
    

    logger.info("Batch experiment variables:")
    logger.info(f"Datasets: {list(datasets_config.keys())}")
    logger.info(f"Sample types: {sample_types}")
    logger.info(f"Sample sizes: {sample_sizes}")
    logger.info(f"Sample columns: {sample_columns}")
    logger.info(f"Total experiment combinations: {len(datasets_config) * len(sample_types) * len(sample_sizes) * len(sample_columns)}")
    
    # Create output directory
    results_dir = "output_result"  # Changed to correct directory name
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info(f"Created output directory: {results_dir}")
    
    # Iterate through all parameter combinations
    experiment_count = 0
    total_experiments = len(datasets_config) * len(sample_types) * len(sample_sizes) * len(sample_columns)
    
    for dataset_name, sample_type, sample_size, sample_col in product(datasets_config.keys(), sample_types, sample_sizes, sample_columns):
        experiment_count += 1
        
        # Get column name configuration for the current dataset
        dataset_config = datasets_config[dataset_name]
        where_name = dataset_config["where_name"]
        join_name = dataset_config["join_name"]
        
        logger.info(f"\n{'='*100}")
        logger.info(f"Experiment {experiment_count}/{total_experiments}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Sampling parameters: {sample_type}, size={sample_size}, col={sample_col}")
        logger.info(f"Column name configuration: where={where_name}, join={join_name}")
        logger.info(f"{'='*100}")
        
        print(f"\n{'='*80}")
        print(f"Starting experiment {experiment_count}/{total_experiments}: {dataset_name}, {sample_type}, size={sample_size}, col={sample_col}")
        print(f"Column name configuration: where={where_name}, join={join_name}")
        print(f"{'='*80}")
        
        # Construct output CSV filename
        # Extract category name: from "crimes(di)-rideshares-dropoff" extract "rideshares-dropoff"
        category_parts = dataset_name.split("-", 1)  # Split into ["crimes(di)", "rideshares-dropoff"]
        if len(category_parts) > 1:
            category = category_parts[1]  # "rideshares-dropoff"
        else:
            category = dataset_name
        
        # Filename format: results_rideshares_dropoff_weighted_size4_col1.csv
        output_file = f"{results_dir}/results_{category.replace('-', '_')}_{sample_type}_size{sample_size}_col{sample_col}.csv"
        
        # Check if dataset paths exist
        base_dir = "dataset/sample-for-our"
        aux_dataset_path = os.path.join(base_dir, "aux_dataset", dataset_name)
        target_dataset_path = os.path.join(base_dir, f"sample-{sample_col}-column", f"{dataset_name}-{sample_type}")
        
        logger.info(f"Dataset path check:")
        logger.info(f"Auxiliary dataset: {aux_dataset_path}")
        logger.info(f"Target dataset: {target_dataset_path}")
        
        if not os.path.exists(aux_dataset_path):
            logger.warning(f"Auxiliary dataset does not exist, skipping experiment: {aux_dataset_path}")
            print(f" Auxiliary dataset does not exist, skipping: {aux_dataset_path}")
            continue
        if not os.path.exists(target_dataset_path):
            logger.warning(f" Target dataset does not exist, skipping experiment: {target_dataset_path}")
            print(f"Target dataset does not exist, skipping: {target_dataset_path}")
            continue
            
        logger.info(" Dataset path validation passed")
        
        # Attack parameter settings
        attack_params = {
            "aux_year": aux_year,
            "table1_name": table1,
            "table1_where_name": where_name, 
            "table1_join_name": join_name,   
            "table1_where_tau": where_tau,
            "table1_where_mu": where_mu,
            "table1_join_tau": join_tau,
            "table1_join_mu": join_mu,
            "co_tau": co_tau,
            "T_max": T_max,
            "reg": reg,
            "dataset_name": dataset_name
        }
        
        results = []
        for year in obs_years:
            output_csv = f"anchors_{table1}_{year}.csv"
            attack_params["output_csv"] = output_csv
            
            try:
                total_start_time = time.time()
                result_where, result_join, acc_where, acc_join, where_rscore, join_rscore, attack_duration, where_mapping_count, join_mapping_count = step1(
                    attack_params, year, sample_type, sample_size, logger=logger
                )
                total_end_time = time.time()
                total_runtime = total_end_time - total_start_time
                where_rscore_percent = where_rscore * 100
                where_vscore_percent = acc_where * 100
                join_rscore_percent = join_rscore * 100
                join_vscore_percent = acc_join * 100
                runtime_seconds = attack_duration
                result_record = {
                    'year': year,
                    'category': category,
                    'algorithm': 'our',
                    'sample_type': sample_type,
                    'sample_size': sample_size,
                    'sample_columns': sample_col,
                    'where_rscore_percent': where_rscore_percent,
                    'where_vscore_percent': where_vscore_percent,
                    'where_mapping_count': where_mapping_count,
                    'join_rscore_percent': join_rscore_percent,
                    'join_vscore_percent': join_vscore_percent,
                    'join_mapping_count': join_mapping_count,
                    'runtime_seconds': runtime_seconds
                }
                results.append(result_record)
                print(f"Attack for year {year} completed:")
                print(f"where: rscore={where_rscore_percent:.2f}%, vscore={where_vscore_percent:.2f}%")
                print(f"join:  rscore={join_rscore_percent:.2f}%, vscore={join_vscore_percent:.2f}%")
                print(f"Time: {runtime_seconds:.4f}s")
            except Exception as e:
                logger.error(f"Attack for year {year} failed: {str(e)}")
                import traceback
                logger.error(f"Error details: {traceback.format_exc()}")
                print(f"Attack for year {year} failed: {str(e)}")
                result_record = {
                    'year': year,
                    'category': category,
                    'algorithm': 'our',
                    'sample_type': sample_type,
                    'sample_size': sample_size,
                    'sample_columns': sample_col,
                    'where_rscore_percent': 0.0,
                    'where_vscore_percent': 0.0,
                    'where_mapping_count': 0,
                    'join_rscore_percent': 0.0,
                    'join_vscore_percent': 0.0,
                    'join_mapping_count': 0,
                    'runtime_seconds': 0.0
                }
                results.append(result_record)
        
        # Save results to CSV file
        if results:
            fieldnames = ['year', 'category', 'algorithm', 'sample_type', 'sample_size', 'sample_columns', 
                         'where_rscore_percent', 'where_vscore_percent', 'where_mapping_count',
                         'join_rscore_percent', 'join_vscore_percent', 'join_mapping_count', 'runtime_seconds']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n Results saved to: {output_file}")
            
            # Log experiment to run.log
            experiment_name = f"{category.replace('-', '_')}_{sample_type}_size{sample_size}_col{sample_col}"
            params = {
                'table_name1': table1,
                'column2_name': join_name,
                'where_clause': 'di',
                'years': obs_years,
                'aux_year': aux_year,
                'table_name2': category.split('-')[0] if '-' in category else 'unknown',
                'column1_name': category.split('-')[1] if '-' in category else 'unknown',
                'sample_type': sample_type,
                'sample_size': sample_size,
                'sample_column': sample_col
            }
            
            # Construct full dataset path information (including all four file types)
            dataset_paths = {}
            
            # Auxiliary dataset file paths (for 2018)
            dataset_paths['Auxiliary where file'] = os.path.join(aux_dataset_path, f"{where_name}_freq_{aux_year}.txt")
            dataset_paths['Auxiliary join file'] = os.path.join(aux_dataset_path, f"{join_name}_freq_{aux_year}.txt")
            dataset_paths['Auxiliary where_join file'] = os.path.join(aux_dataset_path, f"{where_name}_{join_name}_co_{aux_year}.csv")
            dataset_paths['Auxiliary join_where file'] = os.path.join(aux_dataset_path, f"{join_name}_{where_name}_co_{aux_year}.csv")
            
            # Add target dataset paths for each year (observation data)
            for year in obs_years:
                # Frequency files for where and join
                if sample_col == 1:
                    where_file = f"{where_name}_freq_{year}_sample_{sample_size}.txt"
                    join_file = f"{join_name}_freq1_{year}_sample_{sample_size}.txt"
                else:
                    where_file = f"{where_name}_freq_{year}_sample_{sample_size}.txt"
                    join_file = f"{join_name}_freq_{year}_sample_{sample_size}.txt"
                
                # Co-occurrence files for where_join and join_where
                where_join_file = f"{where_name}_{join_name}_co_{year}_sample_{sample_size}.csv"
                join_where_file = f"{join_name}_{where_name}_co_{year}_sample_{sample_size}.csv"
                
                dataset_paths[f'Observation where file ({year})'] = os.path.join(target_dataset_path, where_file)
                dataset_paths[f'Observation join file ({year})'] = os.path.join(target_dataset_path, join_file)
                dataset_paths[f'Observation where_join file ({year})'] = os.path.join(target_dataset_path, where_join_file)
                dataset_paths[f'Observation join_where file ({year})'] = os.path.join(target_dataset_path, join_where_file)
            
            # Log to file
            result_filename = os.path.basename(output_file)
            log_experiment_to_run_log(experiment_name, params, dataset_paths, "Success", result_filename)
            
            # Print summary for this group of experiments
            avg_where_rscore = sum(r['where_rscore_percent'] for r in results) / len(results) if results else 0
            avg_where_vscore = sum(r['where_vscore_percent'] for r in results) / len(results) if results else 0
            avg_join_rscore = sum(r['join_rscore_percent'] for r in results) / len(results) if results else 0
            avg_join_vscore = sum(r['join_vscore_percent'] for r in results) / len(results) if results else 0
            avg_runtime = sum(r['runtime_seconds'] for r in results) / len(results) if results else 0
        else:
            print(f" No successful results, skipping CSV save")

def main():
    logger = setup_logger()
    logger.info("Starting batch attack experiments")
    logger.info(f"Experiment time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()
    try:
        run_batch_experiments()
    except KeyboardInterrupt:
        print(f"\n User interrupted experiment")
        logger.warning("User interrupted experiment")
    except Exception as e:
        print(f"\n Error during experiment: {str(e)}")
        logger.error(f"Error during experiment: {str(e)}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        traceback.print_exc()
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\nBatch experiments completed!")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    
    logger.info(f"Batch experiments completed")
    logger.info(f"Total duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    logger.info(f"Experiment results saved to: {os.path.join(os.getcwd(), 'output_result')}")

if __name__ == "__main__":
    main()
