#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jigsaw Attack Experimental Script for SPARCS Dataset - Procedure Code Attack
This script implements the Jigsaw attack for the CCSR Procedure Code on the SPARCS dataset.

Experimental Setup:
- Data: SPARCS data from 2018-2024
- Attack: Jigsaw attack
- Target Column: CCSR Procedure Code (target column to be recovered)
- Dependent Column: CCSR Diagnosis Code (used for constructing the co-occurrence matrix)
- Sampling: Frequency-weighted and random sampling on the Where column
- Ratios: 10%-100% sampling ratios
- Runs: Each sampling strategy will be run 3 times, and the results will be averaged
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from collections import Counter
from typing import Dict, List, Tuple, Set
import random
from tqdm import tqdm

# Adding the path for Jigsaw attack code (masked directory)
from jigsaw import JigsawAttacker

class JigsawSPARCSProcedureExperiment:
    def __init__(self, data_dir: str = "/path/to/your/data"):
        self.data_dir = data_dir  # Directory for SPARCS dataset
        self.years = [2018, 2019,2020,2021,2022,2023,2024]
        self.sampling_ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        self.sampling_strategies = ['frequency_weighted', 'random']
        self.num_runs = 3
        
        # Jigsaw attack parameters
        self.jigsaw_params = {
            'baseRec': 20,
            'confRec': 10,
            'refinespeed': 10,
            'alpha': 0.6,
            'beta': 0.2
        }
        
        # Results storage
        self.results = []
        
    def load_data(self, year: int) -> pd.DataFrame:
        """Load the dataset for the specified year"""
        file_path = os.path.join(self.data_dir, f"sparcs_{year}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file does not exist: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Loaded data for {year}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def create_co_occurrence_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create the co-occurrence matrix for Procedure Code (Join column) and Diagnosis Code (Where column)"""
        # Calculate frequencies and sort in descending order
        freq_map = df.groupby('CCSR Procedure Code').size().to_dict()
        procedure_codes = sorted(freq_map.keys(), key=lambda x: freq_map[x], reverse=True)
        
        freq_map1 = df.groupby('CCSR Diagnosis Code').size().to_dict()
        diagnosis_codes = sorted(freq_map1.keys(), key=lambda x: freq_map1[x], reverse=True)
        
        print(f"    Matrix created: Procedure codes={len(procedure_codes)}, Diagnosis codes={len(diagnosis_codes)}, Matrix dimensions={len(procedure_codes)}×{len(diagnosis_codes)}")
        
        # Create mappings for rows and columns
        row_map = {v: i for i, v in enumerate(procedure_codes)}
        col_map = {v: j for j, v in enumerate(diagnosis_codes)}
        
        # Create the matrix
        matrix = np.zeros((len(procedure_codes), len(diagnosis_codes)))
        
        # Fill the matrix
        for _, row in df.iterrows():
            val1 = str(row['CCSR Procedure Code'])
            val2 = str(row['CCSR Diagnosis Code'])
            if val1 in row_map and val2 in col_map:
                matrix[row_map[val1], col_map[val2]] += 1
        
        # Calculate matrix statistics
        non_zero_entries = np.count_nonzero(matrix)
        total_entries = matrix.size
        sparsity = 1 - (non_zero_entries / total_entries)
        print(f"    Matrix statistics: Non-zero entries={non_zero_entries}, Total entries={total_entries}, Sparsity={sparsity:.3f}")
        
        return matrix, procedure_codes, diagnosis_codes
    
    def frequency_weighted_sampling(self, df: pd.DataFrame, ratio: float, random_seed: int = None) -> pd.DataFrame:
        """Frequency-weighted sampling based on Diagnosis Code (Where column)"""
        if random_seed is None:
            random_seed = np.random.randint(0, 10000)
        
        # Calculate the frequency of diagnosis codes (Where column)
        diag_freq = df['CCSR Diagnosis Code'].value_counts()
        total_unique_codes = len(diag_freq)
        sample_size = int(total_unique_codes * ratio)
        
        print(f"    Frequency-weighted sampling: Total diagnosis codes={total_unique_codes}, Sampling ratio={ratio:.1%}, Sampled codes={sample_size}")
        
        # Use numpy.random.choice for weighted sampling without replacement
        np.random.seed(random_seed)
        labels = diag_freq.index.tolist()
        weights = diag_freq.values.astype(float)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted sampling without replacement
        sampled_codes = set(np.random.choice(labels, size=sample_size, replace=False, p=weights))
        
        # Filter data to keep only the sampled diagnosis codes
        sampled_df = df[df['CCSR Diagnosis Code'].isin(sampled_codes)]
        
        print(f"    Sampled data: Rows={len(sampled_df)}, Unique diagnosis codes={len(sampled_codes)}")
        
        return sampled_df
    
    def random_sampling(self, df: pd.DataFrame, ratio: float, random_seed: int = None) -> pd.DataFrame:
        """Random sampling based on Diagnosis Code (Where column)"""
        if random_seed is None:
            random_seed = np.random.randint(0, 10000)
        
        # Get all unique diagnosis codes (Where column)
        unique_codes = df['CCSR Diagnosis Code'].unique()
        total_unique_codes = len(unique_codes)
        sample_size = int(total_unique_codes * ratio)
        
        print(f"    Random sampling: Total diagnosis codes={total_unique_codes}, Sampling ratio={ratio:.1%}, Sampled codes={sample_size}, Random seed={random_seed}")
        
        # Randomly sample diagnosis codes
        np.random.seed(random_seed)
        sampled_codes = set(np.random.choice(unique_codes, size=sample_size, replace=False))
        
        # Filter data to keep only the sampled diagnosis codes
        sampled_df = df[df['CCSR Diagnosis Code'].isin(sampled_codes)]
        
        print(f"    Sampled data: Rows={len(sampled_df)}, Unique diagnosis codes={len(sampled_codes)}")
        
        return sampled_df
    
    def calculate_accuracy_metrics(self, true_mapping: Dict[int, int], predicted_mapping: Dict[int, int], 
                                 target_frequencies: Dict[int, int] = None) -> Tuple[float, float]:
        """Calculate value recovery rate and row recovery rate"""
        if not predicted_mapping:
            return 0.0, 0.0
        
        # Value recovery rate: Percentage of correctly recovered values
        correct_values = 0
        total_values = len(predicted_mapping)
        
        for target_idx, predicted_idx in predicted_mapping.items():
            if target_idx in true_mapping and true_mapping[target_idx] == predicted_idx:
                correct_values += 1
        
        value_recovery_rate = correct_values / total_values if total_values > 0 else 0.0
        
        # Row recovery rate: Correctly recovered rows (weighted by frequency) / Total recovered rows (weighted by frequency)
        if target_frequencies is None:
            # If no frequency information, use simple count of matching pairs
            row_recovery_rate = correct_values / len(predicted_mapping) if len(predicted_mapping) > 0 else 0.0
        else:
            # Calculate row recovery rate using frequency information
            correct_rows = 0
            total_rows = 0
            
            for target_idx, predicted_idx in predicted_mapping.items():
                frequency = target_frequencies.get(target_idx, 1)  # Get frequency of this value
                total_rows += frequency
                
                if target_idx in true_mapping and true_mapping[target_idx] == predicted_idx:
                    correct_rows += frequency
            
            row_recovery_rate = correct_rows / total_rows if total_rows > 0 else 0.0
        
        return value_recovery_rate, row_recovery_rate
    
    def run_jigsaw_attack(self, sim_M: np.ndarray, real_M: np.ndarray) -> Tuple[Dict[int, int], float]:
        """Run the Jigsaw attack and return the mapping and execution time"""
        start_time = time.time()
        
        # Initialize the attacker
        attacker = JigsawAttacker(
            sim_M=sim_M, 
            real_M=real_M,
            baseRec=self.jigsaw_params['baseRec'],
            confRec=self.jigsaw_params['confRec'],
            refinespeed=self.jigsaw_params['refinespeed'],
            alpha=self.jigsaw_params['alpha'],
            beta=self.jigsaw_params['beta']
        )
        
        # Run the attack steps
        attacker.attack_step_1()
        attacker.attack_step_2()
        mapping_dict = attacker.attack_step_3()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return mapping_dict, execution_time
    
    def create_true_mapping(self, aux_codes: List[str], target_codes: List[str]) -> Dict[int, int]:
        """Create the true mapping based on auxiliary and target codes"""
        true_mapping = {}
        
        # Find common codes and create mappings
        aux_set = set(aux_codes)
        target_set = set(target_codes)
        common_codes = aux_set.intersection(target_set)
        
        print(f"    Mapping relationship: Auxiliary codes={len(aux_codes)}, Target codes={len(target_codes)}, Common codes={len(common_codes)}")
        
        for code in common_codes:
            aux_idx = aux_codes.index(code)
            target_idx = target_codes.index(code)
            true_mapping[target_idx] = aux_idx
        
        return true_mapping
    
    def run_single_experiment(self, aux_matrix: np.ndarray, aux_proc_codes: List[str], aux_diag_codes: List[str],
                            target_df: pd.DataFrame, sampling_strategy: str, sampling_ratio: float, 
                            run_id: int) -> Dict:
        """Run a single experiment with a given configuration"""
        print(f"Running experiment: Strategy={sampling_strategy}, Ratio={sampling_ratio}, Run={run_id}")
        
        # Sample the target dataset
        if sampling_strategy == 'frequency_weighted':
            sampled_target_df = self.frequency_weighted_sampling(target_df, sampling_ratio, random_seed=run_id)
        else:  # random
            sampled_target_df = self.random_sampling(target_df, sampling_ratio, random_seed=run_id)
        
        # Create co-occurrence matrix for the target dataset
        target_matrix, target_proc_codes, target_diag_codes = self.create_co_occurrence_matrix(sampled_target_df)
        
        # Create the true mapping
        true_mapping = self.create_true_mapping(aux_proc_codes, target_proc_codes)
        
        # Ensure matrix dimensions are compatible by padding with zeros if necessary
        print(f"    Matrix dimension handling:")
        print(f"      Before: Auxiliary matrix={aux_matrix.shape}, Target matrix={target_matrix.shape}")
        
        # Handle column mismatches by padding with zeros
        if aux_matrix.shape[1] > target_matrix.shape[1]:
            diff = aux_matrix.shape[1] - target_matrix.shape[1]
            target_matrix = np.pad(target_matrix, ((0, 0), (0, diff)), mode='constant', constant_values=0)
            print(f"      Column padding: Target matrix padded with {diff} columns")
        elif aux_matrix.shape[1] < target_matrix.shape[1]:
            diff = target_matrix.shape[1] - aux_matrix.shape[1]
            aux_matrix = np.pad(aux_matrix, ((0, 0), (0, diff)), mode='constant', constant_values=0)
            print(f"      Column padding: Auxiliary matrix padded with {diff} columns")
        
        # Skip row mismatch handling: different row numbers allowed
        if aux_matrix.shape[0] != target_matrix.shape[0]:
            print(f"      Row mismatch: Auxiliary matrix={aux_matrix.shape[0]} rows, Target matrix={target_matrix.shape[0]} rows (allowed)")
        
        print(f"      After: Auxiliary matrix={aux_matrix.shape}, Target matrix={target_matrix.shape}")
        
        # Run the Jigsaw attack
        print(f"    Starting Jigsaw attack...")
        predicted_mapping, execution_time = self.run_jigsaw_attack(aux_matrix, target_matrix)
        print(f"    Attack completed: Mappings={len(predicted_mapping)}, Execution time={execution_time:.2f}s")
        
        # Calculate accuracy metrics
        # Get frequency information for the target dataset
        target_frequencies = {}
        for target_idx in range(len(target_proc_codes)):
            # Calculate the frequency of each procedure code in the target dataset
            target_code = target_proc_codes[target_idx]
            target_freq = np.sum(target_matrix[target_idx, :])  # Sum of all columns for that row
            target_frequencies[target_idx] = int(target_freq)
        
        value_recovery_rate, row_recovery_rate = self.calculate_accuracy_metrics(
            true_mapping, predicted_mapping, target_frequencies
        )
        
        print(f"    Accuracy: Value recovery rate={value_recovery_rate:.4f}, Row recovery rate={row_recovery_rate:.4f}")
        
        return {
            'aux_year': None,  # Will be set when calling
            'target_year': None,  # Will be set when calling
            'sampling_strategy': sampling_strategy,
            'sampling_ratio': sampling_ratio,
            'run_id': run_id,
            'value_recovery_rate': value_recovery_rate,
            'row_recovery_rate': row_recovery_rate,
            'execution_time': execution_time,
            'num_mappings': len(predicted_mapping)
        }
    
    def run_all_experiments(self):
        """Run all experiments"""
        print("Starting Jigsaw attack experiment - CCSR Procedure Code")
        print("=" * 50)
        
        total_configs = len(self.years) * len(self.years) * len(self.sampling_strategies) * len(self.sampling_ratios)
        # Run each sampling strategy 3 times
        total_experiments = len(self.years) * (len(self.years) - 1) * len(self.sampling_strategies) * len(self.sampling_ratios) * self.num_runs
        
        print(f"Experiment configuration:")
        print(f"  Attack target: CCSR Procedure Code")
        print(f"  Year range: {self.years}")
        print(f"  Sampling strategies: {self.sampling_strategies}")
        print(f"  Sampling ratios: {[f'{r:.0%}' for r in self.sampling_ratios]}")
        print(f"  Runs: Each sampling strategy is run {self.num_runs} times")
        print(f"  Jigsaw parameters: {self.jigsaw_params}")
        print(f"Total configurations: {total_configs}")
        print(f"Total experiments: {total_experiments}")
        print("=" * 50)
        
        experiment_count = 0
        
        for aux_year in self.years:
            # Load auxiliary dataset and create matrix (only done once)
            print(f"\n=== Loading auxiliary dataset for {aux_year} ===")
            aux_df = self.load_data(aux_year)
            aux_matrix, aux_proc_codes, aux_diag_codes = self.create_co_occurrence_matrix(aux_df)
            print(f"Auxiliary dataset matrix created: {aux_matrix.shape}")
            
            for target_year in self.years:
                # Auxiliary and target datasets should not be the same year
                if aux_year == target_year:
                    continue
                
                # Load target dataset (only done once)
                print(f"\n=== Loading target dataset for {target_year} ===")
                target_df = self.load_data(target_year)
                
                for sampling_strategy in self.sampling_strategies:
                    for sampling_ratio in self.sampling_ratios:
                        print(f"\nConfig: aux={aux_year}, target={target_year}, strategy={sampling_strategy}, ratio={sampling_ratio}")
                        
                        # Run independent sampling for each configuration
                        run_results = []
                        
                        # Run both sampling strategies 3 times
                        num_runs = self.num_runs
                        
                        for run_id in range(num_runs):
                            try:
                                print(f"  Run {run_id+1}/{num_runs}: Independent sampling and attack")
                                
                                result = self.run_single_experiment(
                                    aux_matrix, aux_proc_codes, aux_diag_codes,
                                    target_df, sampling_strategy, sampling_ratio, run_id
                                )
                                # Set year information
                                result['aux_year'] = aux_year
                                result['target_year'] = target_year
                                run_results.append(result)
                                experiment_count += 1
                                
                                print(f"    Result: Value recovery rate={result['value_recovery_rate']:.4f}, "
                                      f"Row recovery rate={result['row_recovery_rate']:.4f}")
                                
                            except Exception as e:
                                print(f"    Experiment error: {e}")
                                continue
                        
                        # Calculate average results for this configuration
                        if run_results:
                            avg_value_recovery = np.mean([r['value_recovery_rate'] for r in run_results])
                            avg_row_recovery = np.mean([r['row_recovery_rate'] for r in run_results])
                            avg_execution_time = np.mean([r['execution_time'] for r in run_results])
                            std_value_recovery = np.std([r['value_recovery_rate'] for r in run_results])
                            std_row_recovery = np.std([r['row_recovery_rate'] for r in run_results])
                            
                            print(f"  Average results: Value recovery rate={avg_value_recovery:.4f}±{std_value_recovery:.4f}, "
                                  f"Row recovery rate={avg_row_recovery:.4f}±{std_row_recovery:.4f}, "
                                  f"Execution time={avg_execution_time:.2f}s (Ran {len(run_results)} times)")
                            
                            self.results.append({
                                'aux_year': aux_year,
                                'target_year': target_year,
                                'sampling_strategy': sampling_strategy,
                                'sampling_ratio': sampling_ratio,
                                'value_recovery_rate': avg_value_recovery,
                                'row_recovery_rate': avg_row_recovery,
                                'execution_time': avg_execution_time,
                                'num_runs': len(run_results)
                            })
        
        print("\nAll experiments completed!")
    
    def save_results(self, output_dir: str = "/path/to/your/output"):
        """Save results to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
