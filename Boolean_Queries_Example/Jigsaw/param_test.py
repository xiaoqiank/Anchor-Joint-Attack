#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Parameter Testing Script
Tests one parameter at a time to find the optimal value.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple

# Adding the path for Jigsaw attack code (masked directory)
sys.path.append('/path/to/your/jigsaw')
from jigsaw import JigsawAttacker

class SingleParamTester:
    def __init__(self, data_dir: str = "/path/to/your/data"):
        self.data_dir = data_dir
        self.years = [2018, 2019]
        
        # Default parameters
        self.default_params = {
            'baseRec': 50,
            'confRec': 25,
            'refinespeed': 15,
            'alpha': 0.5,
            'beta': 0.4
        }
        
        self.results = []
        
    def load_data(self, year: int) -> pd.DataFrame:
        """Load the dataset for the specified year"""
        file_path = os.path.join(self.data_dir, f"sparcs_{year}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file does not exist: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def create_co_occurrence_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create a co-occurrence matrix from the data"""
        # Calculate frequencies and sort in descending order
        freq_map = df.groupby('CCSR Diagnosis Code').size().to_dict()
        diagnosis_codes = sorted(freq_map.keys(), key=lambda x: freq_map[x], reverse=True)
        
        freq_map1 = df.groupby('CCSR Procedure Code').size().to_dict()
        procedure_codes = sorted(freq_map1.keys(), key=lambda x: freq_map1[x], reverse=True)
        
        # Create mappings
        row_map = {v: i for i, v in enumerate(diagnosis_codes)}
        col_map = {v: j for j, v in enumerate(procedure_codes)}
        
        # Create the matrix
        matrix = np.zeros((len(diagnosis_codes), len(procedure_codes)))
        
        # Fill the matrix
        for _, row in df.iterrows():
            val1 = str(row['CCSR Diagnosis Code'])
            val2 = str(row['CCSR Procedure Code'])
            if val1 in row_map and val2 in col_map:
                matrix[row_map[val1], col_map[val2]] += 1
        
        return matrix, diagnosis_codes, procedure_codes
    
    def create_true_mapping(self, aux_codes: List[str], target_codes: List[str]) -> Dict[int, int]:
        """Create the true mapping based on auxiliary and target codes"""
        true_mapping = {}
        
        aux_set = set(aux_codes)
        target_set = set(target_codes)
        common_codes = aux_set.intersection(target_set)
        
        for code in common_codes:
            aux_idx = aux_codes.index(code)
            target_idx = target_codes.index(code)
            true_mapping[target_idx] = aux_idx
        
        return true_mapping
    
    def calculate_accuracy(self, true_mapping: Dict[int, int], predicted_mapping: Dict[int, int]) -> float:
        """Calculate accuracy"""
        if not predicted_mapping:
            return 0.0
        
        correct = 0
        total = len(predicted_mapping)
        
        for target_idx, predicted_idx in predicted_mapping.items():
            if target_idx in true_mapping and true_mapping[target_idx] == predicted_idx:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def run_jigsaw_attack(self, sim_M: np.ndarray, real_M: np.ndarray, params: Dict) -> Tuple[Dict[int, int], float]:
        """Run the Jigsaw attack"""
        start_time = time.time()
        
        attacker = JigsawAttacker(
            sim_M=sim_M, 
            real_M=real_M,
            baseRec=params['baseRec'],
            confRec=params['confRec'],
            refinespeed=params['refinespeed'],
            alpha=params['alpha'],
            beta=params['beta']
        )
        
        attacker.attack_step_1()
        attacker.attack_step_2()
        mapping_dict = attacker.attack_step_3()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return mapping_dict, execution_time
    
    def test_single_parameter(self, param_name: str, param_values: List, test_years: List[int] = [2018, 2019, 2020]):
        """Test different values for a single parameter"""
        print(f"\nTesting parameter: {param_name}")
        print(f"Testing values: {param_values}")
        print("=" * 50)
        
        results = []
        
        for param_value in param_values:
            # Create parameter configuration
            params = self.default_params.copy()
            params[param_name] = param_value
            
            print(f"\nTesting {param_name}={param_value}")
            
            accuracies = []
            
            # Test all year combinations
            for aux_year in test_years:
                for target_year in test_years:
                    if aux_year == target_year:
                        continue
                    
                    try:
                        # Load data
                        aux_df = self.load_data(aux_year)
                        target_df = self.load_data(target_year)
                        
                        # Create co-occurrence matrices
                        aux_matrix, aux_diag_codes, _ = self.create_co_occurrence_matrix(aux_df)
                        target_matrix, target_diag_codes, _ = self.create_co_occurrence_matrix(target_df)
                        
                        # Create true mapping
                        true_mapping = self.create_true_mapping(aux_diag_codes, target_diag_codes)
                        
                        # Ensure matrix dimensions are compatible by padding with zeros if necessary
                        # Handle column mismatches
                        if aux_matrix.shape[1] > target_matrix.shape[1]:
                            diff = aux_matrix.shape[1] - target_matrix.shape[1]
                            target_matrix = np.pad(target_matrix, ((0, 0), (0, diff)), mode='constant', constant_values=0)
                        elif aux_matrix.shape[1] < target_matrix.shape[1]:
                            diff = target_matrix.shape[1] - aux_matrix.shape[1]
                            aux_matrix = np.pad(aux_matrix, ((0, 0), (0, diff)), mode='constant', constant_values=0)
                        
                        # Handle row mismatches
                        if aux_matrix.shape[0] > target_matrix.shape[0]:
                            diff = aux_matrix.shape[0] - target_matrix.shape[0]
                            target_matrix = np.pad(target_matrix, ((0, diff), (0, 0)), mode='constant', constant_values=0)
                        elif aux_matrix.shape[0] < target_matrix.shape[0]:
                            diff = target_matrix.shape[0] - aux_matrix.shape[0]
                            aux_matrix = np.pad(aux_matrix, ((0, diff), (0, 0)), mode='constant', constant_values=0)
                        
                        # Run Jigsaw attack
                        predicted_mapping, execution_time = self.run_jigsaw_attack(aux_matrix, target_matrix, params)
                        
                        # Calculate accuracy
                        accuracy = self.calculate_accuracy(true_mapping, predicted_mapping)
                        accuracies.append(accuracy)
                        
                        print(f"  {aux_year}->{target_year}: {accuracy:.4f}")
                        
                    except Exception as e:
                        print(f"  {aux_year}->{target_year}: Error - {e}")
                        continue
            
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                results.append({
                    'param_name': param_name,
                    'param_value': param_value,
                    'avg_accuracy': avg_accuracy,
                    'num_tests': len(accuracies)
                })
                print(f"  Average accuracy: {avg_accuracy:.4f}")
            else:
                print(f"  No successful results")
        
        return results
    
    def find_best_value(self, results: List[Dict]) -> Dict:
        """Find the optimal parameter value"""
        if not results:
            return {}
        
        best_result = max(results, key=lambda x: x['avg_accuracy'])
        return best_result
    
    def save_results(self, results: List[Dict], param_name: str, output_dir: str = "/path/to/your/output"):
        """Save the results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        filename = f"jigsaw_{param_name}_test_results.csv"
        results_df.to_csv(os.path.join(output_dir, filename), index=False)
        
        # Save the optimal parameter
        best_result = self.find_best_value(results)
        if best_result:
            with open(os.path.join(output_dir, f"best_{param_name}.txt"), 'w') as f:
                f.write(f"Optimal {param_name} parameter:\n")
                f.write(f"{param_name}: {best_result['param_value']}\n")
                f.write(f"Average accuracy: {best_result['avg_accuracy']:.4f}\n")
                f.write(f"Number of tests: {best_result['num_tests']}\n")
        
        print(f"Results saved to {output_dir}")

def main():
    """Main function"""
    tester = SingleParamTester()
    
    # Define parameters and values to test
    param_tests = {
        'baseRec': [20, 30, 40, 50, 60, 70, 80, 100],
        'confRec': [10, 15, 20, 25, 30, 35, 40, 50],
        'refinespeed': [10, 15, 20, 25, 30],
        'alpha': [0.3, 0.4, 0.5, 0.6, 0.7],
        'beta': [0.2, 0.3, 0.4, 0.5, 0.6]
    }
    
    # Store all results
    all_results = {}
    
    try:
        for param_name, param_values in param_tests.items():
            print(f"\n{'='*60}")
            print(f"Starting tests for parameter: {param_name}")
            print(f"{'='*60}")
            
            results = tester.test_single_parameter(param_name, param_values)
            all_results[param_name] = results
            
            # Find the best value
            best_result = tester.find_best_value(results)
            if best_result:
                print(f"\nOptimal {param_name}: {best_result['param_value']}")
                print(f"Average accuracy: {best_result['avg_accuracy']:.4f}")
                
                # Update default parameters
                tester.default_params[param_name] = best_result['param_value']
            
            # Save results
            tester.save_results(results, param_name)
        
        # Save final best parameter combination
        with open("/path/to/your/output/final_best_parameters.txt", 'w') as f:
            f.write("Final optimal Jigsaw attack parameters:\n")
            for param_name, value in tester.default_params.items():
                f.write(f"{param_name}: {value}\n")
        
        print(f"\n{'='*60}")
        print("All parameter tests completed!")
        print("Final optimal parameters:")
        for param_name, value in tester.default_params.items():
            print(f"  {param_name}: {value}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Test process error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
