import os
import sys
import csv
import copy
from run_attack import step1

def run_sensitivity_analysis():
    """
    Runs a sensitivity analysis by varying one parameter at a time.
    """
    freq_dir_aux = "dataset/frequency/crimes(ca)-crashes-beat"
    freq_dir = "dataset/frequency/crimes(ca)-crashes-beat"
    obs_years = [2019, 2020, 2021, 2022]
    aux_year = 2018
    table1 = "crimes"
    where_name = "Community Area"
    join_name = "Beat"

    # Baseline parameters from run_attack.py
    baseline_params = {
        "table1_where_tau": 0.015,
        "table1_where_mu": 0.05,
        "table1_join_tau": 0.0005,
        "table1_join_mu": 0.05,
        "co_tau": 0.05,
        "T_max": 5,
        "reg": 0.001
    }

    # Parameter ranges to test
    params_to_test = {
        # "table1_where_tau": [0.0001,0.0005,0.001,0.005,0.01, 0.05,0.1,0.5],
        # "table1_where_mu": [0.0001,0.0005,0.001,0.005,0.01, 0.05,0.1,0.5],
        # "table1_join_tau": [0.0001,0.0005,0.001,0.005,0.01, 0.05,0.1,0.5],
        # "table1_join_mu": [0.0001,0.0005,0.001,0.005,0.01, 0.05,0.1,0.5],
        # "co_tau": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5],
        
        # "T_max": [3, 5, 7, 9,11],
        "reg": [1e-7,1e-6,0.00001,0.0001,0.001, 0.01, 0.1, 1, 10,100,1000]
    }

    results_file = "sensitivity_results_avg.csv"
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "varied_param", "param_value", "where_tau", "where_mu", "join_tau",
            "join_mu", "co_tau", "T_max", "reg", "avg_where_acc", "avg_join_acc"
        ]
        writer.writerow(header)

        for param_name, param_values in params_to_test.items():
            print(f"--- Varying parameter: {param_name} ---")
            for value in param_values:
                current_params = copy.deepcopy(baseline_params)
                current_params[param_name] = value

                yearly_where_acc = []
                yearly_join_acc = []

                for year in obs_years:
                    attack_params = {
                        "aux_year": aux_year,
                        "table1_name": table1,
                        "table1_where_name": where_name,
                        "table1_join_name": join_name,
                        **current_params,
                        "output_csv": f"anchors_{table1}_{year}.csv"
                    }
                    print(attack_params)

                    _, _, acc1, acc2 = step1(attack_params, year, freq_dir_aux, freq_dir)
                    yearly_where_acc.append(acc1)
                    yearly_join_acc.append(acc2)
                    print(f"Param: {param_name}={value}, Year: {year}, Where Acc: {acc1:.2%}, Join Acc: {acc2:.2%}")

                avg_where_acc = sum(yearly_where_acc) / len(yearly_where_acc)
                avg_join_acc = sum(yearly_join_acc) / len(yearly_join_acc)

                print(f"--- Average for {param_name}={value}: Where Acc: {avg_where_acc:.2%}, Join Acc: {avg_join_acc:.2%} ---")

                row_data = [
                    param_name, value,
                    current_params["table1_where_tau"], current_params["table1_where_mu"],
                    current_params["table1_join_tau"], current_params["table1_join_mu"],
                    current_params["co_tau"], current_params["T_max"], current_params["reg"],
                    f"{avg_where_acc:.4f}", f"{avg_join_acc:.4f}"
                ]
                writer.writerow(row_data)

if __name__ == "__main__":
    run_sensitivity_analysis()
