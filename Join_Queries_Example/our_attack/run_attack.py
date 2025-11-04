import os
import sys
from GetAnchors import GetAnchors 
from XColRecover import XColRecover
from utils.tool import resolve_conflicts, keep_max_score, read_freq_file
from utils.bipartite_matching import bipartite_matching

def step1(args,obs_year,freq_dir_aux,freq_dir):  
    aux_year = args["aux_year"]
    main_table1_name = args["table1_name"] 
    table1_where_name = args["table1_where_name"]
    table1_join_name = args["table1_join_name"] 
    table1_where_tau = args["table1_where_tau"]
    table1_join_tau = args["table1_join_tau"]
    table1_where_mu = args["table1_where_mu"]
    table1_join_mu = args["table1_join_mu"]
    output_csv = args["output_csv"]
    co_tau=args["co_tau"]
    T_max=args["T_max"]
    reg=args["reg"]


    #---di---where---
    aux_where_file = os.path.join(freq_dir_aux, f"{table1_where_name}_freq_{aux_year}.txt")
    aux_where_join_file = os.path.join(freq_dir_aux, f"{table1_where_name}_{table1_join_name}_co_{aux_year}.csv")
    aux_join_file = os.path.join(freq_dir_aux, f"{table1_join_name}_freq_{aux_year}.txt")
    aux_join_where_file = os.path.join(freq_dir_aux, f"{table1_join_name}_{table1_where_name}_co_{aux_year}.csv")

    #---ca---join---
    obs_where_file = os.path.join(freq_dir, f"{table1_where_name}_freq_{obs_year}.txt")
    obs_where_join_file = os.path.join(freq_dir, f"{table1_where_name}_{table1_join_name}_co_{obs_year}.csv")
    obs_join_file = os.path.join(freq_dir, f"{table1_join_name}_freq_{obs_year}.txt")
    obs_join_where_file = os.path.join(freq_dir, f"{table1_join_name}_{table1_where_name}_co_{obs_year}.csv")

    
    results_where,_ = GetAnchors(
        obs_year,
        aux_where_file,
        obs_where_file,
        obs_where_join_file,
        aux_where_join_file,
        table1_where_tau,
        table1_where_mu,
        output_csv,
        type=0,
        e=True,
        reg=reg
    )

    results_join,_ = GetAnchors(
        obs_year,
        aux_join_file,
        obs_join_file,
        obs_join_where_file,
        aux_join_where_file,
        table1_join_tau,
        table1_join_mu,
        output_csv,
        type=0,
        e=True,
        reg=reg
    )

    S1=results_where #where
    S2=results_join  #join
    M1=results_where #where
    M2=results_join  #join

    for i in range(1, T_max):
        R1 = XColRecover(
            obs_year, 
            S2, 
            aux_join_where_file, 
            obs_join_where_file, 
            co_tau
        )

        R2 = XColRecover(
            obs_year, 
            S1, 
            aux_where_join_file, 
            obs_where_join_file, 
            co_tau
        )
        
        # Update M1 and M2
        M1 = M1 + R1 #where
        M2 = M2 + R2 #join
        M1 = resolve_conflicts(M1, co_tau)
        M2 = resolve_conflicts(M2, co_tau) 


        # Remove duplicates: remove items from R1 that are already in S1, and update to the new S1
        s1_pairs = set((a, b) for a, b, _ in S1)
        R1 = [triplet for triplet in R1 if (triplet[0], triplet[1]) not in s1_pairs]
        S1 = R1

        # Remove duplicates: remove items from R2 that are already in S2, and update to the new S2
        s2_pairs = set((a, b) for a, b, _ in S2)
        R2 = [triplet for triplet in R2 if (triplet[0], triplet[1]) not in s2_pairs]
        S2 = R2
    


    # --- Step 1: Update M1 and prepare for bipartite matching ---
    M1 = keep_max_score(M1)
    used_obs1 = set(a for a, _, _ in M1)
    used_aux1 = set(b for _, b, _ in M1)
    obs_labels1, obs_fre1, obs_fre2 = read_freq_file(obs_where_file)
    aux_labels1, aux_fre1, aux_fre2 = read_freq_file(aux_where_file)
    obs_rest1 = [a for i, a in enumerate(obs_labels1) if a not in used_obs1]
    aux_rest1 = [a for i, a in enumerate(aux_labels1) if a not in used_aux1]
    obs_f1 = [obs_fre1[i] for i, a in enumerate(obs_labels1) if a not in used_obs1]
    obs_f2 = [obs_fre2[i] for i, a in enumerate(obs_labels1) if a not in used_obs1]
    aux_f1 = [aux_fre1[i] for i, a in enumerate(aux_labels1) if a not in used_aux1]
    aux_f2 = [aux_fre2[i] for i, a in enumerate(aux_labels1) if a not in used_aux1]
    M1_p = bipartite_matching(aux_rest1, aux_f1, aux_f2, obs_rest1, obs_f1, obs_f2, 0)

    # --- Step 2: Update M2 and prepare for bipartite matching ---
    M2 = keep_max_score(M2)
    used_obs2 = set(a for a, _, _ in M2)
    used_aux2 = set(b for _, b, _ in M2)
    obs_labels2, obs_fre1_2, obs_fre2_2 = read_freq_file(obs_join_file)
    aux_labels2, aux_fre1_2, aux_fre2_2 = read_freq_file(aux_join_file)
    obs_rest2 = [a for i, a in enumerate(obs_labels2) if a not in used_obs2]
    aux_rest2 = [a for i, a in enumerate(aux_labels2) if a not in used_aux2]
    obs_f1_2 = [obs_fre1_2[i] for i, a in enumerate(obs_labels2) if a not in used_obs2]
    obs_f2_2 = [obs_fre2_2[i] for i, a in enumerate(obs_labels2) if a not in used_obs2]
    aux_f1_2 = [aux_fre1_2[i] for i, a in enumerate(aux_labels2) if a not in used_aux2]
    aux_f2_2 = [aux_fre2_2[i] for i, a in enumerate(aux_labels2) if a not in used_aux2]
    M2_p = bipartite_matching(aux_rest2, aux_f1_2, aux_f2_2, obs_rest2, obs_f1_2, obs_f2_2, 0)

    # --- Step 3: Merge results ---
    result1 = [(a, b) for a, b, _ in M1] + M1_p
    result2 = [(a, b) for a, b, _ in M2] + M2_p

    # --- Step 4: Print attack success rate ---
    acc1 = sum(1 for a, b in result1 if a == b) / len(result1) if result1 else 0
    acc2 = sum(1 for a, b in result2 if a == b) / len(result2) if result2 else 0
    print(f"[{obs_year}] where attack success rate: {acc1:.2%} ({sum(1 for a, b in result1 if a == b)}/{len(result1)})")
    print(f"[{obs_year}] join attack success rate: {acc2:.2%} ({sum(1 for a, b in result2 if a == b)}/{len(result2)})")

    return result1, result2, acc1, acc2




if __name__ == "__main__":
    freq_dir_aux = "dataset/frequency/crimes(ca)-crashes-beat"
    freq_dir = "dataset/frequency/crimes(ca)-crashes-beat-20%"  
    obs_years = [2019,2020,2021,2022]
    aux_year = 2018
    attack_name = "ours"
    table1 = "crimes"
    where_name="Community Area"
    join_name="Beat"
    where_tau=0.015
    where_mu=0.05
    join_tau=0.0005
    join_mu=0.05
    co_tau=0.05
    T_max=5
    reg=0.001

    args = sys.argv[1:]
    params = {}
    Result_where={} #Collect results from each iteration
    Result_where={}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=')
            params[key] = value
    if attack_name == "ours":
        attack_params = {
            "aux_year": aux_year,
            "table1_name": table1,
            "table1_where_name": where_name,
            "table1_join_name": join_name,
            "table1_where_tau": where_tau,
            "table1_where_mu": where_mu,
            "table1_join_tau": join_tau,
            "table1_join_mu": join_mu,
            "co_tau":co_tau,
            "T_max":T_max,
            "reg":reg

        }
        for year in obs_years:
            output_csv = f"anchors_{table1}_{year}.csv"
            attack_params["output_csv"] = output_csv
            result_where, result_join, _, _ = step1(attack_params, year,freq_dir_aux,freq_dir)
            # Write two result files
            import csv

            # Save result_where to CSV
            output_where_csv = f"result_where1_{year}.csv"
            with open(output_where_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["obs_label", "aux_label", "score"])
                for triplet in result_where:
                    writer.writerow(triplet)

            # Save result_join to CSV
            output_join_csv = f"result_join1_{year}.csv"
            with open(output_join_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["obs_label", "aux_label"])
                for triplet in result_join:
                    writer.writerow(triplet)