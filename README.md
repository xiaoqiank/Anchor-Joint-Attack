# Anchor-Joint-Attack

This repository implements attacks for plaintext recovery in database query settings. The code base contains data preparation utilities, multiple attack implementations, and helper utilities.

## 1. Setup
### 1.1 Experiment Environment
- OS: Linux (development tested on modern Ubuntu/Debian derivatives)
- Python: 3.8+ (recommend 3.8, 3.9 or 3.10)
- Create a virtual environment and install common scientific/python packages. There is no central `requirements.txt` in the repository; install the typical packages used by the project if not already present:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy pandas scikit-learn networkx
```

### 1.2 Dataset


SPRACS Data (2018-2024): (https://health.data.ny.gov/stories/s/wvua-rr23)

Chicago Data (2018‑2022):
- crimes: (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)

- taxi: (https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew)

- rideshare: (https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2018-2022-/m6dm-c72p)

- crashes: (https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if)

- For the convenience of our experiments, we have ingested the above Chicago datasets into a DuckDB database. The anonymized database file is hosted on OSF and can be accessed here:
https://osf.io/6sehb/?view_only=2f89bc3472594e3ba0ceff0fc76198b6


## 2 Code Architecture
```text
├── Boolean_Queries_Example
|    ├── data_preparation/        # frequency statistics 
|    │       └── frequency.py         
|    ├── Jigsaw/                  # Jigsaw attack
|    │   ├── jigsaw.py            
|    │   └─  run_attack.py        
|    ├── Ours/                    # our proposed attack       
|    |   ├── GetAnchors.py        
|    │   ├── Remain.py            
|    │   ├── XColRecover.py      
|    │   ├── run_attack.py        
|    │   └─  utils/
|    └── Single/                  # Single-column attack
|        └── run_single.py        
└── Join_Queries_Example
     ├── dataset/                 #data processing
     |   ├── chicago_data.db          # Database file
     |   └── data_processing/         # Scripts for data  
     ├── our_attack/         #our attack                     
     └── LAA4STE4SQL/        # The attack proposed by Hoover et al. (USENIX’24) 


```


## 3. Scenario 1:  Plaintext Recovery Attack in Boolean Queries
### 3.1 Running Guide

1) Prepare frequency infromation
```bash
python3 data_preparation/frequency.py
```
2) Run our attack (Ours)
```bash
python3 Ours/run_attack.py
```
3) Run the Jigsaw attack
```bash
python3 Jigsaw/run_attack.py
```

4) Run the Single baseline
```bash
python3 Single/run_single.py
```


## 4. Scenario 2: Plaintext Recovery in Join Queries
  
#### 4.1 Data Preparation

  Use the scripts in `dataset/data_processing/` to generate frequency and distribution files for the attacks:
   * `get_where_freq.py`: Gets the attribute value-count-frequency for the `where` column in table `t1`.
   * `get_join_freq.py`: Gets the cross-column equality information, common attribute values, and frequencies for the `join` columns of tables `t1` and `t2`.
   * `get_joint_freq.py`: Gets the joint distribution information for table `t1`.
   * `sample_weighted.py` / `sample_random.py`: Performs weighted or random sampling on the generated data.

  The subsequent attacks use the processed frequency files as input.

---
#### 4.2 Running the Attacks

* **Run Single Attack :**
  ```bash
  cd our_attack
  python3 run_attack.py
  ```

* **Run Our Attack:**
  ```bash
  cd our_attack
  python3 run_all_param.py
  ```
* **Run attack proposed hoover et al.(https://github.com/ste4sql/LAA4STE4SQL):**
   ```bash
   cd LAA4STE4SQL
   python3 run_all_param.py
   ```





