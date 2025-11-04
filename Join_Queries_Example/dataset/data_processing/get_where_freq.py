# gensi_where_freq.py
import os, duckdb
import pandas as pd


def gensi_freq(con,table,col,years,output_dir):
    for year in years:
        query = f'''
        SELECT "{col}" AS col
        FROM {table}
        WHERE year = {year} AND "{col}" IS NOT NULL
        '''
        df = con.execute(query).fetchdf()
        counts = df['col'].value_counts().sort_index()
        total = counts.sum()

        with open(f"{output_dir}/{col}_freq_{year}.txt", "w") as f:
            for value, count in counts.items():
                freq = count / total
                f.write(f"{value}\t{count}\t{freq}\n")
    
    


if __name__ == "__main__":
    # Dependent column single column leakage
    db_path = "dataset/chicago_data.db"
    con = duckdb.connect(db_path, read_only=True)
    table = "crimes"
    col = "District"#"District","Community Area"
    years = [2018, 2019, 2020, 2021, 2022]
    output_dir = "dataset/frequency/crimes(ca)-taxi-dropoff"
    os.makedirs(output_dir, exist_ok=True)
    gensi_freq(con,table,col,years,output_dir)
    con.close()
    
