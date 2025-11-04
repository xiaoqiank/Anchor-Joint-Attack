import os, duckdb
import pandas as pd
import numpy as np

def genco_st_freq(db_path,table_name,where_name,join_name,years,output_dir):
    con = duckdb.connect(db_path, read_only=True)
    for year in years:
        query = f'''
        SELECT "{where_name}" as col1, "{join_name}" as col2, COUNT(*) as cnt
        FROM {table_name}
        WHERE year = {year} AND "{where_name}" IS NOT NULL AND "{join_name}" IS NOT NULL
        GROUP BY col1, col2
        ORDER BY col1, col2
        '''
        df = con.execute(query).fetchdf()
        df = df[df['cnt'] > 2]
        df['freq'] = df.groupby('col1')['cnt'].transform(lambda x: x / x.sum()).round(6)
        df = df[['col1', 'col2', 'cnt', 'freq']]
        out_path = f"{output_dir}/{where_name}_{join_name}_co_{year}.csv"
        df.to_csv(out_path, index=False)
    con.close()


if __name__ == "__main__":
    db_path = "dataset/chicago_data.db"
    table_name = "crimes"
    where_name =  "District"#"District","Beat","Ward"
    join_name = "Community Area" #"Community Area"
    years = [2018, 2019, 2020, 2021, 2022]
    output_dir = "dataset/frequency/crimes(ca)-taxi-pickup"
    os.makedirs(output_dir, exist_ok=True)
    genco_st_freq(db_path,table_name,where_name,join_name,years,output_dir)