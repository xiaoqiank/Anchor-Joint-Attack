import pandas as pd
import os

DATA_DIR = "DATA_DIRECTORY"
OUTPUT_ROOT = "OUTPUT_DIRECTORY"

AUXILIARY_DATA = os.path.join(DATA_DIR, "auxiliary_data.csv")
TARGET_DATA = os.path.join(DATA_DIR, "target_data.csv")

def process_dataset(input_file, output_prefix):
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"{'='*60}")

    df = pd.read_csv(input_file)
    total_rows = len(df)

    print(f"Total rows: {total_rows:,}")
    print(f"Columns: {list(df.columns)}")

    df.columns = ['A', 'B']

    df['A'] = df['A'].astype(str)
    df['B'] = df['B'].astype(str)

    output_dir = os.path.join(OUTPUT_ROOT, output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    def write_freq_txt(col_name, filename_suffix):
        counts = df[col_name].value_counts().reset_index()
        counts.columns = ['Value', 'Count']
        counts['Frequency'] = counts['Count'] / total_rows

        file_path = os.path.join(output_dir, f"{col_name}_freq_{filename_suffix}.txt")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{'Value':<15}    Count        Frequency\n")
            for _, row in counts.iterrows():
                val = row['Value']
                cnt = row['Count']
                freq = row['Frequency']
                f.write(f"{val:<15}    {cnt:<10}    {freq:.10f}\n")

        print(f"  Saved: {file_path}")

    print(f"\nWriting single-column frequency files...")
    write_freq_txt('A', output_prefix)
    write_freq_txt('B', output_prefix)

    print(f"\nWriting joint distribution files...")

    joint_counts = df.groupby(['A', 'B']).size().reset_index(name='JointCount')

    a_totals = df.groupby('A').size().reset_index(name='TotalA')
    a_b_df = pd.merge(joint_counts, a_totals, on='A')
    a_b_df['CondFreq'] = a_b_df['JointCount'] / a_b_df['TotalA']

    out_a_b = a_b_df[['A', 'B', 'JointCount', 'CondFreq']].copy()
    out_a_b.to_csv(os.path.join(output_dir, f"A_B_{output_prefix}.csv"), index=False)
    print(f"  Saved: A_B_{output_prefix}.csv")

    b_totals = df.groupby('B').size().reset_index(name='TotalB')
    b_a_df = pd.merge(joint_counts, b_totals, on='B')
    b_a_df['CondFreq'] = b_a_df['JointCount'] / b_a_df['TotalB']

    out_b_a = b_a_df[['B', 'A', 'JointCount', 'CondFreq']].copy()
    out_b_a.to_csv(os.path.join(output_dir, f"B_A_{output_prefix}.csv"), index=False)
    print(f"  Saved: B_A_{output_prefix}.csv")

    joint_df = joint_counts.copy()
    joint_df['JointFreq'] = joint_df['JointCount'] / total_rows
    joint_df.to_csv(os.path.join(output_dir, f"Joint_{output_prefix}.csv"), index=False)
    print(f"  Saved: Joint_{output_prefix}.csv")

    print(f"\nCompleted processing: {output_prefix}")

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        print(f"Created directory: {OUTPUT_ROOT}")

    process_dataset(AUXILIARY_DATA, "auxiliary")
    process_dataset(TARGET_DATA, "target")

    print(f"\n{'='*60}")
    print("All tasks completed successfully!")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()