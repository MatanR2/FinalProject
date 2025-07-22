import pandas as pd
import os

def merge_csv_files(file_paths, output_path):
    # Initialize an empty list to store dataframes
    dfs = []

    # Read each CSV file and append to the list
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"Warning: File {file_path} not found")

    # Concatenate all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)

        # Write to output file
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully merged {len(dfs)} files into {output_path}")
    else:
        print("No files were successfully read. No output generated.")

# Files to merge in order (October, November, December)
files_to_merge = [
    "/content/drive/My Drive/Colab Notebooks/combined_sorted_october_data.csv",
    "/content/drive/My Drive/Colab Notebooks/combined_sorted_november_data.csv",
    "/content/drive/My Drive/Colab Notebooks/combined_sorted_december_data.csv",
    "/content/drive/My Drive/Colab Notebooks/combined_sorted_january_data.csv",
    "/content/drive/My Drive/Colab Notebooks/combined_sorted_february_data.csv",
    "/content/drive/My Drive/Colab Notebooks/combined_sorted_march_data.csv"
]

# Output file path
output_file = "/content/drive/My Drive/Colab Notebooks/combined_all_months_data.csv"

# Call the function
merge_csv_files(files_to_merge, output_file)
