import csv
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from init_setup.ticker_cusip_data import cusip_set, cusip_to_ticker
from utils.mappings import BASE_DIR_DATA_PARSE, STOCKS_SHS_Q_END_PRICES_FILE, BASE_DIR_FINAL


def find_duplicates_by_name():
    pattern = r"(.+)_([0-9]{4})_(Q[1-4])-\w+\.csv$"
    grouped_files = defaultdict(list)

    for dirpath, dirs, files in os.walk(BASE_DIR_FINAL):
        for file in files:
            if file.endswith('.csv'):
                match = re.match(pattern, file)
                if match:
                    key = match.group(1), match.group(2), match.group(3)
                    full_path = os.path.join(dirpath, file)
                    grouped_files[key].append(full_path)

    duplicates = {key: paths for key, paths in grouped_files.items() if len(paths) > 1}

    for (filer, year, quarter), paths in duplicates.items():
        print(f"Duplicate for {filer}, {year}, {quarter}:")
        for path in paths:
            print(f"    {path}")


def find_duplicate_csv_files(base_dir):
    # Dictionary to map filenames to list of their full paths
    files_dict = defaultdict(list)

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                files_dict[file].append(full_path)

    # Filter and print duplicates
    duplicates = {fname: paths for fname, paths in files_dict.items() if len(paths) > 1}

    if duplicates:
        print("Duplicate CSV files found:")
        for fname, paths in duplicates.items():
            print(f"Filename: {fname}")
            for path in paths:
                print(f"  - {path}")
            print()
    else:
        print("No duplicate CSV filenames found.")


def find_duplicate_folders(base_dir):
    # Dictionary to map folder name to list of full paths where it appears
    folder_map = defaultdict(list)

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            folder_map[d].append(os.path.join(root, d))

    # Find folder names that appear more than once
    duplicates = {folder: paths for folder, paths in folder_map.items() if len(paths) > 1}

    return duplicates


def remove_column_values_from_file(file):
    df = pd.read_csv(file)
    df["quarter_end_price"] = np.nan  # Or df["quarter_end_price"] = ''
    df.to_csv(STOCKS_SHS_Q_END_PRICES_FILE, index=False)


def delete_files_in_subfolders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.csv'):  # Check for CSV files (case-insensitive)
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


def extract_filername_year_quarter_accession(filename):
    """
    Extract filer name, year, quarter, accession number from filename.

    Expected format: filername_{year}_Q{quarter} - {accession_nr}.csv
    """
    pattern = r'^(?P<filername>.+)_(?P<year>\d{4})_Q(?P<quarter>[1-4])-(?P<accession_nr>\d+)\.csv$'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    filername = match.group('filername')
    year = int(match.group('year'))
    quarter = int(match.group('quarter'))
    accession_nr = match.group('accession_nr')
    return filername, year, quarter, accession_nr


def extract_year_quarter_from_filename(filename):
    # This regex looks for four digits (year) and Q[1-4] pattern in the filename
    match = re.search(r'(\d{4})_Q([1-4])', filename)
    if match:
        year = match.group(1)
        quarter = f"Q{match.group(2)}"
        return year, quarter
    else:
        raise ValueError(f"Year and quarter not found in filename: {filename}")


def find_file(filename, directory=BASE_DIR_DATA_PARSE):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {directory}")


def load_and_aggregate(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    df['title_of_class'] = df['title_of_class'].astype(str).str.strip()
    df = df[~df['title_of_class'].str.contains('option', case=False, na=False)]
    df['cusip'] = df['cusip'].astype(str)
    df = df[df['cusip'].isin(cusip_set)]
    df['ticker'] = df['cusip'].map(cusip_to_ticker)

    return df.groupby(['ticker']).agg({
        'share_amount': 'sum'
    }).reset_index()


def compare_13fs(file1: str, file2: str, min_diff: Optional[int] = 10):
    agg1 = load_and_aggregate(find_file(file1 + ".csv", directory=BASE_DIR_DATA_PARSE))
    agg2 = load_and_aggregate(find_file(file2 + ".csv", directory=BASE_DIR_DATA_PARSE))

    # Merge on keys with outer join to keep all holdings
    comparison = pd.merge(
        agg1, agg2,
        on=['ticker'],
        how='outer',
        suffixes=('_file1', '_file2')
    )

    # Fill NaNs with 0 for missing shares
    comparison['share_amount_file1'] = comparison['share_amount_file1'].fillna(0)
    comparison['share_amount_file2'] = comparison['share_amount_file2'].fillna(0)

    # Calculate difference in share amounts
    comparison['share_amount_diff'] = comparison['share_amount_file2'] - comparison['share_amount_file1']

    # Filter rows where the absolute difference is greater than min_diff
    differences = comparison[comparison['share_amount_diff'].abs() > min_diff]

    # Select columns to display
    cols_to_print = [
        'ticker', 'share_amount_file2', 'share_amount_file1', 'share_amount_diff'
    ]

    if not differences.empty:
        print(differences[cols_to_print])
    else:
        print("No differences found in share_amount between the two files.")


def count_rows_in_csv(file_path):
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader) - 1  # subtract header
    return row_count
