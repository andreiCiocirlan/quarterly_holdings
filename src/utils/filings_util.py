import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from init_setup.ticker_cusip_data import cusip_to_ticker, cusip_set
from utils.date_util import get_year_and_quarter
from utils.file_util import extract_filername_year_quarter_accession, extract_year_quarter_from_filename
from utils.mappings import BASE_DIR_FINAL, STOCKS_SHS_Q_END_PRICES_FILE, CIK_TO_PARSED_13F_DIR, \
    CIK_TO_FINAL_DIR, QUARTER_END_PRICE_DICT, BASE_DIR_DATA_PARSE, CIK_TO_FILER, HEADERS


def check_latest_13f(ciks, page_count=5):
    normalized_ciks = set(str(int(cik)) for cik in ciks)
    base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    found_ciks = []

    for page in range(1, page_count + 1):
        params = {
            'company': '',
            'CIK': '',
            'type': '13F-HR',
            'owner': 'include',
            'count': '100',
            'action': 'getcurrent',
            'start': (page - 1) * 100
        }
        response = requests.get(base_url, params=params, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the div containing filings
        main_div = soup.find('div', style=lambda value: value and 'margin-left: 10px' in value)
        if not main_div:
            print("Main div with filings not found on page", page)
            continue

        # Find the filings table by header detection as before
        tables = main_div.find_all('table')
        filings_table = None
        for table in tables:
            header_row = table.find('tr')
            if not header_row:
                continue
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all('th')]
            if 'form' in headers and 'description' in headers and 'filing date' in headers:
                filings_table = table
                break

        if not filings_table:
            print("Filings table not found on page", page)
            continue

        rows = filings_table.find_all('tr')[1:]  # skip header

        # Iterate rows with index to access previous row
        for i in range(1, len(rows)):
            company_row = rows[i - 1]
            filing_row = rows[i]

            # Extract company + CIK from company_row
            company_cols = company_row.find_all('td')
            if len(company_cols) < 3:
                continue
            company_td = company_cols[2]
            a_tag = company_td.find('a')
            if not a_tag:
                continue
            cik_text = a_tag.get_text(strip=True)

            # Extract filing info from filing_row
            filing_cols = filing_row.find_all('td')
            if len(filing_cols) < 5:
                continue

            form_type = filing_cols[0].get_text(strip=True)
            filing_date = filing_cols[4].get_text(strip=True)

            # Extract CIK number from cik_text
            match = re.search(r'\((\d{10})\)', cik_text)
            if not match:
                continue
            cik_raw = match.group(1)
            cik_norm = str(int(cik_raw))  # '2070929' (no leading zeros)

            if cik_norm in normalized_ciks and form_type in ['13F-HR', '13F-HR/A']:
                found_ciks.append(cik_norm)

    print(f"Found recent filings for ciks: {found_ciks}")

    return found_ciks


def check_csv_structure(file_path):
    try:
        df = pd.read_csv(file_path)

        required_cols = ['rank', 'ticker', 'share_amount', 'share_value', 'percentage', 'change', 'change_pct',
                         'ownership_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols} for {file_path}")
            return False

        if df['rank'].duplicated().any():
            dup_ranks = df['rank'][df['rank'].duplicated()].unique()
            print(f"Duplicate ranks found: {dup_ranks} for {file_path}")
            return False

        ranks = df['rank'].values
        expected_ranks = np.arange(1, len(df) + 1)
        if not np.array_equal(ranks, expected_ranks):
            print(f"Ranks are not continuous from 1 to {len(df)} for {file_path}")
            return False

        last_rank = df['rank'].max()
        if last_rank != len(df):
            print(f"Last rank {last_rank} does not match number of rows {len(df)}.")
            return False

        # print("CSV structure check passed.")
        return True

    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return False


def checkup_final_directories_csv_structure():
    for dirpath, dirnames, filenames in os.walk(BASE_DIR_FINAL):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                check_csv_structure(file_path=file_path)


def find_13fs_without_matching_raw_file():
    accession_final = set()
    for root, _, files in os.walk(BASE_DIR_FINAL):
        for file in files:
            if file.endswith(".csv"):
                # Split filename by '-' and get second part (index 1)
                parts = file.split('-')
                if len(parts) > 1:
                    accession_nr = parts[1].split('.')[0]  # Remove .csv extension if attached
                    accession_final.add(accession_nr)

    # Collect accession_nrs from BASE_DIR_DATA_PARSE
    accession_data_parse = set()
    for root, _, files in os.walk(BASE_DIR_DATA_PARSE):
        for file in files:
            if file.endswith(".csv"):
                # The file names here are accession_nrs directly (no split needed)
                accession_nr = file.split('.')[0]
                accession_data_parse.add(accession_nr)

    # Get accession numbers in FINAL but not in DATA_PARSE
    difference = accession_final - accession_data_parse
    return difference


def delete_final_13f_by_accession(cik, restatement_accession, report_date):
    deleted_files = []
    year, quarter = get_year_and_quarter(report_date)
    filer_name = CIK_TO_FILER.get(cik)
    final_dir = CIK_TO_FINAL_DIR.get(cik)
    if not filer_name or not final_dir:
        raise ValueError(f"Could not map CIK {cik} to filer name/final dir")

    final_path = str(os.path.join(BASE_DIR_FINAL, year, quarter, CIK_TO_FINAL_DIR.get(cik)))
    restatement_accession = restatement_accession.lstrip('0')  # Ensure no leading zeros mismatch

    pattern = re.compile(
        rf"^{re.escape(filer_name)}_{year}_{quarter}-([0-9]+)\.csv$"
    )

    for root, _, files in os.walk(final_path):
        for file in files:
            if not file.endswith(".csv"):
                continue
            match = pattern.match(file)
            if match:
                accession_nr = match.group(1).lstrip('0')
                # Skip the restatement accession itself
                if accession_nr == restatement_accession:
                    continue
                path_to_delete = os.path.join(root, file)
                try:
                    os.remove(path_to_delete)
                    deleted_files.append(path_to_delete)
                    print(f"Removed final 13F file: {path_to_delete}")
                except Exception as e:
                    print(f"Failed to remove {path_to_delete}: {e}")

    if not deleted_files:
        print(f"No previous 13F final file found for accession {restatement_accession}, {filer_name}, {year} {quarter}")

    return deleted_files



def combine_quarterly_files(folder_path, filer_name, cik_to_filer, CIK_TO_FINAL_DIR, BASE_DIR_FINAL, cusip_set, cusip_to_ticker):
    # Step 1: Group filenames by CONFORMED_DATE
    files_by_date = {}
    for fname in os.listdir(folder_path):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(folder_path, fname)
        df = pd.read_csv(fpath)
        # Use lower for robustness
        conformed_date = str(df.iloc[0]['conformed_date']) if 'conformed_date' in df.columns else str(df.iloc[0]['CONFORMED_DATE'])
        files_by_date.setdefault(conformed_date, []).append(fpath)

    # Step 2: For each conformed_date, merge and process the DataFrames
    for date, files in files_by_date.items():
        combined_df = pd.concat([pd.read_csv(f, dtype={'CUSIP': str}) for f in files], ignore_index=True)
        combined_df.columns = combined_df.columns.str.lower()  # Standardize column names

        # Filter for valid cusips
        combined_df = combined_df[combined_df['cusip'].astype(str).isin(cusip_set)].copy()

        # Filter out rows where put_call is 'Put' or 'Call'
        if 'put_call' in combined_df.columns:
            combined_df = combined_df[~combined_df['put_call'].isin(['Put', 'Call'])].copy()

        # Add ticker column
        combined_df['ticker'] = combined_df['cusip'].map(cusip_to_ticker)

        if combined_df.empty:
            print(f"No CUSIPs found for date {date}")
            continue

        # Group and sum
        grouped = (
            combined_df.groupby(['ticker', 'cusip'], as_index=False)
            .agg({'share_amount': 'sum', 'share_value': 'sum'})
        )

        output = grouped[['ticker', 'share_amount', 'share_value']]
        output_sorted = output.sort_values(by='share_value', ascending=False).reset_index(drop=True)
        output_sorted['rank'] = output_sorted.index + 1
        total_share_value = output_sorted['share_value'].sum()
        output_sorted['percentage'] = (output_sorted['share_value'] / total_share_value * 100).round(4)
        output_ranked = output_sorted[['rank', 'ticker', 'share_amount', 'share_value', 'percentage']]

        # --- For naming: pick the latest accession_number (lexically max) or join all
        accessions = set(combined_df['accession_number'])
        accession_nr = max(accessions)

        filed_date = date
        year = filed_date[:4]
        month = int(filed_date[4:6])
        quarter = (month - 1) // 3 + 1

        # Save
        aum_folder = CIK_TO_FINAL_DIR.get(list(cik_to_filer)[0])
        output_dir = Path(BASE_DIR_FINAL) / year / f"Q{quarter}" / aum_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{filer_name}_{year}_Q{quarter}-{accession_nr}.csv"
        output_path = output_dir / filename
        output_ranked.to_csv(output_path, index=False)
        print(f"Saved combined/filtered holdings for {year} Q{quarter} to {output_path}")


def delete_stale_13f_raw(cik, report_date):
    output_path = CIK_TO_PARSED_13F_DIR.get(cik)
    csv_dir = str(os.path.join(output_path, cik))

    # Find and delete any existing CSV in the folder with the same report_date on second row
    if os.path.isdir(csv_dir):
        for existing_file in os.listdir(csv_dir):
            existing_path = os.path.join(csv_dir, existing_file)
            if existing_path.endswith('.csv') and os.path.isfile(existing_path):
                try:
                    # Read second row (index 1) without header
                    existing_df = pd.read_csv(existing_path, nrows=1)
                    if not existing_df.empty:
                        existing_report_date = str(existing_df.iloc[0]['CONFORMED_DATE'])
                        if existing_report_date == str(report_date):
                            os.remove(existing_path)
                            print(f"Removed 13F file: {existing_path}")
                except Exception as e:
                    print(f"Warning: Could not read {existing_path} due to {e}. Skipping deletion.")


def generate_13f_and_add_chg_ownership_for_ciks(ciks: set[str] | list[str]):
    filtered_cik_to_filer = {cik: name for cik, name in CIK_TO_FILER.items() if cik in ciks}
    if filtered_cik_to_filer:
        _generate_13f_and_add_change_ownership_column(filtered_cik_to_filer)
    else:
        print("No matching CIKs found in this batch.")


def _generate_13f_and_add_change_ownership_column(cik_to_filer: dict[str, str], tickers=None):
    _generate_13f_csv(cik_to_filer)
    add_change_ownership_columns(cik_to_filer, tickers)
    add_quarter_end_price(cik_to_filer)


# convert raw csv to shorter version: rank,ticker,share_amount,share_value
def _generate_13f_csv(cik_to_filer: dict[str, str]):
    source_dir = CIK_TO_PARSED_13F_DIR.get(list(cik_to_filer)[0])
    for cik, filer_name in cik_to_filer.items():
        folder_path = os.path.join(source_dir, cik)  # Combine base_dir and CIK
        if not os.path.isdir(folder_path):
            continue

        # Replace the flat file loop with the combine-per-quarter logic!
        combine_quarterly_files(
            folder_path, filer_name,
            cik_to_filer,
            CIK_TO_FINAL_DIR, BASE_DIR_FINAL,
            cusip_set, cusip_to_ticker
        )


def load_shares_outstanding(file_path):
    shares_df = pd.read_csv(file_path, dtype={'year': str, 'quarter': str})
    shares_df = shares_df.dropna(subset=['year', 'quarter', 'outstanding_shares'])
    shares_df['shares_outstanding'] = pd.to_numeric(shares_df['outstanding_shares'], errors='coerce').astype('Int64')
    shares_df['year'] = shares_df['year'].astype(int)
    shares_df['quarter'] = shares_df['quarter'].str.upper()
    shares_df['ticker'] = shares_df['ticker'].astype(str).str.upper().str.strip()
    return shares_df


def read_and_normalize_csv(file_path):
    df = pd.read_csv(file_path)
    if 'ticker' not in df.columns:
        raise ValueError(f"'ticker' column missing in {file_path}")
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    return df


def get_prev_shares_map(prev_file_path):
    if not os.path.exists(prev_file_path):
        return {}
    df_prev = pd.read_csv(prev_file_path)
    if 'ticker' not in df_prev.columns or 'share_amount' not in df_prev.columns:
        return {}
    df_prev['ticker'] = df_prev['ticker'].astype(str).str.upper().str.strip()
    df_prev_filtered = df_prev[df_prev['share_amount'] > 0]
    return dict(zip(df_prev_filtered['ticker'], df_prev_filtered['share_amount']))


def calculate_changes(df_update, prev_map):
    def calc_change(row):
        prev_amt = prev_map.get(row['ticker'])
        if prev_amt is None or pd.isna(prev_amt):
            return "NEW", np.nan
        else:
            change_val = row['share_amount'] - prev_amt
            change_pct = change_val / prev_amt if prev_amt != 0 else np.nan
            return change_val, change_pct

    return df_update.apply(calc_change, axis=1, result_type='expand')


def add_change_ownership_columns(cik_to_filer, tickers=None):
    aum_folder = CIK_TO_FINAL_DIR.get(list(cik_to_filer)[0])
    shares_df = load_shares_outstanding(STOCKS_SHS_Q_END_PRICES_FILE)
    filers = cik_to_filer.values()
    filers_set = set(f.strip().upper() for f in filers)
    tickers_set = set(t.upper() for t in tickers) if tickers is not None else None

    quarters_order = ['Q4', 'Q3', 'Q2', 'Q1']
    years = sorted([d for d in os.listdir(BASE_DIR_FINAL) if d.isdigit()], reverse=True)

    for year in years:
        year_path = os.path.join(BASE_DIR_FINAL, year)
        if not os.path.isdir(year_path):
            continue

        for quarter in quarters_order:
            quarter_path = os.path.join(year_path, quarter)
            if not os.path.isdir(quarter_path):
                continue

            aum_path = os.path.join(quarter_path, aum_folder)
            if not os.path.isdir(aum_path):
                print(f"Warning: Expected filings directory missing: {aum_path}. No filings available "
                      f"for filer '{aum_folder}' in year {year}, quarter {quarter}. Continuing to next.")
                continue

            for fname in os.listdir(aum_path):
                if not fname.endswith('.csv'):
                    continue

                filer_name, file_year, file_quarter = parse_filer_info_from_filename(fname)
                if not filer_name or not file_year or not file_quarter:
                    print(f"Skipping file with unexpected name format: {fname}")
                    continue
                if filer_name.strip().upper() not in filers_set:
                    continue

                file_path = os.path.join(aum_path, fname)
                try:
                    df_cur = read_and_normalize_csv(file_path)

                    # Keep only rows with positive share_amount
                    df_cur = df_cur[df_cur['share_amount'] > 0]

                    if tickers_set is not None:
                        mask = df_cur['ticker'].str.upper().isin(tickers_set)
                    else:
                        mask = pd.Series(True, index=df_cur.index)

                    if not mask.any():
                        continue

                    df_update = df_cur.loc[mask].copy()

                    # Previous quarter info
                    prev_year, prev_quarter = get_prev_quarter(file_year, file_quarter)
                    prev_file_path = find_prev_quarter_file(aum_folder, filer_name, prev_year, prev_quarter)

                    prev_map = {}
                    if prev_file_path:
                        prev_map = get_prev_shares_map(prev_file_path)

                    # Add year and quarter columns for merging later
                    df_update['year'] = int(file_year)
                    df_update['quarter'] = f"Q{file_quarter}"

                    # Calculate changes (NEW, CLOSED_OUT logic removed)
                    df_update[['change', 'change_pct']] = calculate_changes(df_update, prev_map)

                    # Merge with shares outstanding to calculate ownership_pct
                    merged = df_update.merge(
                        shares_df,
                        on=['ticker', 'year', 'quarter'],
                        how='left',
                        validate='many_to_one'
                    )

                    merged['ownership_pct'] = (merged['share_amount'] / merged['outstanding_shares']) * 100

                    merged = merged.drop(columns=['year', 'quarter', 'outstanding_shares'])

                    df_update = merged

                    # Update df_cur inplace
                    for col in ['change', 'change_pct', 'ownership_pct']:
                        if col not in df_cur.columns:
                            if col == 'change':
                                df_cur[col] = pd.Series(dtype='object')
                            else:
                                df_cur[col] = np.nan
                        if col == 'change':
                            df_cur[col] = df_cur[col].astype(object)  # Important for mixed types
                        df_cur.loc[mask, col] = df_update[col].values

                    # Handle mixed types in 'change' column
                    df_cur['change'] = df_cur['change'].astype(object)

                    df_cur.to_csv(file_path, index=False)
                    print(f"Processed {fname} successfully, updated {mask.sum()} rows.")

                except Exception as e:
                    print(f"Error processing {fname}: {e}")


def replace_price(row, year, quarter):
    ticker = row['ticker'] if 'ticker' in row else None

    if not ticker or not year or not quarter:
        return row['quarter_end_price']  # no info to lookup

    dict_price = QUARTER_END_PRICE_DICT.get((ticker, year, quarter))

    return dict_price if dict_price is not None else row['quarter_end_price']


def add_quarter_end_price(cik_to_filer, base_dir=BASE_DIR_FINAL, threshold=0.5):
    filer_names = set(cik_to_filer.values())
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.endswith('.csv'):
                continue

            if not any(file.startswith(name) for name in filer_names):
                continue

            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            df['quarter_end_price'] = (df['share_value'] / df['share_amount']).round(2)

            prop_below_1 = (df['quarter_end_price'] < 1).mean()
            if prop_below_1 > threshold:
                year, quarter = extract_year_quarter_from_filename(file)
                df['quarter_end_price'] = df.apply(lambda row: replace_price(row, year, quarter), axis=1)
                print(f"replaced with dict prices: {file_path}")

            df.to_csv(file_path, index=False)


def load_data_by_year_quarter(base_dir, year, quarter=None):
    all_data = []
    year = str(year)  # ensure string for matching

    # List year folders, filter for the requested year only
    year_path = os.path.join(base_dir, year)
    if not os.path.isdir(year_path):
        raise ValueError(f"Year folder {year_path} does not exist")

    if quarter is not None:
        quarter_folders = [f"Q{quarter}"]
    else:
        # List all quarter folders inside the year folder (e.g., Q1, Q2, Q3, Q4)
        quarter_folders = [d for d in os.listdir(year_path)
                           if os.path.isdir(os.path.join(year_path, d)) and d.startswith('Q')]

    for q_folder in quarter_folders:
        quarter_path = os.path.join(year_path, q_folder)
        for fname in os.listdir(quarter_path):
            if not fname.endswith('.csv'):
                continue

            filer, file_year, file_quarter, _ = extract_filername_year_quarter_accession(fname)
            if file_year != year:
                continue
            if quarter is not None and file_quarter != str(quarter):
                continue

            file_path = os.path.join(quarter_path, fname)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            quarter_str = f"{file_year}_Q{file_quarter}"
            df['quarter'] = quarter_str
            df['filer'] = filer
            all_data.append(df)

    if not all_data:
        raise ValueError(f"No data found for year={year} quarter={quarter}")

    df_all = pd.concat(all_data, ignore_index=True)
    df_all['ticker'] = df_all['ticker'].astype(str).str.upper().str.strip()

    return df_all


def parse_filer_info_from_filename(filename):
    # Example: Bank_of_America_Corp_DE_2024_Q4-7085825000101.csv
    # Returns (filer_name, year, quarter)
    base = os.path.basename(filename)
    accession_nr_split = base.rsplit('-', 2)  # accession number = split by hyphen second elem
    parts = accession_nr_split[0].rsplit('_', 2)
    if len(parts) < 3:
        return None, None, None
    filer_name = parts[0]
    year = int(parts[1])
    quarter = int(parts[2].replace('Q', '').replace('.csv', ''))
    return filer_name, year, quarter


def get_prev_quarter(year, quarter):
    if quarter == 1:
        return year - 1, 4
    else:
        return year, quarter - 1


def find_prev_quarter_file(aum_folder: str, filer_name, prev_year, prev_quarter):
    prev_quarter_folder = f"Q{prev_quarter}"
    search_dir = os.path.join(BASE_DIR_FINAL, str(prev_year), prev_quarter_folder, aum_folder)

    # Pattern: filername_{year}_Q{quarter}-*.csv
    pattern = f"{filer_name}_{prev_year}_Q{prev_quarter}-*.csv"
    search_pattern = os.path.join(search_dir, pattern)

    matched_files = glob.glob(search_pattern)

    if not matched_files:
        print(f"No previous quarter file found for {filer_name} in {prev_year} Q{prev_quarter}")
        return None

    # If multiple matches (e.g., amendments), pick the latest by filename sorting
    matched_files.sort()
    return matched_files[-1]
