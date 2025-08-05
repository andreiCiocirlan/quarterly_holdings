import os
from collections import defaultdict

import pandas as pd

from utils.data_loader import load_cik_to_filer_and_aum, filter_cik_by_aum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATA_EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NASDAQ_FILE_PATH = os.path.join(DATA_EXTERNAL_DIR, 'nasdaq_listings_full.csv')
NYSE_FILE_PATH = os.path.join(DATA_EXTERNAL_DIR, 'nyse_listings_full.csv')
ALL_FILERS_CSV = os.path.join(DATA_EXTERNAL_DIR, 'all_filers.csv')
ALL_OWNERSHIP_2024_Q4_CSV = os.path.join(DATA_EXTERNAL_DIR, 'all_ownership_2024_Q4.csv')
ALL_OWNERSHIP_2025_Q1_CSV = os.path.join(DATA_EXTERNAL_DIR, 'all_ownership_2025_Q1.csv')
TOP_100_ADDED_TICKERS_CSV = os.path.join(DATA_EXTERNAL_DIR, "top_100_added.csv")
TOP_100_REDUCED_TICKERS_CSV = os.path.join(DATA_EXTERNAL_DIR, "top_100_reduced.csv")
OVER_100_OWNERSHIP_TICKERS_CSV = os.path.join(DATA_EXTERNAL_DIR, "over_100_ownership.csv")
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")
FILER_ACCESSION_METADATA = os.path.join(DATA_EXTERNAL_DIR, 'filer_accession_metadata.csv')
SUBMISSIONS_FILERS_DIR = os.path.join(DATA_DIR, "submissions", "filers")
SUBMISSIONS_STOCKS_DIR = os.path.join(DATA_DIR, "submissions", "stocks")
STOCKS_SHS_Q_END_PRICES_FILE = os.path.join(DATA_EXTERNAL_DIR, 'stock_shs_q_end_prices.csv')

BASE_DIR_DATA_PARSE = os.path.join(DATA_DIR, 'parsed')
RAW_13F_PARSED_HOLDINGS_OVER_250B = os.path.join(BASE_DIR_DATA_PARSE, 'raw_13f_aum_over_250b')
RAW_13F_PARSED_HOLDINGS_50B_TO_250B = os.path.join(BASE_DIR_DATA_PARSE, 'raw_13f_aum_50b_to_250b')
RAW_13F_PARSED_HOLDINGS_25B_TO_50B = os.path.join(BASE_DIR_DATA_PARSE, 'raw_13f_aum_25b_to_50b')
RAW_13F_PARSED_HOLDINGS_10B_TO_25B = os.path.join(BASE_DIR_DATA_PARSE, 'raw_13f_aum_10b_to_25b')
RAW_13F_PARSED_HOLDINGS_5B_TO_10B = os.path.join(BASE_DIR_DATA_PARSE, 'raw_13f_aum_5b_to_10b')
RAW_13F_PARSED_HOLDINGS_3B_TO_5B = os.path.join(BASE_DIR_DATA_PARSE, 'raw_13f_aum_3b_to_5b')
RAW_13F_PARSED_HOLDINGS_1B_TO_3B = os.path.join(BASE_DIR_DATA_PARSE, 'raw_13f_aum_1b_to_3b')
BASE_DIR_FINAL = os.path.join(DATA_DIR, 'final')

HEADERS = {'User-Agent': 'test-email@example.com'}

RAW_PARSED_HOLDINGS_DIRECTORIES = [
    RAW_13F_PARSED_HOLDINGS_OVER_250B,
    RAW_13F_PARSED_HOLDINGS_50B_TO_250B,
    RAW_13F_PARSED_HOLDINGS_25B_TO_50B,
    RAW_13F_PARSED_HOLDINGS_10B_TO_25B,
    RAW_13F_PARSED_HOLDINGS_5B_TO_10B,
    RAW_13F_PARSED_HOLDINGS_3B_TO_5B,
    RAW_13F_PARSED_HOLDINGS_1B_TO_3B
]


def _build_cik_to_accession():
    cik_to_accessions = defaultdict(list)

    for base_dir in RAW_PARSED_HOLDINGS_DIRECTORIES:
        if not os.path.isdir(base_dir):
            continue
        for cik_folder in os.listdir(base_dir):
            cik_path = os.path.join(base_dir, cik_folder)
            if os.path.isdir(cik_path):
                for filename in os.listdir(cik_path):
                    if filename.endswith(".csv"):
                        accession_nr = filename[:-4]  # Remove '.csv'
                        cik_to_accessions[cik_folder].append(accession_nr)

    return dict(cik_to_accessions)


def _build_ticker_year_quarter_dict_for(column):
    if column is None:
        raise ValueError("You must specify a column name")

    # Read CSV with required columns (including the requested one)
    usecols = ['ticker', 'year', 'quarter', column]
    df = pd.read_csv(STOCKS_SHS_Q_END_PRICES_FILE, usecols=usecols, dtype={'ticker': str, 'year': str, 'quarter': str})

    # Clean missing values in the column — convert NaN to None
    df[column] = df[column].where(pd.notnull(df[column]), None)

    # Build dictionary keyed by (ticker, year, quarter)
    result_dict = {
        (row['ticker'], row['year'], row['quarter']): row[column]
        for _, row in df.iterrows()
    }

    return result_dict


def _build_cik_accession_filing_date_dict():
    # Read only necessary columns
    usecols = ['cik', 'accession_nr', 'filing_date']
    df = pd.read_csv(FILER_ACCESSION_METADATA, usecols=usecols, dtype={'cik': str, 'accession_nr': str})

    # Normalize filing_date: treat empty strings or NaN as None
    df['filing_date'] = df['filing_date'].replace("", pd.NA)
    df['filing_date'] = df['filing_date'].where(pd.notnull(df['filing_date']), None)

    result_dict = {
        (row['cik'], row['accession_nr']): row['filing_date']
        for _, row in df.iterrows()
    }
    return result_dict


def _initialize_mappings():
    cik_to_filer_and_aum = load_cik_to_filer_and_aum(ALL_FILERS_CSV)

    cik_to_filer_and_aum_1b_to_3b = filter_cik_by_aum(cik_to_filer_and_aum, 1, 2.99)
    cik_to_filer_and_aum_3b_to_5b = filter_cik_by_aum(cik_to_filer_and_aum, 3, 4.99)
    cik_to_filer_and_aum_5b_to_10b = filter_cik_by_aum(cik_to_filer_and_aum, 5, 9.99)
    cik_to_filer_and_aum_10b_to_25b = filter_cik_by_aum(cik_to_filer_and_aum, 10, 24.99)
    cik_to_filer_and_aum_25b_to_50b = filter_cik_by_aum(cik_to_filer_and_aum, 25, 49.99)
    cik_to_filer_and_aum_50b_to_250b = filter_cik_by_aum(cik_to_filer_and_aum, 50, 249.99)
    cik_to_filer_and_aum_over_250b = filter_cik_by_aum(cik_to_filer_and_aum, 250, 10000)

    cik_to_filer = {cik: info[0] for cik, info in cik_to_filer_and_aum.items()}
    cik_to_filer_1b_to_3b = {cik: info[0] for cik, info in cik_to_filer_and_aum_1b_to_3b.items()}
    cik_to_filer_3b_to_5b = {cik: info[0] for cik, info in cik_to_filer_and_aum_3b_to_5b.items()}
    cik_to_filer_5b_to_10b = {cik: info[0] for cik, info in cik_to_filer_and_aum_5b_to_10b.items()}
    cik_to_filer_10b_to_25b = {cik: info[0] for cik, info in cik_to_filer_and_aum_10b_to_25b.items()}
    cik_to_filer_25b_to_50b = {cik: info[0] for cik, info in cik_to_filer_and_aum_25b_to_50b.items()}
    cik_to_filer_50b_to_250b = {cik: info[0] for cik, info in cik_to_filer_and_aum_50b_to_250b.items()}
    cik_to_filer_over_250b = {cik: info[0] for cik, info in cik_to_filer_and_aum_over_250b.items()}

    quarter_end_price_dict = _build_ticker_year_quarter_dict_for('quarter_end_price')
    outstanding_shares_dict = _build_ticker_year_quarter_dict_for('outstanding_shares')
    cik_accession_to_filing_date = _build_cik_accession_filing_date_dict()

    cik_to_accessions = _build_cik_to_accession()

    return {
        "CIK_TO_FILER_AND_AUM": cik_to_filer_and_aum,
        "CIK_TO_FILER_AND_AUM_1B_TO_3B": cik_to_filer_and_aum_1b_to_3b,
        "CIK_TO_FILER_AND_AUM_3B_TO_5B": cik_to_filer_and_aum_3b_to_5b,
        "CIK_TO_FILER_AND_AUM_5B_TO_10B": cik_to_filer_and_aum_5b_to_10b,
        "CIK_TO_FILER_AND_AUM_10B_TO_25B": cik_to_filer_and_aum_10b_to_25b,
        "CIK_TO_FILER_AND_AUM_25B_TO_50B": cik_to_filer_and_aum_25b_to_50b,
        "CIK_TO_FILER_AND_AUM_50B_TO_250B": cik_to_filer_and_aum_50b_to_250b,
        "CIK_TO_FILER_AND_AUM_OVER_250B": cik_to_filer_and_aum_over_250b,
        "CIK_TO_FILER": cik_to_filer,
        "CIK_TO_FILER_1B_TO_3B": cik_to_filer_1b_to_3b,
        "CIK_TO_FILER_3B_TO_5B": cik_to_filer_3b_to_5b,
        "CIK_TO_FILER_5B_TO_10B": cik_to_filer_5b_to_10b,
        "CIK_TO_FILER_10B_TO_25B": cik_to_filer_10b_to_25b,
        "CIK_TO_FILER_25B_TO_50B": cik_to_filer_25b_to_50b,
        "CIK_TO_FILER_50B_TO_250B": cik_to_filer_50b_to_250b,
        "CIK_TO_FILER_OVER_250B": cik_to_filer_over_250b,

        "QUARTER_END_PRICE_DICT": quarter_end_price_dict,
        "OUTSTANDING_SHARES_DICT": outstanding_shares_dict,
        "CIK_TO_ACCESSIONS": cik_to_accessions,
        "CIK_ACCESSION_TO_FILING_DATE": cik_accession_to_filing_date

    }


_mappings = _initialize_mappings()


CIK_TO_FILER_AND_AUM = _mappings["CIK_TO_FILER_AND_AUM"]

CIK_TO_FILER_AND_AUM_1B_TO_3B = _mappings["CIK_TO_FILER_AND_AUM_1B_TO_3B"]
CIK_TO_FILER_AND_AUM_3B_TO_5B = _mappings["CIK_TO_FILER_AND_AUM_3B_TO_5B"]
CIK_TO_FILER_AND_AUM_5B_TO_10B = _mappings["CIK_TO_FILER_AND_AUM_5B_TO_10B"]
CIK_TO_FILER_AND_AUM_10B_TO_25B = _mappings["CIK_TO_FILER_AND_AUM_10B_TO_25B"]
CIK_TO_FILER_AND_AUM_25B_TO_50B = _mappings["CIK_TO_FILER_AND_AUM_25B_TO_50B"]
CIK_TO_FILER_AND_AUM_50B_TO_250B = _mappings["CIK_TO_FILER_AND_AUM_50B_TO_250B"]
CIK_TO_FILER_AND_AUM_OVER_250B = _mappings["CIK_TO_FILER_AND_AUM_OVER_250B"]
CIK_TO_FILER = _mappings["CIK_TO_FILER"]
CIK_TO_FILER_1B_TO_3B = _mappings["CIK_TO_FILER_1B_TO_3B"]
CIK_TO_FILER_3B_TO_5B = _mappings["CIK_TO_FILER_3B_TO_5B"]
CIK_TO_FILER_5B_TO_10B = _mappings["CIK_TO_FILER_5B_TO_10B"]
CIK_TO_FILER_10B_TO_25B = _mappings["CIK_TO_FILER_10B_TO_25B"]
CIK_TO_FILER_25B_TO_50B = _mappings["CIK_TO_FILER_25B_TO_50B"]
CIK_TO_FILER_50B_TO_250B = _mappings["CIK_TO_FILER_50B_TO_250B"]
CIK_TO_FILER_OVER_250B = _mappings["CIK_TO_FILER_OVER_250B"]
QUARTER_END_PRICE_DICT = _mappings["QUARTER_END_PRICE_DICT"]
OUTSTANDING_SHARES_DICT = _mappings["OUTSTANDING_SHARES_DICT"]
CIK_TO_ACCESSIONS = _mappings["CIK_TO_ACCESSIONS"]
CIK_ACCESSION_TO_FILING_DATE = _mappings["CIK_ACCESSION_TO_FILING_DATE"]
FILERNAME_TO_CIK = {v[0]: k for k, v in CIK_TO_FILER_AND_AUM.items()}

FILER_NAMES = tuple(formatted_name for formatted_name, _ in CIK_TO_FILER_AND_AUM.values())
CIK_FINAL_DIR_PAIRS = [
    (CIK_TO_FILER_1B_TO_3B, 'aum_1b_to_3b'),
    (CIK_TO_FILER_3B_TO_5B, 'aum_3b_to_5b'),
    (CIK_TO_FILER_5B_TO_10B, 'aum_5b_to_10b'),
    (CIK_TO_FILER_10B_TO_25B, 'aum_10b_to_25b'),
    (CIK_TO_FILER_25B_TO_50B, 'aum_25b_to_50b'),
    (CIK_TO_FILER_50B_TO_250B, 'aum_50b_to_250b'),
    (CIK_TO_FILER_OVER_250B, 'aum_over_250b'),
]

CIK_PARSED_DIR_PAIRS = [
    (CIK_TO_FILER_1B_TO_3B, RAW_13F_PARSED_HOLDINGS_1B_TO_3B),
    (CIK_TO_FILER_3B_TO_5B, RAW_13F_PARSED_HOLDINGS_3B_TO_5B),
    (CIK_TO_FILER_5B_TO_10B, RAW_13F_PARSED_HOLDINGS_5B_TO_10B),
    (CIK_TO_FILER_10B_TO_25B, RAW_13F_PARSED_HOLDINGS_10B_TO_25B),
    (CIK_TO_FILER_25B_TO_50B, RAW_13F_PARSED_HOLDINGS_25B_TO_50B),
    (CIK_TO_FILER_50B_TO_250B, RAW_13F_PARSED_HOLDINGS_50B_TO_250B),
    (CIK_TO_FILER_OVER_250B, RAW_13F_PARSED_HOLDINGS_OVER_250B)
]


def _build_cik_to_final_dir():
    cik_to_final_dir = {}
    for cik_dict, base_dir in CIK_FINAL_DIR_PAIRS:
        cik_to_final_dir.update({cik: base_dir for cik in cik_dict.keys()})
    return cik_to_final_dir


def _build_cik_to_parsed_dir():
    cik_to_final_dir = {}
    for cik_dict, base_dir in CIK_PARSED_DIR_PAIRS:
        cik_to_final_dir.update({cik: base_dir for cik in cik_dict.keys()})
    return cik_to_final_dir


CIK_TO_FINAL_DIR = _build_cik_to_final_dir()
CIK_TO_PARSED_13F_DIR = _build_cik_to_parsed_dir()
