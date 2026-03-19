import csv
import glob
import os
import shutil
import tempfile
from datetime import datetime

import pandas as pd
from sec_api import MappingApi

from utils.mappings import NYSE_FILE_PATH, NASDAQ_FILE_PATH, RAW_PARSED_HOLDINGS_DIRECTORIES

api_key = "0c6d75ca6a92cd650f0dd82b8325457177e78f071c65f1554884685b2a2d0617"
mappingApi = MappingApi(api_key=api_key)

_ALLOWED_KEYWORDS = ['Common Stock']
_EXCLUDE_KEYWORDS = ['Warrant', 'Preferred', 'ETF', 'ETN', 'CEF', 'UNIT', 'ETD', 'ETMF']


def _load_listings(nasdaq_path, nyse_path):
    nasdaq_df = pd.read_csv(nasdaq_path, dtype=str)
    nyse_df = pd.read_csv(nyse_path, dtype=str)
    combined_df = pd.concat([nasdaq_df, nyse_df], ignore_index=True)
    filtered_df = combined_df[combined_df['isDelisted'].astype(str).str.lower() != 'true']
    filtered_df = filtered_df[filtered_df['cik'].notna() & (filtered_df['cik'].str.strip() != '')]

    return filtered_df


def _is_tradeable_stock(category):
    if not any(keyword in category for keyword in _ALLOWED_KEYWORDS):
        return False
    if any(keyword in category for keyword in _EXCLUDE_KEYWORDS):
        return False
    return True


def _get_filtered_listings(nasdaq_path, nyse_path):
    listings_df = _load_listings(nasdaq_path, nyse_path)
    filtered_df = listings_df[listings_df['category'].apply(_is_tradeable_stock)]
    return filtered_df


def _get_cik_to_ticker(filtered_df):
    return {
        row['cik']: row['ticker']
        for _, row in filtered_df.iterrows()
        if not pd.isna(row['cik']) and row['cik'] != '' and row['cik'] is not None
    }


def _get_ticker_to_cik(filtered_df):
    return {
        row['ticker']: row['cik']
        for _, row in filtered_df.iterrows()
        if not pd.isna(row['cik']) and row['cik'] != '' and row['cik'] is not None
    }


def _get_cusip_to_ticker(filtered_df):
    cusip_to_ticker = {}
    for _, row in filtered_df.iterrows():
        ticker = row['ticker']
        cusip_field = row.get('cusip')
        if pd.notna(cusip_field):
            for c in str(cusip_field).split():
                cusip_to_ticker[c.strip()] = ticker
    return cusip_to_ticker


def _get_cusip_set(cusip_to_ticker):
    return set(cusip_to_ticker.keys())


filtered_listings = _get_filtered_listings(NASDAQ_FILE_PATH, NYSE_FILE_PATH)
cik_to_ticker = _get_cik_to_ticker(filtered_listings)
ticker_to_cik = _get_ticker_to_cik(filtered_listings)
cusip_to_ticker = _get_cusip_to_ticker(filtered_listings)
cusip_set = _get_cusip_set(cusip_to_ticker)


def update_all_exchange_cusips_from_raw_holdings(cutoff_date="2025-12-31"):
    """
    TWO-PASS approach using YOUR existing _extract_new_common_stock_cusips:
    1. Extract ALL new CUSIPs from every 13F file (modified after cutoff_date 2025-12-31)
    2. Single final update of NASDAQ/NYSE files
    """
    print("🔍 PASS 1: Extracting new CUSIPs from all 13F files...")

    # STEP 1: Extract ALL new CUSIPs
    all_new_cusips_by_issuer = {}
    total_files_processed = 0
    cutoff_timestamp = datetime.strptime(cutoff_date, "%Y-%m-%d").timestamp()

    for directory in RAW_PARSED_HOLDINGS_DIRECTORIES:
        if not os.path.exists(directory):
            print(f"⚠️  Skipping {os.path.basename(directory)}")
            continue

        csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
        recent_csv_files = [f for f in csv_files if os.path.getctime(f) > cutoff_timestamp]

        if not recent_csv_files:
            continue

        print(f"📁 Scanning {os.path.basename(directory)} ({len(recent_csv_files)} recent files)...")

        for csv_file in recent_csv_files:
            try:
                file_new_cusips = _extract_new_common_stock_cusips(csv_file, known_cusips=cusip_set)

                # Merge into master dict (deduplicate across files)
                for issuer, cusips in file_new_cusips.items():
                    if issuer not in all_new_cusips_by_issuer:
                        all_new_cusips_by_issuer[issuer] = set()
                    all_new_cusips_by_issuer[issuer].update(cusips)

                total_files_processed += 1

            except Exception as e:
                print(f"❌ Error in {os.path.basename(csv_file)}: {str(e)}")
                continue

    # Report results
    total_new_cusips = sum(len(cusips) for cusips in all_new_cusips_by_issuer.values())
    print("\n" + "="*60)
    print(f"✅ PASS 1 COMPLETE")
    print(f"   Files scanned: {total_files_processed}")
    print(f"   Issuers with new CUSIPs: {len(all_new_cusips_by_issuer)}")
    print(f"   Total new CUSIPs: {total_new_cusips}")
    print("="*60)

    if not all_new_cusips_by_issuer:
        print("ℹ️  No new CUSIPs found.")
        return

    # STEP 2: Single final update using YOUR _update_exchange_file
    print("\n🔧 PASS 2: Final NASDAQ/NYSE update...")
    nasdaq_updates = _update_exchange_file(all_new_cusips_by_issuer, NASDAQ_FILE_PATH)
    nyse_updates = _update_exchange_file(all_new_cusips_by_issuer, NYSE_FILE_PATH)

    print("\n🎉 BULK UPDATE COMPLETE")
    print(f"NASDAQ: {nasdaq_updates} rows | NYSE: {nyse_updates} rows")


def _extract_new_common_stock_cusips(f_path_13f, known_cusips):
    """Extract CUSIPs for common stock (COM/COM NEW) from 13F for existing issuers."""
    existing_issuers = set(filtered_listings['name'].astype(str).str.strip().str.upper())
    new_cusips_by_issuer = {}

    with open(f_path_13f, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get('TITLE_OF_CLASS', '').strip().upper()
            issuer = row.get('NAME_OF_ISSUER', '').strip().upper()
            cusip = row.get('CUSIP', '').strip()

            if not (title.startswith('COM') or title == 'COM NEW'):
                continue

            if issuer in existing_issuers and cusip and cusip not in known_cusips:
                new_cusips_by_issuer.setdefault(issuer, set()).add(cusip)

    return new_cusips_by_issuer


def _update_exchange_file(new_cusips_by_issuer, exchange_path):
    """Safely append new CUSIPs to exchange file (tradeable stocks only)."""
    exchange_df = pd.read_csv(exchange_path)

    # Preserve integer columns
    for col in ['cik', 'sic']:
        if col in exchange_df.columns:
            exchange_df[col] = pd.to_numeric(exchange_df[col], errors='coerce').astype('Int64')

    # Filter: tradeable + known tickers
    approved_mask = (
            exchange_df['category'].apply(_is_tradeable_stock) &
            exchange_df['ticker'].isin(ticker_to_cik.keys())
    )
    approved_df = exchange_df[approved_mask]

    updates_made = 0
    for issuer, new_cusips in new_cusips_by_issuer.items():
        matches = approved_df[approved_df['name'].str.strip().str.upper() == issuer]
        for idx in matches.index:
            ticker = approved_df.at[idx, 'ticker']
            if ticker not in ticker_to_cik:
                continue

            current = set(str(exchange_df.at[idx, 'cusip']).strip().split())
            exchange_df.at[idx, 'cusip'] = ' '.join(sorted(current | new_cusips))
            updates_made += 1
            print(f"✅ {issuer} ({ticker}): +{len(new_cusips)} CUSIPs")

    if updates_made:
        _atomic_csv_write(exchange_df, exchange_path)
        print(f"✅ Updated {exchange_path} ({updates_made} rows)")
    else:
        print(f"No updates needed for {exchange_path}")

    return updates_made


def _atomic_csv_write(df, output_path):
    """Write DataFrame to CSV atomically - FIXED for Windows."""
    # Create temp file in same directory to avoid permission issues
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.csv', dir=os.path.dirname(output_path))

    try:
        # Write to temp file
        df.to_csv(tmp_path, index=False)
    finally:
        # CRITICAL: Close and delete temp file handle immediately
        os.close(tmp_fd)

    # Atomic move - now safe
    shutil.move(tmp_path, output_path)

def fetch_and_save_cusip_data():
    # Fetch all NASDAQ listings
    nasdaq_companies = mappingApi.resolve('exchange', 'NASDAQ')
    nasdaq_df = pd.DataFrame(nasdaq_companies)

    # Save NASDAQ data to CSV (all columns)
    nasdaq_df.to_csv(NASDAQ_FILE_PATH, index=False)
    print(f"Saved NASDAQ listings ({len(nasdaq_df)}) to nasdaq_listings_full.csv")

    # Fetch all NYSE listings
    nyse_companies = mappingApi.resolve('exchange', 'NYSE')
    nyse_df = pd.DataFrame(nyse_companies)

    # Save NYSE data to CSV (all columns)
    nyse_df.to_csv(NYSE_FILE_PATH, index=False)
    print(f"Saved NYSE listings ({len(nyse_df)}) to nyse_listings_full.csv")


def main():
    fetch_and_save_cusip_data()
    # TODO: delete row with CIK = 1747777 from NYSE file (it doesn't have a category)


if __name__ == "__main__":
    main()
