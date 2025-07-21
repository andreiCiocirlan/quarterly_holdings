import pandas as pd
from sec_api import MappingApi

from utils.mappings import NYSE_FILE_PATH, NASDAQ_FILE_PATH

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


filtered_listings = _get_filtered_listings(NASDAQ_FILE_PATH, NASDAQ_FILE_PATH)
cik_to_ticker = _get_cik_to_ticker(filtered_listings)
ticker_to_cik = _get_ticker_to_cik(filtered_listings)
cusip_to_ticker = _get_cusip_to_ticker(filtered_listings)
cusip_set = _get_cusip_set(cusip_to_ticker)


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
