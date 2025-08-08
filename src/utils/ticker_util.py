import json
import os
import re
from collections import defaultdict, Counter

import pandas as pd

from init_setup.ticker_cusip_data import cik_to_ticker
from utils.date_util import get_year_and_quarter
from utils.mappings import QUARTER_END_PRICE_DICT, SUBMISSIONS_STOCKS_DIR


def find_tickers_with_10qks(year='2025', quarter='Q1', cik_list=None):
    tickers_found = []
    year = str(year)
    quarter = quarter.upper()

    if cik_list is not None:
        filtered_ciks = []
        for cik in cik_list:
            cik_str = str(int(cik))
            ticker = cik_to_ticker.get(cik_str)
            if not ticker:
                continue  # skip if ticker mapping missing
            key = f"{ticker}_{year}_{quarter}"
            if key in QUARTER_END_PRICE_DICT:
                print(f'already found {ticker} {year} {quarter}')
                continue  # skip CIK/ticker already found in your dict
            filtered_ciks.append(cik_str)
        cik_set = set(filtered_ciks)
    else:
        cik_set = None

    for filename in os.listdir(SUBMISSIONS_STOCKS_DIR):
        if not filename.endswith('.json'):
            continue

        cik_match = re.match(r'CIK0*([0-9]+)\.json', filename)
        if not cik_match:
            continue
        cik_str = cik_match.group(1)

        if cik_set is not None and cik_str not in cik_set:
            continue  # skip files outside the requested CIks

        filepath = os.path.join(SUBMISSIONS_STOCKS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        report_dates = filings.get("reportDate", [])

        for form, report_date in zip(forms, report_dates):
            form_upper = form.upper()
            if form_upper not in ("10-Q", "10-K"):
                continue

            filing_year, filing_quarter = get_year_and_quarter(report_date)
            if filing_year == year and filing_quarter == quarter:
                tickers_found.append(cik_to_ticker.get(cik_str))
                break  # no need to check more filings for this CIK

    return tickers_found


def get_tickers_missing_price(tickers, year, quarter, price_dict):
    key_tuples = {(ticker.strip().upper(), str(year), quarter) for ticker in tickers}
    existing_keys = set(price_dict.keys())
    missing_keys = key_tuples - existing_keys

    # Extract just tickers from missing keys
    missing_tickers = {k[0] for k in missing_keys}

    return list(missing_tickers)


def has_q_end_price(ticker, year, quarter):
    if not isinstance(ticker, str):
        return False
    key = (ticker.strip().upper(), str(year), quarter)
    return key in QUARTER_END_PRICE_DICT and QUARTER_END_PRICE_DICT[key] is not None


def get_prices_for_all_quarters(base_dir, year_quarter_list, tickers_to_include=None):
    records = []

    for year, quarter in year_quarter_list:
        ticker_prices = get_prices_per_ticker(base_dir, year, quarter)  # Your existing function

        # Filter tickers with min count >= 10 and by tickers_to_include if provided
        ticker_prices_filtered = {
            ticker: counts
            for ticker, counts in ticker_prices.items()
            if sum(counts.values()) >= 10 and (tickers_to_include is None or ticker in tickers_to_include)
        }

        most_freq_price = get_most_frequent_price_per_ticker(ticker_prices_filtered)
        print(f'Did not update {year} {quarter} tickers: {tickers_to_include - set(ticker_prices_filtered.keys())}')

        for ticker, price in most_freq_price.items():
            records.append({'ticker': ticker, 'year': year, 'quarter': quarter, 'quarter_end_price': price})

    return pd.DataFrame(records)


def get_prices_per_ticker(base_dir, year='2025', quarter='Q1'):
    ticker_prices = defaultdict(Counter)

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and f'{year}_{quarter}' in file:
                file_path = os.path.join(root, file)

                try:
                    df = pd.read_csv(file_path)
                    df = df[df['share_amount'] >= 1000]

                    # Calculate quarter_end_price as before
                    df['quarter_end_price'] = (df['share_value'] / df['share_amount']).round(2)

                    for ticker, price in zip(df['ticker'], df['quarter_end_price']):
                        ticker_prices[ticker][price] += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Filter tickers with more than one unique price
    filtered_ticker_prices = {ticker: prices
                              for ticker, prices in ticker_prices.items()
                              if len(prices) > 1}

    # Sort tickers by max count of any price descending
    sorted_ticker_prices = dict(
        sorted(
            filtered_ticker_prices.items(),
            key=lambda item: max(item[1].values()),
            reverse=True
        )
    )
    return sorted_ticker_prices


def check_price_discrepancies_simple(base_dir):
    pattern = re.compile(r'.*_(\d{4})_(Q[1-4]).*')

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith('.csv'):
                continue

            match = pattern.search(file)
            if not match:
                print(f'mismatch pattern for {file}')
                continue

            year, quarter = match.group(1), match.group(2)
            file_path = os.path.join(root, file)
            print(f'processing {file_path}')

            try:
                df = pd.read_csv(file_path, usecols=['ticker', 'quarter_end_price'])
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                continue

            for _, row in df.iterrows():
                ticker = row['ticker']
                file_price = row['quarter_end_price']

                key = f"{ticker}_{year}_{quarter}"
                dict_price = QUARTER_END_PRICE_DICT.get(ticker, year, quarter)

                # simple comparison handling NaN/None
                if dict_price is None and pd.notnull(file_price):
                    print(f"{key}: Missing in dict, file has {file_price}")
                elif dict_price is not None and pd.isnull(file_price):
                    print(f"{key}: Present in dict {dict_price}, missing in file")
                elif dict_price is not None and pd.notnull(file_price):
                    try:
                        if abs(float(dict_price) - float(file_price)) > 1e-4:
                            print(f"{key}: Price mismatch dict={dict_price} file={file_price}")
                    except:
                        print(f"{key}: Price conversion error dict={dict_price} file={file_price}")


def get_most_frequent_price_per_ticker(ticker_prices):
    most_freq_price = {}
    for ticker, price_counts in ticker_prices.items():
        # get top 2 in case first is 0.0
        common_prices = price_counts.most_common(2)
        price = (
            common_prices[1][0]
            if common_prices and common_prices[0][0] == 0.0 and len(common_prices) > 1 and common_prices[1][0] != 0.0
            else common_prices[0][0]
            if common_prices else None
        )
        if price == 0.0:
            print(f"Warning: ticker {ticker} has no non-zero prices with counts {price_counts}")
        if price is not None:
            most_freq_price[ticker] = price
    return most_freq_price


def get_quarter_end_price(ticker, year, quarter):
    price = QUARTER_END_PRICE_DICT.get(ticker, year, quarter)
    if price is None:
        print(f"No data found for ticker={ticker}, year={year}, quarter={quarter}")
    return price
