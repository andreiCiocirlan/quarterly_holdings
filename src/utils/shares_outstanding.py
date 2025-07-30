import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta, datetime

import pandas as pd
import requests
from lxml import etree

from cfg.cfg_requests import limited_get
from init_setup.ticker_cusip_data import ticker_to_cik, cik_to_ticker
from utils.date_util import get_year_and_quarter
from utils.filings_util import get_prev_quarter
from utils.mappings import STOCKS_SHS_Q_END_PRICES_FILE, BASE_DIR_FINAL, SUBMISSIONS_STOCKS_DIR, HEADERS
from utils.ticker_util import get_prices_for_all_quarters, has_q_end_price

STANDARD_MEMBERS = {'CommonStockMember', 'CommonClassAMember'}
EXCLUDED_MEMBERS = {'CommonClassBMember'}

PRIORITY_MAP = {
    ('EntityCommonStockSharesOutstanding', 'CommonStockMember'): 0,
    ('EntityCommonStockSharesOutstanding', 'ANY_CUSTOM_MEMBER'): 2,
    ('EntityCommonStockSharesOutstanding', None): 1,
    ('CommonStockSharesOutstanding', 'CommonStockMember'): 3,
    ('CommonStockSharesOutstanding', 'ANY_CUSTOM_MEMBER'): 4,
    ('CommonStockSharesOutstanding', None): 5,
    ('EntityCommonStockSharesOutstanding', 'CommonClassAMember'): 6,
    ('CommonStockSharesOutstanding', 'CommonClassAMember'): 7,
}


def strip_namespace(tag):
    # Remove namespace if present, keep prefix if any (e.g. 'dei:EntityCommonStockSharesOutstanding')
    if '}' in tag:
        return tag.split('}', 1)[1]
    else:
        return tag


def get_next_quarter(year, quarter):
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    idx = quarters.index(quarter)
    if idx == 3:  # Q4
        return year + 1, 'Q1'
    else:
        return year, quarters[idx + 1]


def get_document_period_end_date(root):
    # Find the DocumentPeriodEndDate
    for elem in root.iter():
        if strip_namespace(elem.tag) == 'DocumentPeriodEndDate':
            return elem.text
    return None


def get_report_date_interval(date_str):
    """
    Given a report_date (datetime or date), return (start_date, end_date)
    where:
    - start_date is the first day of the quarter containing report_date
    - end_date is the last day of the month 5 months after start_date
    """
    report_date = datetime.strptime(date_str, "%Y-%m-%d")
    month = report_date.month
    year = report_date.year

    # Calculate quarter start month
    quarter_start_month = 3 * ((month - 1) // 3) + 1
    start_date = date(year, quarter_start_month, 1)

    # Calculate month and year for end date (start_date + 5 months)
    end_month = quarter_start_month + 5
    end_year = year
    if end_month > 12:
        end_month -= 12
        end_year += 1

    # Find last day of end_month
    # Method: first day of next month minus one day
    if end_month == 12:
        next_month_first_day = date(end_year + 1, 1, 1)
    else:
        next_month_first_day = date(end_year, end_month + 1, 1)

    end_date = next_month_first_day - timedelta(days=1)

    return start_date, end_date


def context_contains_member(context, member_value):
    """
    Return True if member_value appears anywhere in element text or attribute values in the context.
    """
    for elem in context.iter():
        if elem.text and member_value in elem.text:
            return True
        if any(member_value in attr_val for attr_val in elem.attrib.values()):
            return True
    return False


def context_has_valid_member(root, context_id, required_member):
    context = root.find(f".//{{*}}context[@id='{context_id}']")
    if context is None:
        return False

    for parent_tag in ['segment', 'scenario']:
        parent = context.find(f".//{{*}}{parent_tag}")
        if parent is not None:
            if context_contains_member(parent, required_member):
                return True
    return False


def extract_shares(sh_tag, min_plausible_shares=1_000_000):
    try:
        value_text = sh_tag.text
        if not value_text:
            return None
        value_text = value_text.replace(',', '').strip()
        value_int = int(value_text)
        if value_int < min_plausible_shares:
            return None  # Ignore implausibly low values
        return value_int

    except Exception:
        return None


def get_member_key_for_ticker(xml_content, ticker):
    root = ET.fromstring(xml_content)
    ticker_lower = ticker.lower()

    titles = root.findall(f".//{{*}}Security12bTitle")
    symbols = root.findall(f".//{{*}}TradingSymbol")

    # Build maps from contextRef to lists of titles and symbols
    title_map = {el.attrib.get('contextRef'): el.text.strip() if el.text else '' for el in titles}
    symbol_map = {el.attrib.get('contextRef'): el.text.strip().lower() if el.text else '' for el in symbols}

    for context_ref, title in title_map.items():
        symbol = symbol_map.get(context_ref, '')
        if symbol == ticker_lower:
            # Determine member_key based on title content
            if 'class' in title.lower():
                if ' B ' in title:
                    return 'CommonClassBMember'
                elif ' A ' in title:
                    return 'CommonClassAMember'
            elif 'series' in title.lower():
                if ' C ' in title:
                    return 'CommonClassCMember'
            elif 'non-voting' in title.lower():
                return 'NonvotingCommonStockMember'

    return None


def get_member_from_context(root, context_ref):
    context_elem = root.find(f".//{{*}}context[@id='{context_ref}']")
    if context_elem is None:
        return None, None  # No context found

    # Extract date (instant or endDate)
    date_elem = context_elem.find(".//{*}instant")
    if date_elem is None or not date_elem.text:
        date_elem = context_elem.find(".//{*}endDate")
    if date_elem is None or not date_elem.text:
        return None, None

    try:
        dt = datetime.strptime(date_elem.text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None, None

    # Extract member from segment/explicitMember if present
    segment_elem = context_elem.find(".//{*}segment")
    explicit_member_elem = None
    if segment_elem is not None:
        explicit_member_elem = segment_elem.find(".//{*}explicitMember")

    if explicit_member_elem is not None and explicit_member_elem.text:
        member_text = explicit_member_elem.text
        member_name = member_text.split(':')[-1]  # Remove prefix
    else:
        member_name = None  # No member

    return member_name, dt


def priority_index(candidate):
    return PRIORITY_MAP.get(
        (candidate['tag_name'], candidate['member_key']),
        100  # default low priority for unknown combos
    )


def explicit_priority(candidate, explicit_member_key):
    # Exact match gets highest priority 0, else fallback to PRIORITY_MAP
    if candidate['member_key'] == explicit_member_key:
        return 0
    return priority_index(candidate) + 10  # lower priority for others


def extract_shs_outstanding_and_date(xml_content, ticker):
    root = ET.fromstring(xml_content)

    report_date = get_document_period_end_date(root)
    if report_date is None:
        return None, None

    explicit_member_key = get_member_key_for_ticker(xml_content, ticker)

    candidate_tags = []
    for tag_name in ['EntityCommonStockSharesOutstanding', 'CommonStockSharesOutstanding']:
        for sh_tag in root.findall(f".//{{*}}{tag_name}"):
            context_ref = sh_tag.attrib.get('contextRef')
            if not context_ref:
                continue

            member_name, context_date = get_member_from_context(root, context_ref)
            if member_name is None:
                # Treat no member as None
                member_key = None
            else:
                if explicit_member_key:
                    if explicit_member_key not in member_name:
                        continue  # skip non-matching members
                    member_key = member_name
                else:
                    # No explicit member key, use existing logic
                    if member_name in EXCLUDED_MEMBERS:
                        continue
                    if member_name in STANDARD_MEMBERS:
                        member_key = member_name
                    else:
                        member_key = 'ANY_CUSTOM_MEMBER'

            # Check if context date matches report date interval
            start_interval, end_interval = get_report_date_interval(report_date)
            if context_date is None or not (start_interval <= context_date <= end_interval):
                continue

            candidate_tags.append({
                'sh_tag': sh_tag,
                'member_key': member_key,
                'tag_name': tag_name
            })

    if explicit_member_key:
        candidate_tags.sort(key=lambda c: explicit_priority(c, explicit_member_key))
    else:
        candidate_tags.sort(key=priority_index)

    # Extract shares from highest priority candidate with a valid value
    for candidate in candidate_tags:
        shares = extract_shares(candidate['sh_tag'])
        if shares is not None:
            return shares, report_date

    return None, report_date


def to_xml_filename(primary_doc_filename):
    # Replace ".htm" at the end with "_htm.xml"
    return re.sub(r'\.htm$', r'_htm.xml', primary_doc_filename, flags=re.IGNORECASE)


def find_filing_URL_for_cik_year_quarter(cik, year : str, quarter : str, form_types=('10-Q', '10-K')):
    cik_str = str(int(cik)).zfill(10)
    cik_clean = str(int(cik))  # Remove leading zeros
    filename = f"CIK{cik_str}.json"

    filepath = os.path.join(SUBMISSIONS_STOCKS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Submission file not found for CIK {cik} at {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    report_dates = filings.get("reportDate", [])
    accession_numbers = filings.get("accessionNumber", [])
    primary_documents = filings.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        if i >= len(report_dates):
            continue  # skip malformed entries

        filing_year, filing_quarter = get_year_and_quarter(report_dates[i])
        if filing_year == year and filing_quarter == quarter:
            accession_number_no_dashes = accession_numbers[i].replace('-', '')
            filename_xml = to_xml_filename(primary_documents[i])
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_clean}/{accession_number_no_dashes}/{filename_xml}"
            )
            return filing_url

    return None  # No matching filing found


def get_latest_10qk_urls(cik, lookback=3, form_type="10-Q", url=None):
    if url:
        resp = requests.get(url, headers=HEADERS)
    else:
        resp = requests.get(f"https://data.sec.gov/submissions/CIK{str(int(cik)).zfill(10)}.json", headers=HEADERS)

    if resp.status_code != 200:
        raise Exception(f"Failed to fetch submissions for CIK {cik}")

    data = resp.json()

    if url and "-" in url:
        filings = data
    else:
        filings = data.get('filings', {}).get('recent', {})
    filing_types = filings.get('form', [])
    accession_numbers = filings.get('accessionNumber', [])
    cik_clean = str(int(cik))  # Remove leading zeros

    urls = []
    count = 0
    for i, ftype in enumerate(filing_types):
        if ftype == form_type:
            accession_number = accession_numbers[i]
            accession_number_no_dashes = accession_number.replace('-', '')
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_clean}/{accession_number_no_dashes}/{accession_number}-index.html"
            )
            urls.append(filing_url)
            count += 1
            if count >= lookback:
                break

    if not urls:
        print(f"{cik} {cik_to_ticker.get(cik)} {form_type} not found!")
        return []

    return urls


def find_htm_xml_link(filing_index_url):
    response = limited_get(filing_index_url)
    if response.status_code != 200:
        print(f"Failed to fetch filing index page!: {filing_index_url}")
        return None

    html = etree.HTML(response.content)

    # Find all <a> hrefs ending with '_htm.xml'
    links = html.xpath('//a[contains(@href, "_htm.xml")]/@href')

    if not links:
        print(f"No '_htm.xml' file !: {filing_index_url}")
        return None

    # Usually there is only one primary XBRL instance document
    htm_xml_link = links[0]

    # Complete URL if relative
    if not htm_xml_link.startswith('http'):
        base_url = 'https://www.sec.gov'
        htm_xml_link = base_url + htm_xml_link

    return htm_xml_link


def shs_outstanding_for_ticker(ticker, df, lookback=4, year_quarter=None, xml_urls=None):
    cik = ticker_to_cik.get(ticker)
    lookback_years = lookback // 4

    if xml_urls is not None:
        urls_to_process = xml_urls
    else:
        # Fetch 10-K filings for last n years
        if lookback_years > 0:
            ten_k_urls = get_latest_10qk_urls(cik, lookback=lookback_years, form_type="10-K")
        else:
            ten_k_urls = []

        # Fetch 10-Q filings for last n quarters
        ten_q_urls = get_latest_10qk_urls(cik, lookback=lookback, form_type="10-Q")

        if not ten_q_urls and not ten_k_urls:
            return df

        # Combine filings urls
        all_filing_urls = ten_k_urls + ten_q_urls

        # Convert filing URLs to corresponding XML urls using find_htm_xml_link
        urls_to_process = []
        for filing_url in all_filing_urls:
            xml_url = find_htm_xml_link(filing_url)
            if xml_url:
                urls_to_process.append(xml_url)
            else:
                print(f"XML link not found for filing: {filing_url}")

    # Process all known xml urls
    for xml_url in urls_to_process:
        response = limited_get(xml_url)
        if response.status_code != 200:
            print(f"Failed to fetch XML: {xml_url}")
            continue

        xml_content = response.content
        shares, reported_date = extract_shs_outstanding_and_date(xml_content, ticker)
        if shares and reported_date:
            year, quarter = get_year_and_quarter(reported_date)
            if year_quarter is not None and f"{year}_{quarter}" != year_quarter:
                continue

            new_row = {'ticker': ticker, 'year': year, 'quarter': quarter, 'outstanding_shares': shares}

            # Check if row exists
            mask = (
                    (df['ticker'] == ticker) &
                    (df['year'] == year) &
                    (df['quarter'] == quarter)
            )

            # Replace existing row(s)
            if mask.any():
                df.loc[mask, ['outstanding_shares']] = shares
            else:
                # Append new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Imported {shares} shares outstanding for {ticker} and {reported_date}")
        else:
            print(f"Shares outstanding not found for {ticker} {reported_date} from link: {xml_url}")

    return df


def get_tickers_without_shares_outstanding(quarter: str = "Q1", year=2025):
    df = pd.read_csv(STOCKS_SHS_Q_END_PRICES_FILE)

    # Get all unique tickers in the file
    all_tickers = set(df['ticker'].unique())

    # Filter rows for the specified year and quarter
    df_filtered = df[(df['year'] == int(year)) & (df['quarter'] == quarter)]

    # Get tickers that have data for the specified year and quarter
    tickers_with_data = set(df_filtered['ticker'].unique())

    # Find tickers without data for that period
    tickers_without_data = all_tickers - tickers_with_data

    return tickers_without_data


def add_quarter_end_price_to_sh_outstanding_file(year_quarter_list=None):
    # Load existing shares outstanding file, including existing quarter_end_price column
    df = pd.read_csv(STOCKS_SHS_Q_END_PRICES_FILE,
                     dtype={'ticker': str, 'year': str, 'quarter': str, 'outstanding_shares': 'Int64'})

    # If no year_quarter_list passed, get all unique (year, quarter) pairs except [2025, 'Q3']
    if not year_quarter_list:
        year_quarter_list = [
            [year, quarter]
            for year, quarter in df[['year', 'quarter']].drop_duplicates().values.tolist()
            if not (year == '2025' and quarter == 'Q3')
        ]

    # Filter df to rows for the requested year_quarter_list to update prices only here
    mask = df.apply(lambda row: [row['year'], row['quarter']] in year_quarter_list, axis=1)

    # Get distinct tickers in df
    outstanding_shares_tickers = set(df['ticker'].unique())

    # Get price DataFrame for all requested quarters/tickers
    price_df = get_prices_for_all_quarters(BASE_DIR_FINAL, year_quarter_list, outstanding_shares_tickers)

    # For rows to update, merge price_df to get new quarter_end_price
    df_update = df[mask].drop(columns=['quarter_end_price'], errors='ignore').merge(
        price_df, on=['ticker', 'year', 'quarter'], how='left')

    # Replace updated rows in original df with df_update (which now has new prices)
    df.loc[mask, 'quarter_end_price'] = df_update['quarter_end_price'].values

    # Save updated DataFrame back to the CSV file, overwriting it
    df.to_csv(STOCKS_SHS_Q_END_PRICES_FILE, index=False)

    print(f"Updated CSV saved to {STOCKS_SHS_Q_END_PRICES_FILE}.")


def get_latest_10q_reports(submission_json_path, latest_n=1, form_type="10-Q", expected_year=None,
                           expected_quarter=None):
    """
    Reads the submission.json file, finds the latest n 10-Q filings by acceptanceDateTime,
    and returns list of tuples: (acceptance_datetime, report_date, year, quarter)
    """
    if form_type:
        form_type_allowed = form_type
    else:
        form_type_allowed = "10-Q"
    with open(submission_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filings = data.get('filings', {}).get('recent', {})
    forms = filings.get('form', [])
    report_dates = filings.get('reportDate', [])
    acceptance_dates = filings.get('acceptanceDateTime', [])

    # Collect only 10-Q filings with their indexed data
    ten_q_filings = []
    for i, form in enumerate(forms):
        if form == form_type_allowed:
            acceptance_dt = datetime.strptime(acceptance_dates[i], "%Y-%m-%dT%H:%M:%S.%fZ")
            report_date = report_dates[i]
            year, quarter = get_year_and_quarter(report_date)
            if expected_year is not None and expected_quarter is not None:
                if year == expected_year and quarter == expected_quarter:
                    ten_q_filings.append({
                        'acceptance_datetime': acceptance_dt,
                        'report_date': report_date,
                        'year': year,
                        'quarter': quarter
                    })

    # Sort filings by acceptance_datetime descending, pick latest n
    ten_q_filings.sort(key=lambda x: x['acceptance_datetime'], reverse=True)
    latest_filings = ten_q_filings[:latest_n]

    # Return in preferred format:
    return latest_filings


def update_multiple_years_quarters(year_quarter_list=None):
    if not year_quarter_list:
        year_quarter_list = [['2025', 'Q2']]

    for year, quarter in year_quarter_list:
        update_year_quarter_stocks_shs_and_q_end_price(year=year, quarter=quarter)


def update_year_quarter_stocks_shs_and_q_end_price(year=None, quarter=None):
    if not year:
        year = '2025'
    if not quarter:
        quarter = 'Q2'

    df = pd.read_csv(STOCKS_SHS_Q_END_PRICES_FILE)
    for cik, ticker in cik_to_ticker.items():
        filename = f'CIK{str(int(cik)).zfill(10)}.json'
        path = os.path.join(SUBMISSIONS_STOCKS_DIR, filename)
        result = get_latest_10q_reports(path, expected_year=year, expected_quarter=quarter)
        if result:
            prev_year, prev_quarter = get_prev_quarter(year, int(quarter.lstrip("Q")))
            prev_quarter = f"Q{prev_quarter}"

            if not has_q_end_price(ticker, year, quarter) and has_q_end_price(ticker, prev_year, prev_quarter):
                print(f'importing {year}, {quarter} for {ticker} shares outstanding')
                df = shs_outstanding_for_ticker(ticker, df=df, lookback=1)


    df['outstanding_shares'] = pd.to_numeric(df['outstanding_shares'], errors='coerce').astype('Int64')
    df.to_csv(STOCKS_SHS_Q_END_PRICES_FILE, index=False)

    add_quarter_end_price_to_sh_outstanding_file(year_quarter_list=[ [year, quarter]])


def main():
    df = pd.read_csv(STOCKS_SHS_Q_END_PRICES_FILE)
    tickers = ['STLA', 'AAPL', 'TSLA', 'TTD']
    tickers = [ 'BAC', 'JPM']
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if cik is not None:
            url = f"https://data.sec.gov/submissions/CIK{str(int(cik)).zfill(10)}-submissions-001.json"
            print(get_latest_10qk_urls(ticker_to_cik.get(ticker), lookback=6, form_type="10-Q", url=url))
            # df = shs_outstanding_for_ticker(ticker, df=df, lookback=6)

    not_found = ['AREC','ATIIU','AVR','AWRE','BBNX','BZAI','CBOE','CMBT','CRC','DBD','DMAA','DXYZ','FA','GCI','GNK','JACS','KRMN','LAUR','MLAC','MSBI','MTSI','MTSR','PACS','PLMK','SD','SNDK','SPHR','STLA','SVV','TAK','TIC','TLN','TRP','VIST','VREX', 'AMPY']
    not_found = ['TAK']


if __name__ == "__main__":
    main()
