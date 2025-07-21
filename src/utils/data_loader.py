import re
import pandas as pd
from utils.constants import VALID_QUARTERS

def _parse_holdings_value(value_str):
    if not isinstance(value_str, str):
        return None
    value_str = value_str.strip().replace(',', '')
    pattern = r'^\$?([\d\.]+)\s*([kKmMbBtT]?)$'
    match = re.match(pattern, value_str)
    if not match:
        return None
    number, suffix = match.groups()
    try:
        number = float(number)
    except ValueError:
        return None
    suffix = suffix.lower()
    if suffix == 'k':
        multiplier = 1e-12
    elif suffix == 'm':
        multiplier = 1e-9
    elif suffix == 'b':
        multiplier = 1
    elif suffix == 't':
        multiplier = 1e3
    else:
        multiplier = 1e-9
    return number * multiplier

import os

def load_cik_to_filer_and_aum(csv_file):
    if not os.path.exists(csv_file):
        return {}

    df = pd.read_csv(csv_file, dtype=str, usecols=['cik', 'formatted_name', 'holdings_value', 'last_reported'])
    df = df[df['last_reported'].isin(VALID_QUARTERS)]
    df['holdings_billion'] = df['holdings_value'].apply(_parse_holdings_value)
    df = df.dropna(subset=['holdings_billion'])
    return df.set_index('cik')[['formatted_name', 'holdings_billion']].apply(tuple, axis=1).to_dict()

def filter_cik_by_aum(cik_to_filer_and_aum, min_aum, max_aum):
    return {cik: info for cik, info in cik_to_filer_and_aum.items()
            if min_aum <= info[1] <= max_aum}