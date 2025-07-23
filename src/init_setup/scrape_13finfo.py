import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.mappings import ALL_FILERS_CSV

BASE_URL = "https://13f.info"
LETTERS = '0abcdefghijklmnopqrstuvwxyz'


def scrape_managers_table(letter):
    """
    Scrapes the managers table for a given letter from 13f.info.

    Returns:
        pd.DataFrame with columns:
        ['name', 'location', 'last_reported', 'holdings_count', 'holdings_value', 'link']
    """
    url = f"{BASE_URL}/managers/{letter}"
    print(f"Fetching managers starting with '{letter}'...")
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table')
    if not table:
        print(f"No table found on page for letter '{letter}'.")
        return pd.DataFrame()  # empty DataFrame

    headers = ['name', 'location', 'last_reported', 'holdings_count', 'holdings_value', 'link']

    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cols = []
        link = ''
        tds = tr.find_all('td')
        for idx, td in enumerate(tds):
            if idx == 0:
                a_tag = td.find('a')
                if a_tag and a_tag.has_attr('href'):
                    link = a_tag['href']
                    cols.append(a_tag.text.strip())
                else:
                    cols.append(td.text.strip())
            else:
                cols.append(td.text.strip())
        cols.append(link)
        rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)
    return df


def scrape_managers_by_letter(letter):
    df = scrape_managers_table(letter)
    if df.empty:
        return []

    managers = []
    for _, row in df.iterrows():
        cik = extract_cik_from_link(row['link'])
        managers.append({
            'cik': cik,
            'last_reported': row['last_reported'],
            'holdings_value': row['holdings_value']
        })
    return managers


def update_filings(csv_path=ALL_FILERS_CSV):
    df = pd.read_csv(csv_path, dtype={'cik': str})
    df['cik'] = df['cik'].str.lstrip('0')

    updated = False

    for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
        print(f"Processing letter {letter}...")
        scraped_managers = scrape_managers_by_letter(letter)
        for manager in scraped_managers:
            cik = manager['cik']
            if cik not in df['cik'].values:
                # Ignore new CIKs not in the CSV
                continue

            idx = df.index[df['cik'] == cik][0]
            if (df.at[idx, 'last_reported'] != manager['last_reported'] or
                    df.at[idx, 'holdings_value'] != manager['holdings_value']):
                print(f"Updating CIK {cik}: last_reported {df.at[idx, 'last_reported']} -> {manager['last_reported']}, "
                      f"holdings_value {df.at[idx, 'holdings_value']} -> {manager['holdings_value']}")
                df.at[idx, 'last_reported'] = manager['last_reported']
                df.at[idx, 'holdings_value'] = manager['holdings_value']
                updated = True

    if updated:
        df.to_csv(csv_path, index=False)
        print("CSV updated with new data.")
    else:
        print("No updates detected.")


def extract_cik_from_link(link):
    if isinstance(link, str) and link.startswith('/manager/'):
        part = link[len('/manager/'):]
        cik_with_zeros = part.split('-')[0]
        cik = cik_with_zeros.lstrip('0')
        return cik if cik else '0'
    return ''


def format_company_name(name: str) -> str:
    # Split by whitespace, capitalize each word, join with underscore
    words = name.strip().split()
    capitalized_words = [word.capitalize() for word in words]
    return "_".join(capitalized_words)


def sanitize_formatted_name(name: str) -> str:
    sanitized = re.sub(r'[^\w\s]', '', str(name))
    return sanitized


def save_managers_table_with_links(letter, output_csv=ALL_FILERS_CSV):
    df = scrape_managers_table(letter)
    if df.empty:
        print(f"No data to save for letter '{letter}'.")
        return

    # Filter for minimum AUM of $1B or more
    df_filtered = df[df['holdings_value'].str.endswith(('B', 'T'), na=False)]

    # Blackrock 2x ciks, hence Q2 2024 etc. valid quarter
    valid_quarters = ["Q2 2024", "Q3 2024", "Q4 2024", "Q1 2025", "Q2 2025", "Q3 2025"]

    # Filter for valid quarters (e.g. Q4 2024, Q1 2025)
    df_filtered = df_filtered[df_filtered['last_reported'].isin(valid_quarters)]

    # Add formatted_name and cik columns
    df_filtered['name'] = df_filtered['name'].apply(sanitize_formatted_name)
    df_filtered['holdings_count'] = df_filtered['holdings_count'].apply(sanitize_formatted_name)
    df_filtered['formatted_name'] = df_filtered['name'].apply(format_company_name)
    df_filtered['formatted_name'] = df_filtered['formatted_name'].str.replace(' ', '_')
    df_filtered['cik'] = df_filtered['link'].apply(extract_cik_from_link)

    # Drop 'location' and 'name' columns to match desired structure
    df_filtered = df_filtered.drop(columns=['location', 'name'])

    # Reorder columns to a consistent order
    desired_cols = ['formatted_name', 'cik', 'last_reported', 'holdings_count', 'holdings_value', 'link']
    df_filtered = df_filtered[desired_cols]

    # Save or append CSV
    if os.path.exists(output_csv):
        df_filtered.to_csv(output_csv, mode='a', header=False, index=False)
        print(f"Appended {len(df_filtered)} rows to {output_csv}")
    else:
        df_filtered.to_csv(output_csv, mode='w', header=True, index=False)
        print(f"Created {output_csv} with {len(df_filtered)} rows")


def main():
    for letter in LETTERS:
        save_managers_table_with_links(letter, output_csv=ALL_FILERS_CSV)


    # update_filings()


if __name__ == "__main__":
    main()
