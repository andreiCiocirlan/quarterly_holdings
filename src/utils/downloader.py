import json
import os
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from cfg.cfg_requests import limited_get
from utils.constants import HEADERS
from utils.date_util import get_year_and_quarter
from utils.filings_util import delete_stale_13f_raw, delete_final_13f_by_accession
from utils.mappings import CIK_TO_PARSED_13F_DIR, SUBMISSIONS_FILERS_DIR
from utils.parser import parse_holdings


def get_amendment_type(primary_doc_url):
    # Download and parse XML to find <amendmentType>
    resp = limited_get(primary_doc_url)
    tree = ElementTree.fromstring(resp.content)
    # Try to find amendmentType in the XML
    amendment_type = None
    for elem in tree.iter():
        if elem.tag.lower().endswith('amendmenttype'):
            amendment_type = elem.text.strip().upper()
            break
    return amendment_type  # RESTATEMENT, NEW HOLDINGS, or None


def latest_filing_metadata(cik, latest_n_filings=1, skip_quarters_years=None, use_requests=True):
    """
    Fetch the latest_n 13F-HR accession numbers and their report dates for a given CIK.

    Args:
        cik (str or int): The CIK number.
        latest_n_filings (int): Number of latest filings to return.
        skip_quarters_years (optional): Quarters/years to skip (not implemented here).
        use_requests (bool): If True, fetch data from SEC website; if False, load from local file.

    Returns:
        List of tuples: [(accession_number, report_date, primary_doc), ...]
    """
    cik_str = str(int(cik)).zfill(10)

    if use_requests:
        # https://data.sec.gov/submissions/CIK0002012383.json
        url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
    else:
        filename = f"CIK{cik_str}.json"
        filepath = os.path.join(SUBMISSIONS_FILERS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Submission file not found for CIK {cik} at {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    primary_documents = filings.get("primaryDocument", [])
    report_dates = filings.get("reportDate", [])
    acceptance_datetimes = filings.get("acceptanceDateTime", [])

    # Collect all filings by quarter for 13F-HR and 13F-HR/A
    filings_by_quarter = {}
    for form, accession_nr, report_date, primary_doc, acceptance_dt in zip(
            forms, accession_numbers, report_dates, primary_documents, acceptance_datetimes
    ):
        if form not in ("13F-HR", "13F-HR/A") or not report_date:
            continue

        year, quarter = get_year_and_quarter(report_date)
        if year is None or quarter is None:
            continue

        if skip_quarters_years and f"{quarter} {year}" in skip_quarters_years:
            continue

        filings_by_quarter.setdefault(report_date, []).append({
            "form": form,
            "accession_number": accession_nr,
            "primary_doc": os.path.basename(primary_doc),
            "acceptance_datetime": acceptance_dt,
        })

    # Filter out report dates in the future
    current_year = datetime.now().year
    valid_quarters = [
        date for date in filings_by_quarter.keys()
        if date[:4].isdigit() and int(date[:4]) <= current_year
    ]

    # Sort quarters by report_date descending and pick latest_n_filings quarters
    latest_quarters = sorted(valid_quarters, reverse=True)[:latest_n_filings]

    results = []
    for quarter in latest_quarters:
        quarter_filings = filings_by_quarter[quarter]

        # Step 1. Identify amendment type for each filing in the quarter
        amendment_types_map = {}
        for f in quarter_filings:
            accession = f["accession_number"]
            if f["form"] == "13F-HR/A":
                accession_no_nodash = accession.replace("-", "")
                xml_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_nodash}/{f['primary_doc']}"
                amend_type = get_amendment_type(xml_url)
                amendment_types_map[accession] = amend_type
            else:
                amendment_types_map[accession] = None  # original filing

        # Step 2. Partition filings in this period by type
        restatements = [
            f for f in quarter_filings
            if amendment_types_map.get(f["accession_number"]) == "RESTATEMENT"
        ]
        originals_and_new_holdings = [
            f for f in quarter_filings
            if amendment_types_map.get(f["accession_number"]) in (None, "NEW HOLDINGS")
        ]

        # Step 3. Output logic as per spec
        if restatements:
            # Only keep the latest RESTATEMENT (by acceptance_datetime)
            most_recent = max(restatements, key=lambda f: f["acceptance_datetime"])
            results.append((
                most_recent["accession_number"],
                quarter.replace("-", ""),
                most_recent["primary_doc"],
                "RESTATEMENT"
            ))
        else:
            # Keep the original plus all NEW HOLDINGS amendments for this period
            for f in originals_and_new_holdings:
                results.append((
                    f["accession_number"],
                    quarter.replace("-", ""),
                    f["primary_doc"],
                    amendment_types_map.get(f["accession_number"])
                ))

    return results


def get_13f_xml(cik: str, accession_number: str, primary_doc="primary_doc.xml"):
    # remove leading zeros (if any)
    cik_clean = str(int(cik))
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_number}"

    resp = limited_get(base_url)

    # Step 2: Parse HTML to find XML links
    soup = BeautifulSoup(resp.text, 'html.parser')
    prefix = f"/Archives/edgar/data/{cik_clean}/{accession_number}/"

    xml_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.xml') and href.startswith(prefix) and not href.lower().endswith(primary_doc):
            xml_links.append(href)

    if not xml_links:
        print("No suitable XML files found.")
        return None

    # prioritize: 'infotable.xml'
    preferred_xml = None
    for href in xml_links:
        if 'infotable' in href.lower():
            preferred_xml = href
            break

    if not preferred_xml:
        preferred_xml = xml_links[0]  # fallback to first XML

    if preferred_xml:
        xml_url = f"https://www.sec.gov{preferred_xml}"
        xml_resp = limited_get(xml_url)

        return xml_resp.text
    else:
        print("No suitable XML files found.")
        return None


def download_filing_to_csv(cik: str, latest_n_filings=1, skip_quarters_years=None, use_requests=True):
    latest_metadata = latest_filing_metadata(cik, latest_n_filings, skip_quarters_years, use_requests)
    output_path = CIK_TO_PARSED_13F_DIR.get(cik)
    for accession_number, report_date, primary_doc, amendment_type in latest_metadata:
        accession_number_clean = accession_number.replace('-', '')
        csv_file_path = os.path.join(output_path, cik, f"{accession_number_clean.lstrip('0')}.csv")
        if os.path.exists(csv_file_path):
            continue  # Skip downloading this filing

        if amendment_type == 'RESTATEMENT':
            # delete 13F being replaced
            delete_stale_13f_raw(cik, report_date)
            delete_final_13f_by_accession(cik, accession_number_clean, report_date)

        # get xml data and parse
        xml_data = get_13f_xml(cik, accession_number_clean, primary_doc)
        parsed_data = parse_holdings(xml_data, accession_number_clean, report_date)

        # save to csv
        file_path = f"{output_path}/{cik}/{accession_number_clean.lstrip('0')}.csv"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if missing
        parsed_data.to_csv(file_path, index=False)
        print(f"Saved {report_date} 13F-HR file: {file_path}")
