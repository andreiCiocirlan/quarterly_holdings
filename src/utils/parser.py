import re
import xml.etree.ElementTree as ET

import pandas as pd


def clean_text(s):
    if pd.isna(s):
        return s
    # Replace multiple whitespace (including newlines, tabs) with a single space
    s = re.sub(r'\s+', ' ', s)

    # Strip leading/trailing whitespace
    s = s.strip()

    # Remove unwanted characters except word chars, spaces, comma, dot, hyphen, and double quotes
    s = re.sub(r'[^\w\s,.\-"]', '', s)
    return s


def get_text(element, path, namespaces):
    found = element.find(path, namespaces)
    return found.text if found is not None else pd.NA


def parse_holdings(xml_data: str, accession_number_clean: str, conformed_date: str) -> pd.DataFrame:
    try:
        namespaces = {
            'ns1': 'http://www.sec.gov/edgar/document/thirteenf/informationtable',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }

        root = ET.fromstring(xml_data)
        parsed_data = []

        for info_table in root.findall('.//ns1:infoTable', namespaces):
            data = {
                'ACCESSION_NUMBER': accession_number_clean,
                'CONFORMED_DATE': conformed_date,
                'NAME_OF_ISSUER': get_text(info_table, 'ns1:nameOfIssuer', namespaces),
                'TITLE_OF_CLASS': get_text(info_table, 'ns1:titleOfClass', namespaces),
                'CUSIP': get_text(info_table, 'ns1:cusip', namespaces),
                'SHARE_VALUE': get_text(info_table, 'ns1:value', namespaces),
                'SHARE_AMOUNT': get_text(info_table, 'ns1:shrsOrPrnAmt/ns1:sshPrnamt', namespaces),
                'PUT_CALL': get_text(info_table, 'ns1:putCall', namespaces),
                'SH_PRN': get_text(info_table, 'ns1:shrsOrPrnAmt/ns1:sshPrnamtType', namespaces),
                'DISCRETION': get_text(info_table, 'ns1:investmentDiscretion', namespaces),
                'SOLE_VOTING_AUTHORITY': get_text(info_table, 'ns1:votingAuthority/ns1:Sole', namespaces),
                'SHARED_VOTING_AUTHORITY': get_text(info_table, 'ns1:votingAuthority/ns1:Shared', namespaces),
                'NONE_VOTING_AUTHORITY': get_text(info_table, 'ns1:votingAuthority/ns1:None', namespaces),
            }
            parsed_data.append(data)

        # Convert to DataFrame
        df = pd.DataFrame(parsed_data)

        # Convert numeric columns
        for col in ['SHARE_VALUE', 'SHARE_AMOUNT', 'SOLE_VOTING_AUTHORITY', 'SHARED_VOTING_AUTHORITY',
                    'NONE_VOTING_AUTHORITY']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce').astype(pd.Int64Dtype())

        # sanitize string cols (remove special characters, comma etc.)
        for col in ['NAME_OF_ISSUER', 'TITLE_OF_CLASS']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(clean_text)

        return df

    except Exception as e:
        # Return empty DataFrame on error
        return pd.DataFrame()
