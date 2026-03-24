from datetime import datetime

def is_valid_report_date(date_str):
    """Returns True if date_str parses successfully with get_year_and_quarter formats."""
    if not date_str or date_str in ['.', 'null', '', 'None']:  # Quick reject common invalids
        return False
    try:
        get_year_and_quarter(date_str)  # Reuse your existing validator
        return True
    except ValueError:
        return False

def get_year_and_quarter(date_str):
    date_formats = ["%Y-%m-%d", "%Y%m%d"]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD' or 'YYYYMMDD'.")

    month = dt.month
    year = dt.year

    if 1 <= month <= 3:
        quarter = 'Q1'
    elif 4 <= month <= 6:
        quarter = 'Q2'
    elif 7 <= month <= 9:
        quarter = 'Q3'
    elif 10 <= month <= 12:
        quarter = 'Q4'
    else:
        raise ValueError(f"Invalid month extracted from date: {month}")

    return str(year), quarter
