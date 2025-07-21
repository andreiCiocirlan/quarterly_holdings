from datetime import datetime


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
