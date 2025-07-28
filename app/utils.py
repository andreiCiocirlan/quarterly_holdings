import os
import re

import psycopg

# Database connection parameters (use environment variables or defaults)
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5434')
DB_NAME = os.getenv('DB_NAME', 'filings_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')


def get_db_connection():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        # Local fallback (e.g. your local dev machine)
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(DATABASE_URL)
    return psycopg.connect(DATABASE_URL)


# Helper function to build filer URL slug
def filer_url(cik, formatted_name):
    name_slug = formatted_name.replace('_', '-').lower()
    name_slug = re.sub(r'[^a-z0-9\-]', '', name_slug)
    return f"/manager/{cik}-{name_slug}"


def format_large_number(num):
    if num is None:
        return '-'
    try:
        num = float(num)
    except (ValueError, TypeError):
        return str(num)

    trillion = 1_000_000_000_000
    billion = 1_000_000_000
    million = 1_000_000

    if num >= trillion:
        return f"${num / trillion:.2f} T"
    elif num >= billion:
        return f"${num / billion:.2f} B"
    elif num >= million:
        return f"${num / million:.2f} M"
    else:
        return f"${num:,.2f}"
