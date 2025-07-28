import re


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
