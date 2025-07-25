import math
import os
import re

import psycopg
from flask import Flask, jsonify, request, abort, render_template

app = Flask(__name__)

# Database connection parameters (use environment variables or defaults)
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'filings_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')


def get_db_connection():
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn


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


@app.template_filter('currency')
def currency_filter(value):
    if value is None:
        return '-'
    return f"${value:,.2f}"


# Helper function to build filer URL slug
def filer_url(cik, formatted_name):
    name_slug = formatted_name.replace('_', '-').lower()
    name_slug = re.sub(r'[^a-z0-9\-]', '', name_slug)
    return f"/manager/{cik}-{name_slug}"


# Register as Jinja2 global
app.jinja_env.globals.update(filer_url=filer_url)
app.jinja_env.filters['format_large_number'] = format_large_number

@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        WITH latest_filings AS (
            SELECT
                accession_nr,
                cik,
                year,
                quarter,
                holdings_count,
                holdings_value,
                ROW_NUMBER() OVER (PARTITION BY cik ORDER BY year DESC, quarter DESC) AS rn
            FROM filings
        )
        SELECT
            f.cik,
            f.formatted_name,
            lf.holdings_count,
            lf.holdings_value,
            lf.accession_nr,
            lf.year,
            lf.quarter
        FROM filers f
        LEFT JOIN latest_filings lf ON f.cik = lf.cik AND lf.rn = 1
        ORDER BY lf.holdings_value DESC NULLS LAST
        LIMIT 200;
    ''')
    filers_raw = cur.fetchall()
    cur.close()
    conn.close()

    # Convert tuples to lists to allow modification
    filers = []
    for filer in filers_raw:
        filer_list = list(filer)
        filer_list[3] = format_large_number(filer_list[3])  # Format holdings_value (4th element, index 3)
        filers.append(tuple(filer_list))

    return render_template('index.html', filers=filers)


@app.route('/letter/<letter>')
def filers_by_letter(letter):
    # Sanitize input: uppercase and ensure single letter A-Z
    letter = letter.upper()
    if letter not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        abort(404)

    conn = get_db_connection()
    cur = conn.cursor()

    # Query filers whose name starts with the letter (case-insensitive)
    query = """
        WITH latest_filings AS (
            SELECT
                f.cik,
                fl.accession_nr,
                fl.year,
                fl.quarter,
                fl.holdings_count,
                fl.holdings_value,
                ROW_NUMBER() OVER (PARTITION BY f.cik ORDER BY fl.year DESC, fl.quarter DESC) AS rn
            FROM filers f
            LEFT JOIN filings fl ON fl.cik = f.cik
            WHERE f.formatted_name ILIKE %s
        )
        SELECT
            f.cik,
            f.formatted_name,
            f.link,
            lf.accession_nr,
            lf.year,
            lf.quarter,
            COALESCE(lf.holdings_count, 0) AS holdings_count,
            COALESCE(lf.holdings_value, 0) AS holdings_value
        FROM filers f
        LEFT JOIN latest_filings lf ON lf.cik = f.cik AND lf.rn = 1
        WHERE f.formatted_name ILIKE %s
        ORDER BY f.formatted_name ASC;
    """

    like_pattern = letter + '%'
    cur.execute(query, (like_pattern, like_pattern))
    filers = cur.fetchall()

    cur.close()
    conn.close()

    return render_template('filers_by_letter.html', filers=filers, letter=letter)


@app.route('/manager/<manager_slug>')
def manager(manager_slug):
    import re
    from flask import abort, render_template

    match = re.match(r'(\d+)-(.+)', manager_slug)
    if not match:
        abort(404)
    cik = match.group(1)

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch filer info
    cur.execute('SELECT cik, formatted_name FROM filers WHERE cik = %s', (cik,))
    filer = cur.fetchone()
    if not filer:
        abort(404)

    # Fetch filings
    cur.execute('''
        SELECT accession_nr, year, quarter, form_type, holdings_count, holdings_value
        FROM filings
        WHERE cik = %s
        ORDER BY year DESC, quarter DESC;
    ''', (cik,))
    filings = cur.fetchall()

    # For each filing, calculate top 20 holdings percentage
    filings_with_pct = []
    top_20_pct = 0
    for filing in filings:
        accession_nr = filing[0]

        # Fetch holdings for this filing
        cur.execute('''
            SELECT share_value
            FROM holdings
            WHERE accession_nr = %s
            ORDER BY share_value DESC;
        ''', (accession_nr,))
        holdings = cur.fetchall()

        if holdings:
            values = [h[0] for h in holdings if h[0] is not None]
            total_value = sum(values)
            top_20_value = sum(values[:20]) if len(values) >= 20 else sum(values)
            top_20_pct = (top_20_value / total_value) * 100 if total_value > 0 else 0

        # Format holdings_value (6th element, index 5)
        formatted_holdings_value = format_large_number(filing[5])

        # Append filing data + top_20_pct + formatted holdings_value
        # Replace holdings_value with formatted string
        filing_list = list(filing)
        filing_list[5] = formatted_holdings_value
        filings_with_pct.append(tuple(filing_list) + (top_20_pct,))

    if filings:
        most_recent_accession = filings[0][0]
        most_recent_year = filings[0][1]
        most_recent_quarter = filings[0][2]
    else:
        most_recent_accession = None
        most_recent_year = None
        most_recent_quarter = None

    # --- Get quarterly holdings values for this cik ---
    cur.execute('''
        SELECT year, quarter, holdings_value
        FROM filings
        WHERE cik = %s
        ORDER BY year, quarter
    ''', (cik,))

    quarterly_rows = cur.fetchall()
    quarterly_data = [
        {'year': r[0], 'quarter': r[1], 'value': r[2] or 0}
        for r in quarterly_rows
    ]

    cur.close()
    conn.close()

    labels = [f"Q{q['quarter']} {q['year']}" for q in quarterly_data]
    values = [q['value'] for q in quarterly_data]

    return render_template('manager.html',
                           filings=filings_with_pct,
                           cik=cik,
                           formatted_name=filer[1],
                           top_20_pct=top_20_pct,
                           chart_labels=labels,
                           cik_padded = str(cik).zfill(10),
                           chart_values=values,
                           most_recent_accession=most_recent_accession,
                           most_recent_year=most_recent_year,
                           most_recent_quarter=most_recent_quarter
                           )


@app.route('/holdings/<accession_nr>')
def holdings(accession_nr):
    conn = get_db_connection()
    cur = conn.cursor()

    # --- Get current filing info ---
    cur.execute('''
        SELECT cik, year, quarter, holdings_count, holdings_value
        FROM filings
        WHERE accession_nr = %s
    ''', (accession_nr,))
    filing_info = cur.fetchone()
    if not filing_info:
        abort(404)
    cik, current_year, current_quarter, holdings_count, holdings_value = filing_info

    # --- Get filer name ---
    cur.execute('''
        SELECT formatted_name
        FROM filers
        WHERE cik = %s
    ''', (cik,))
    filer_name_row = cur.fetchone()
    filer_name = filer_name_row[0] if filer_name_row else "Unknown Filer"

    # --- Get all filings for this cik (for Compare dropdown) ---
    cur.execute('''
        SELECT accession_nr, year, quarter
        FROM filings
        WHERE cik = %s
        ORDER BY year DESC, quarter DESC
    ''', (cik,))
    all_filings = cur.fetchall()

    # --- Get all holdings for current accession_nr (no pagination) ---
    cur.execute('''
        SELECT *
        FROM holdings
        WHERE accession_nr = %s
        ORDER BY share_value DESC NULLS LAST
    ''', (accession_nr,))

    columns = [desc[0] for desc in cur.description] if cur.description else []
    rows = cur.fetchall()
    holdings = [dict(zip(columns, row)) for row in rows] if columns else []

    cur.close()
    conn.close()

    # Render template with full holdings list
    return render_template(
        'holdings.html',
        holdings=holdings,
        all_filings=all_filings,
        accession_nr=accession_nr,
        current_year=current_year,
        current_quarter=current_quarter,
        filer_name=filer_name,
        holdings_value=holdings_value,
        cik=cik,
        holdings_count=holdings_count
    )


@app.route('/api/search_filers')
def search_filers():
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])  # return empty list for short queries

    # Replace spaces with underscores to match stored formatted_name
    if q.startswith('0') and q.lstrip('0').isdigit():
        search_term = q.lstrip('0') or '0'
    else:
        search_term = q.replace(' ', '_')

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT cik, formatted_name
        FROM filers
        WHERE formatted_name ILIKE %s or cik ILIKE %s
        ORDER BY formatted_name ASC
        LIMIT 10;
    """, (f'%{search_term}%',f'%{search_term}%'))
    results = cur.fetchall()
    cur.close()
    conn.close()

    # Convert to list of dicts for JSON
    filers = [{'cik': row[0], 'formatted_name': row[1]} for row in results]
    return jsonify(filers)


@app.template_filter()
def format_percent_or_na(value, decimals=2, times_100=False):
    try:
        fval = float(value)
        if math.isnan(fval):
            return "N/A"
        if times_100:
            fval = fval * 100
        return f"{fval:,.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"


@app.template_filter()
def format_number_or_na(value, decimals=2):
    try:
        fval = float(value)
        if math.isnan(fval):
            return "N/A"

        format_str = f"{{:,.{decimals}f}}".format(fval)
        return format_str
    except (ValueError, TypeError):
        return "N/A"


@app.route('/holdings/<accession_nr>/compare/<compare_accession_nr>')
def compare_holdings(accession_nr, compare_accession_nr):
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch filing info for both accession numbers
    cur.execute('SELECT cik, year, quarter FROM filings WHERE accession_nr = %s', (accession_nr,))
    filing1 = cur.fetchone()
    if not filing1:
        abort(404)
    cik1, year1, quarter1 = filing1

    cur.execute('SELECT cik, year, quarter FROM filings WHERE accession_nr = %s', (compare_accession_nr,))
    filing2 = cur.fetchone()
    if not filing2:
        abort(404)
    cik2, year2, quarter2 = filing2

    # Ensure both filings belong to the same filer (same cik)
    if cik1 != cik2:
        abort(400, description="Filings belong to different filers and cannot be compared.")

    # Fetch filer name for cik1
    cur.execute('SELECT formatted_name FROM filers WHERE cik = %s', (cik1,))
    filer_row = cur.fetchone()
    filer_name = filer_row[0] if filer_row else "Unknown Filer"

    newer_accession = accession_nr
    older_accession = compare_accession_nr
    newer_filing = filing1
    older_filing = filing2
    latest_accession_nr = newer_accession

    # Fetch holdings for older filing
    cur.execute('SELECT * FROM holdings WHERE accession_nr = %s', (older_accession,))
    columns = [desc[0] for desc in cur.description] if cur.description else []
    rows_older = cur.fetchall()
    holdings_older = [dict(zip(columns, row)) for row in rows_older] if columns else []

    # Fetch holdings for newer filing
    cur.execute('SELECT * FROM holdings WHERE accession_nr = %s', (newer_accession,))
    rows_newer = cur.fetchall()
    holdings_newer = [dict(zip(columns, row)) for row in rows_newer] if columns else []

    holdings_older_dict = {h['ticker']: h for h in holdings_older}
    holdings_newer_dict = {h['ticker']: h for h in holdings_newer}

    all_keys = set(holdings_older_dict.keys()) | set(holdings_newer_dict.keys())

    comparison = []

    for key in all_keys:
        old = holdings_older_dict.get(key)
        new = holdings_newer_dict.get(key)

        shares_old = old['share_amount'] if old else 0
        shares_new = new['share_amount'] if new else 0
        value_old = old['share_value'] if old else 0
        value_new = new['share_value'] if new else 0

        # Shares difference
        shares_diff = shares_new - shares_old

        # Shares change percentage with NEW/REMOVED logic
        if old is None and new is not None:
            shares_change_pct = "NEW"
        elif old is not None and new is None:
            shares_change_pct = "REMOVED"
        else:
            if shares_old == 0:
                shares_change_pct = None
            else:
                shares_change_pct = ((shares_new - shares_old) / shares_old) * 100

        # Value difference
        value_diff = value_new - value_old

        # Value change percentage with NEW/REMOVED logic
        if old is None and new is not None:
            value_change_pct = "NEW"
        elif old is not None and new is None:
            value_change_pct = "REMOVED"
        else:
            if value_old == 0:
                value_change_pct = None
            else:
                value_change_pct = ((value_new - value_old) / value_old) * 100

        comparison.append({
            'ticker': key,
            'shares_old': shares_old,
            'shares_new': shares_new,
            'shares_diff': shares_diff,
            'shares_change_pct': shares_change_pct,
            'value_old': value_old,
            'value_new': value_new,
            'value_diff': value_diff,
            'value_change_pct': value_change_pct,
        })

    # Sort by value_new desc
    comparison.sort(key=lambda x: x['value_new'] if x['value_new'] is not None else 0, reverse=True)

    cur = conn.cursor()
    cur.execute('SELECT accession_nr, year, quarter FROM filings WHERE cik = %s ORDER BY year DESC, quarter DESC', (cik1,))
    all_filings = cur.fetchall()
    cur.close()

    return render_template(
        'compare.html',
        comparison=comparison,
        filing1={'accession_nr': older_accession, 'year': older_filing[1], 'quarter': older_filing[2]},
        filing2={'accession_nr': newer_accession, 'year': newer_filing[1], 'quarter': newer_filing[2]},
        filer_name=filer_name,
        cik=cik1,
        all_filings=all_filings,
        latest_accession_nr=latest_accession_nr
    )


@app.route('/tickers/<string:ticker>/quarterly_holdings')
def ticker_quarterly_holdings(ticker):
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
        SELECT
          f.year,
          f.quarter,
          SUM(h.share_amount) AS total_shares,
          SUM(h.share_value) AS total_value,
          s.shares_outstanding,
          s.inst_ownership,
          s.quarter_end_price
        FROM holdings h
        JOIN filings f ON h.accession_nr = f.accession_nr
        LEFT JOIN stocks s ON s.ticker = h.ticker AND s.year = f.year AND s.quarter = f.quarter
        WHERE h.ticker = %s
        GROUP BY f.year, f.quarter, s.shares_outstanding, s.inst_ownership, s.quarter_end_price
        ORDER BY f.year DESC, f.quarter DESC;
    """
    cur.execute(query, (ticker.upper(),))
    holdings_data = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('ticker_holdings.html', holdings=holdings_data, ticker=ticker.upper())


@app.route('/<string:ticker>/<int:year>/<int:quarter>')
def ticker_quarter_detail(ticker, year, quarter):
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
        SELECT
          fi.formatted_name AS filer_name,
          h.share_amount,
          h.share_value,
          h.ownership_pct
        FROM holdings h
        JOIN filings f ON h.accession_nr = f.accession_nr
        JOIN filers fi ON f.cik = fi.cik
        WHERE
          h.ticker = %s
          AND f.year = %s
          AND f.quarter = %s
        ORDER BY h.ownership_pct DESC NULLS LAST, h.share_amount DESC NULLS LAST;
    """
    cur.execute(query, (ticker, year, quarter))
    holders = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('ticker_quarter_detail.html',
                           ticker=ticker.upper(),
                           year=year,
                           quarter=quarter,
                           holders=holders)


@app.route('/filer/<string:cik>/ticker/<string:ticker>/history')
def ticker_history(cik, ticker):
    ticker = ticker.upper()
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch filer name for the CIK
    cur.execute('SELECT formatted_name  FROM filers WHERE cik = %s', (cik,))
    filer_row = cur.fetchone()
    filer_name = filer_row[0] if filer_row else "Unknown Filer"

    # Query all filings of the filer that include this ticker, ordered by year, quarter descending
    query = """
        SELECT 
            f.year,
            f.quarter,
            h.share_amount,
            h.share_value,
            f.holdings_value,
            (h.share_value::numeric / f.holdings_value::numeric) * 100 AS pct_holdings
        FROM filings f
        JOIN holdings h ON f.accession_nr = h.accession_nr
        WHERE f.cik = %s AND UPPER(h.ticker) = %s
        ORDER BY f.year DESC, f.quarter DESC
    """
    cur.execute(query, (cik, ticker))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        abort(404, description=f"No holdings found for ticker '{ticker}' and filer CIK {cik}")

    # Prepare data for template
    history = [{
        "year": year,
        "quarter": quarter,
        "shares": share_amount,
        "value": share_value,
        "holdings_value": holdings_value,
        "pct_holdings": pct_holdings
    } for year, quarter, share_amount, share_value, holdings_value, pct_holdings in rows]

    return render_template('ticker_history.html',
                           ticker=ticker,
                           filer_name=filer_name,
                           cik=cik,
                           history=history)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
