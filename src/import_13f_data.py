from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import psycopg  # psycopg v3

from utils.file_util import extract_filername_year_quarter_accession
from utils.mappings import *

# Database connection parameters (use environment variables or defaults)
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5434')
DB_NAME = os.getenv('DB_NAME', 'filings_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')


def create_db_connection():
    # Replace with your actual connection parameters
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def create_holdings_table(conn):
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS holdings (
            id SERIAL PRIMARY KEY,
            accession_nr varchar(50) NOT NULL REFERENCES filings(accession_nr) ON DELETE CASCADE,
            rank INT,
            ticker varchar(50) NOT NULL,
            share_amount BIGINT,
            share_value NUMERIC(20, 2) DEFAULT 0.00,
            percentage NUMERIC(10,6) DEFAULT 0.000000,
            change_amount varchar(50),
            change_pct varchar(50),
            ownership_pct DOUBLE PRECISION DEFAULT 0.00,
            UNIQUE(accession_nr, ticker)
        );
    """
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()
    print("Table 'holdings' created.")


def create_filings_table(conn):
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS filings (
            accession_nr varchar(50) PRIMARY KEY,   -- accession number as primary key
            cik varchar(10) NOT NULL REFERENCES filers(cik),
            holdings_count INT DEFAULT 0,
            holdings_value NUMERIC(20, 2) DEFAULT 0.00,
            year INT NOT NULL,
            quarter INT NOT NULL,
            form_type VARCHAR(10)
        );
    """
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()
    print("Table 'filings' created.")


def create_filers_table(conn):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS filers (
        cik varchar(10) PRIMARY KEY,
        formatted_name TEXT NOT NULL,
        link TEXT
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()
    print("Table 'filers' created.")


def create_stocks_table(conn):
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS stocks (
        ticker varchar(50) NOT NULL,
        year INT NOT NULL,
        quarter INT NOT NULL,
        quarter_end_price DOUBLE PRECISION,          -- quarter_end_price
        inst_ownership DOUBLE PRECISION,          -- sum of ownership_pct from holdings
        shares_outstanding BIGINT,                 -- from your CSV file
        PRIMARY KEY (ticker, year, quarter)
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()
    print("Table 'stocks' created.")


def add_filings_holding_count_and_value(conn):
    update_query = """
        UPDATE filings f
        SET
            holdings_count = COALESCE(h.count, 0),
            holdings_value = COALESCE(h.total_value, 0)
        FROM (
            SELECT
                accession_nr,
                COUNT(*) AS count,
                SUM(share_value) AS total_value
            FROM holdings
            GROUP BY accession_nr
        ) h
        WHERE f.accession_nr = h.accession_nr;
    """
    with conn.cursor() as cur:
        cur.execute(update_query)
        print(f'Added holdings_count, holdings_value in filings table')
    conn.commit()


def insert_filers(df, conn):
    insert_query = """
        INSERT INTO filers (formatted_name, cik, link)
        VALUES (%s, %s, %s)
        ON CONFLICT (cik) DO UPDATE SET
            formatted_name = EXCLUDED.formatted_name,
            link = EXCLUDED.link;
    """
    records = [
        (
            row['formatted_name'],
            str(row['cik']),
            row['link']
        )
        for _, row in df.iterrows()
    ]
    with conn.cursor() as cur:
        cur.executemany(insert_query, records)
        print(f'INSERTED {len(records)} filers')
    conn.commit()


def load_shs_and_quarter_end_price(csv_file, tickers=None):
    df = pd.read_csv(csv_file, dtype=str)
    df['quarter'] = df['quarter'].str.extract(r'Q([1-4])').astype(int)
    df['year'] = df['year'].astype(int)
    df['shares_outstanding'] = pd.to_numeric(df['outstanding_shares'], errors='coerce').astype('Int64')
    df['quarter_end_price'] = pd.to_numeric(df['quarter_end_price'], errors='coerce')
    if tickers:
        if isinstance(tickers, str):
            # single ticker string
            df = df[df['ticker'] == tickers]
        else:
            # list or set of tickers
            df = df[df['ticker'].isin(tickers)]

    return df[['ticker', 'year', 'quarter', 'shares_outstanding', 'quarter_end_price']]


def load_inst_ownership(conn, tickers=None):
    query = """
    SELECT
        h.ticker,
        f.year,
        f.quarter,
        SUM(COALESCE(h.ownership_pct, 0)) AS inst_ownership
    FROM holdings h
    JOIN filings f ON h.accession_nr = f.accession_nr
    """
    params = []
    if tickers:
        if isinstance(tickers, str):
            query += " WHERE h.ticker = %s"
            params.append(tickers)
        else:
            placeholders = ','.join(['%s'] * len(tickers))
            query += f" WHERE h.ticker IN ({placeholders})"
            params.extend(tickers)

    query += " GROUP BY h.ticker, f.year, f.quarter"

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=['ticker', 'year', 'quarter', 'inst_ownership'])


def merge_stock_data(shares_df, inst_own_df):
    merged = pd.merge(shares_df, inst_own_df, on=['ticker', 'year', 'quarter'], how='outer')
    # Fill missing values if needed
    merged['shares_outstanding'] = merged['shares_outstanding'].fillna(0).astype('int64')
    merged['inst_ownership'] = pd.to_numeric(merged['inst_ownership'], errors='coerce').fillna(0.0)
    merged['quarter_end_price'] = pd.to_numeric(merged['quarter_end_price'], errors='coerce').fillna(0.0)
    return merged


def upsert_stocks(conn, merged_df):
    upsert_sql = """
    INSERT INTO stocks (ticker, year, quarter, shares_outstanding, quarter_end_price, inst_ownership)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, year, quarter)
    DO UPDATE SET
        shares_outstanding = EXCLUDED.shares_outstanding,
        quarter_end_price = EXCLUDED.quarter_end_price,
        inst_ownership = EXCLUDED.inst_ownership;
    """
    records = list(merged_df.itertuples(index=False, name=None))
    with conn.cursor() as cur:
        cur.executemany(upsert_sql, records)
    conn.commit()
    print(f"Upserted {len(records)} rows into stocks table.")


def update_stocks_table(conn, tickers=None):
    shares_df = load_shs_and_quarter_end_price(STOCKS_SHS_Q_END_PRICES_FILE, tickers)
    inst_own_df = load_inst_ownership(conn, tickers)
    merged_df = merge_stock_data(shares_df, inst_own_df)
    upsert_stocks(conn, merged_df)


def insert_stocks(conn):
    # Load CSV into pandas DataFrame
    df = pd.read_csv(STOCKS_SHS_Q_END_PRICES_FILE, dtype=str)

    # Extract quarter number from 'Q1', 'Q2', etc.
    df['quarter'] = df['quarter'].str.extract(r'Q([1-4])').astype(int)
    df['year'] = df['year'].astype(int)
    df['shares_outstanding'] = pd.to_numeric(df['outstanding_shares'], errors='coerce').astype('Int64')

    # Prepare data for insertion as list of tuples
    records = list(df[['ticker', 'year', 'quarter', 'shares_outstanding']].itertuples(index=False, name=None))

    upsert_sql = """
    INSERT INTO stocks (ticker, year, quarter, shares_outstanding)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (ticker, year, quarter)
    DO UPDATE SET shares_outstanding = EXCLUDED.shares_outstanding;
    """
    with conn.cursor() as cur:
        cur.executemany(upsert_sql, records)
    conn.commit()
    print(f"Inserted/updated {len(records)} rows into stocks table.")


def update_inst_ownership(conn):
    update_sql = """
    WITH inst_ownership_agg AS (
        SELECT
            h.ticker,
            q.year,
            q.quarter,
            SUM(h.ownership_pct) AS inst_ownership
        FROM holdings h
        JOIN filings f ON h.accession_nr = f.accession_nr
        JOIN quarters q ON f.quarter_id = q.id
        GROUP BY h.ticker, q.year, q.quarter
    )
    INSERT INTO stocks (ticker, year, quarter, inst_ownership)
    SELECT ticker, year, quarter, inst_ownership FROM inst_ownership_agg
    ON CONFLICT (ticker, year, quarter)
    DO UPDATE SET inst_ownership = EXCLUDED.inst_ownership;
    """
    with conn.cursor() as cur:
        cur.execute(update_sql)
    conn.commit()
    print("Updated inst_ownership in stocks table.")


def insert_filing(cur, cik, accession_nr, year, quarter, form_type='13F-HR'):
    cur.execute("""
        INSERT INTO filings (accession_nr, cik, year, quarter, form_type)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (accession_nr) DO NOTHING;
    """, (accession_nr, cik, year, quarter, form_type))


def insert_holdings(cur, accession_nr, df):
    data_to_insert = []
    for _, row in df.iterrows():
        data_to_insert.append((
            accession_nr,
            row.get('rank'),
            row.get('ticker'),
            row.get('share_amount'),
            row.get('share_value'),
            row.get('percentage'),
            row.get('change'),
            row.get('change_pct'),
            row.get('ownership_pct')
        ))

    if not data_to_insert:
        return

    columns = [
        "accession_nr", "rank", "ticker", "share_amount", "share_value",
        "percentage", "change_amount", "change_pct", "ownership_pct"
    ]

    # Build placeholders for multi-row insert
    values_placeholder = ", ".join(
        "(" + ", ".join(["%s"] * len(columns)) + ")"
        for _ in data_to_insert
    )

    query = f"""
        INSERT INTO holdings ({', '.join(columns)})
        VALUES {values_placeholder}
        ON CONFLICT (accession_nr, ticker) DO NOTHING;
    """

    # Flatten data_to_insert list of tuples into a single tuple
    flattened_values = [item for sublist in data_to_insert for item in sublist]

    cur.execute(query, flattened_values)


def import_quarterly_file(db_conn, cik, accession_nr, year, quarter, csv_path):
    df = pd.read_csv(csv_path)

    try:
        with db_conn.cursor() as cur:
            insert_filing(cur, cik, accession_nr, year, quarter)
            insert_holdings(cur, accession_nr, df)
        db_conn.commit()
    except Exception as e:
        db_conn.rollback()  # Rollback the failed transaction
        raise  # Re-raise the exception so caller knows about it


def import_filings_holdings(conn, aum_buckets):
    for base_dir, cik_to_filer in aum_buckets:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: base directory {base_path} does not exist. Skipping.")
            continue

        csv_files = list(base_path.rglob('*.csv'))
        print(f"Found {len(csv_files)} CSV files in {base_path}")

        for csv_file in csv_files:
            filename = csv_file.name  # with extension
            try:
                filername, year, quarter, accession_nr = extract_filername_year_quarter_accession(filename)
            except ValueError:
                print(f"Warning: filename format unexpected, skipping file: {filename}")
                continue

            matched_cik = FILERNAME_TO_CIK.get(filername)
            if matched_cik is None:
                print(f"Warning: No matching CIK found for file: {csv_file.name} (name: {filername})")
                continue

            print(f"Processing CIK {matched_cik} file: {csv_file}")
            try:
                import_quarterly_file(conn, matched_cik, accession_nr, year, quarter, csv_file)
            except Exception as e:
                print(f"Error importing {csv_file}: {e}")
                conn.rollback()  # Rollback on error to reset transaction state


def update_ownership_pct_holdings_and_stocks(conn, tickers=None, shs_outstanding_file=STOCKS_SHS_Q_END_PRICES_FILE):
    """
    Recompute and update ownership_pct in holdings based on share_amount / shares_outstanding,
    then update inst_ownership in stocks as sum of updated ownership_pct.
    If ticker is provided, only update data for that ticker.
    """

    df_shares = pd.read_csv(shs_outstanding_file, dtype=str)
    df_shares['quarter'] = df_shares['quarter'].str.extract(r'Q([1-4])').astype(int)
    df_shares['year'] = df_shares['year'].astype(int)
    df_shares['shares_outstanding'] = pd.to_numeric(df_shares['outstanding_shares'], errors='coerce').astype('Int64')

    # If tickers list is provided, filter by tickers
    if tickers:
        tickers_set = set(tickers)
        df_shares = df_shares[df_shares['ticker'].isin(tickers_set)]

    # Create a dictionary for quick lookup: {(ticker, year, quarter): shares_outstanding}
    shares_dict = {
        (row.ticker, row.year, row.quarter): row.shares_outstanding
        for row in df_shares.itertuples(index=False)
    }

    with conn.cursor() as cur:
        sql = """
            SELECT h.id, h.ticker, f.year, f.quarter, h.share_amount
            FROM holdings h
            JOIN filings f ON h.accession_nr = f.accession_nr
        """
        params = []
        if tickers:
            placeholders = ','.join(['%s'] * len(tickers))
            sql += f" WHERE h.ticker IN ({placeholders})"
            params.extend(tickers)

        cur.execute(sql, params)
        holdings_data = cur.fetchall()

        holdings_updates = []
        for hid, tkr, year, quarter, share_amount in holdings_data:
            key = (tkr, year, quarter)
            total_shares = shares_dict.get(key)
            if total_shares is None or total_shares == 0:
                print(f"Warning: Missing or zero shares outstanding for {key}")
                ownership_pct = 0.0
            else:
                ownership_pct = (share_amount / total_shares) * 100
            holdings_updates.append((ownership_pct, hid))

        # Batch update holdings ownership_pct
        update_holdings_sql = "UPDATE holdings SET ownership_pct = %s WHERE id = %s"
        cur.executemany(update_holdings_sql, holdings_updates)

    conn.commit()  # Commit holdings updates before next step

    # Step 3: Update stocks table using your existing method
    update_stocks_table(conn, tickers)


def process_single_csv(csv_file):
    conn = create_db_connection()
    filename = csv_file.name
    try:
        filername, year, quarter, accession_nr = extract_filername_year_quarter_accession(filename)
    except ValueError:
        print(f"Skipping file with unexpected format: {filename}")
        return

    matched_cik = FILERNAME_TO_CIK.get(filername)
    if matched_cik is None:
        print(f"No matching CIK for file: {filename}")
        return

    try:
        import_quarterly_file(conn, matched_cik, accession_nr, year, quarter, csv_file)
        conn.commit()
    except Exception as e:
        print(f"Error importing {filename}: {e}")
        conn.rollback()
    finally:
        conn.close()


def gather_all_csv_files(base_dir):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: base directory {base_path} does not exist.")
        return []

    return list(base_path.rglob('*.csv'))

def import_all_files_parallel(base_dir=BASE_DIR_FINAL, max_workers=8):
    all_csv_files = gather_all_csv_files(base_dir)
    print(f"Total CSV files to process: {len(all_csv_files)}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_csv, csv_file) for csv_file in all_csv_files]

        for future in as_completed(futures):
            # Optionally handle results or exceptions here
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker: {e}")


def drop_all_tables(conn):
    with conn.cursor() as cur:
        tables = ['holdings', 'filings', 'stocks', 'filers']  # drop in this order to respect FK dependencies if any
        for table in tables:
            try:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                print(f"Dropped table '{table}' if it existed.")
            except Exception as e:
                print(f"Error dropping table {table}: {e}")
    conn.commit()


def main():
    # Read CSV
    df = pd.read_csv(ALL_FILERS_CSV)

    # Connect to DB
    conn = create_db_connection()
    conn.autocommit = False  # Explicit transaction control

    try:
            drop_all_tables(conn)

            create_filers_table(conn)
            conn.commit()
            insert_filers(df, conn)
            conn.commit()

            create_filings_table(conn)
            conn.commit()

            create_holdings_table(conn)
            conn.commit()

            create_stocks_table(conn)
            conn.commit()

            import_all_files_parallel(base_dir=BASE_DIR_FINAL, max_workers=12)

            add_filings_holding_count_and_value(conn)
            conn.commit()

            update_stocks_table(conn)
            conn.commit()

        # fix ownership pct for ticker
        # print("Starting update_ownership_pct_holdings_and_stocks...")
        # start_time = time.time()
        # update_ownership_pct_holdings_and_stocks(conn, tickers=['AAPL'])
        # conn.commit()
        # print(f"Update completed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        conn.rollback()
        print(f"Error during update: {e}")
    finally:
        conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
