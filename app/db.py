import os
import psycopg

def get_db_connection():
    """
    Establish and return a connection to the PostgreSQL database.
    Tries to use the DATABASE_URL environment variable (set by Heroku).
    Falls back to local default connection parameters if DATABASE_URL is not set.
    """
    DATABASE_URL = os.environ.get("DATABASE_URL") # this is defined in heroku config vars
    if not DATABASE_URL:
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_PORT = os.getenv('DB_PORT', '5434')
        DB_NAME = os.getenv('DB_NAME', 'filings_db')
        DB_USER = os.getenv('DB_USER', 'postgres')
        DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    return psycopg.connect(DATABASE_URL)
