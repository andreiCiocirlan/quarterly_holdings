# quarterly_holdings
 
# App should startup and populate the database from the data.sql file
# In order to obtain the dump.sql:

# 1. run import_13f_data.py to populate the database
# 2. run in a new docker terminal
    docker exec -t quarterly_holdings-postgres-1 pg_dump -U postgres -d filings_db > dump.sql
    
    For Heroku: new terminal at docker-entrypoint-initdb.d
        heroku pg:psql -a quarterly-holdings -f data.sql
# 3. rename dump.sql to data.sql and move it inside docker-entrypoint-initdb.d
# 4. Now the app starts normally if the database is populated, otherwise inserts data.


# Ports:
Inside Docker python-app        : postgres  : 5434
Outside Docker (host machine)   : localhost : 5432