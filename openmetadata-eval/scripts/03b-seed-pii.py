"""Seed a small realistic-PII table in the trial Snowflake account so that
the auto-classification workflow has something meaningful to detect.

Run via the openmetadata-ingestion image (already has snowflake-connector-python).
"""

import os
import snowflake.connector

SCHEMA = "OM_EVAL"
TABLE = "PEOPLE"

ROWS = [
    ("Alice Johnson", "alice.johnson@gmail.com", "212-555-0143", "123-45-6789",
     "1980-04-12", "742 Evergreen Terrace, Springfield, IL 62704"),
    ("Bob Williams", "bobw@yahoo.com", "415-555-9821", "987-65-4321",
     "1975-09-30", "1600 Amphitheatre Pkwy, Mountain View, CA 94043"),
    ("Charlie Brown", "charlie.brown@outlook.com", "646-555-0177", "555-66-7777",
     "1992-12-01", "20 W 34th St, New York, NY 10001"),
    ("Diana Garcia", "dgarcia@protonmail.com", "305-555-2210", "111-22-3333",
     "1988-07-22", "1 Infinite Loop, Cupertino, CA 95014"),
    ("Eve Martinez", "eve.martinez@company.com", "713-555-7711", "444-55-6666",
     "1990-03-15", "500 Terry A Francois Blvd, San Francisco, CA 94158"),
    ("Frank Patel", "frankp@example.org", "202-555-0144", "222-33-4444",
     "1983-11-08", "350 5th Ave, New York, NY 10118"),
    ("Grace Kim", "grace.kim@workmail.com", "503-555-7090", "333-44-5555",
     "1995-06-19", "999 N Northlake Way, Seattle, WA 98103"),
    ("Henry O'Connor", "h.oconnor@business.co", "619-555-2030", "888-99-0000",
     "1978-02-25", "1313 Mockingbird Ln, Hollywood, CA 90028"),
]

CREATE_SCHEMA = f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"
DROP_TABLE = f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}"
CREATE_TABLE = f"""
CREATE TABLE {SCHEMA}.{TABLE} (
    FULL_NAME      STRING,
    EMAIL_ADDRESS  STRING,
    PHONE_NUMBER   STRING,
    SSN            STRING,
    DATE_OF_BIRTH  DATE,
    HOME_ADDRESS   STRING
)
COMMENT = 'Synthetic PII fixture for OpenMetadata auto-classification eval.'
"""

INSERT = f"INSERT INTO {SCHEMA}.{TABLE} VALUES (%s,%s,%s,%s,%s,%s)"


def main() -> None:
    conn = snowflake.connector.connect(
        user=os.environ["SF_USER"],
        password=os.environ["SF_PWD"],
        account=os.environ["SF_ACC"],
        warehouse=os.environ["SF_WH"],
        role=os.environ["SF_ROLE"],
        database=os.environ["SF_DB"],
    )
    cur = conn.cursor()
    try:
        cur.execute(f"USE DATABASE {os.environ['SF_DB']}")
        cur.execute(CREATE_SCHEMA)
        print(f"schema {SCHEMA}: {cur.fetchall()}")
        cur.execute(DROP_TABLE)
        print(f"drop  : {cur.fetchall()}")
        cur.execute(CREATE_TABLE)
        print(f"create: {cur.fetchall()}")
        cur.executemany(INSERT, ROWS)
        print(f"insert: rowcount={cur.rowcount}")
        cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.{TABLE}")
        print(f"verify: {cur.fetchone()[0]} rows in {SCHEMA}.{TABLE}")
        cur.execute(f"SELECT * FROM {SCHEMA}.{TABLE} LIMIT 2")
        print("preview:", cur.fetchall())
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
