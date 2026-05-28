"""Run a handful of realistic queries against OM-tracked Snowflake tables
so the next usage-ingestion has interesting history to pull. Idempotent —
the queries are pure SELECTs that don't change data."""

import os
import textwrap
import time

import snowflake.connector

QUERIES = [
    """SELECT c_mktsegment, COUNT(*) cnt
         FROM snowflake_sample_data.tpch_sf1.customer
        GROUP BY 1
        ORDER BY 2 DESC""",
    """SELECT c_nationkey, AVG(c_acctbal) avg_bal
         FROM snowflake_sample_data.tpch_sf1.customer
        GROUP BY 1""",
    """SELECT o_orderpriority, SUM(o_totalprice) revenue
         FROM snowflake_sample_data.tpch_sf1.orders
        WHERE o_orderdate >= DATE '1995-01-01'
        GROUP BY 1""",
    """SELECT l.l_returnflag, l.l_linestatus, SUM(l.l_extendedprice * (1 - l.l_discount)) net_rev
         FROM snowflake_sample_data.tpch_sf1.lineitem l
         JOIN snowflake_sample_data.tpch_sf1.orders   o ON l.l_orderkey = o.o_orderkey
        GROUP BY 1, 2""",
    """SELECT s.s_name, SUM(ps.ps_supplycost * ps.ps_availqty) inventory_value
         FROM snowflake_sample_data.tpch_sf1.supplier s
         JOIN snowflake_sample_data.tpch_sf1.partsupp ps ON s.s_suppkey = ps.ps_suppkey
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 25""",
    """SELECT n.n_name region, COUNT(c.c_custkey) customers
         FROM snowflake_sample_data.tpch_sf1.nation   n
         JOIN snowflake_sample_data.tpch_sf1.customer c ON c.c_nationkey = n.n_nationkey
        GROUP BY 1""",
    """SELECT COUNT(*) FROM snowflake_sample_data.tpch_sf1.lineitem""",
    """SELECT MAX(o_orderdate), MIN(o_orderdate)
         FROM snowflake_sample_data.tpch_sf1.orders""",
]


def main() -> None:
    conn = snowflake.connector.connect(
        user=os.environ["SF_USER"],
        password=os.environ["SF_PWD"],
        account=os.environ["SF_ACC"],
        warehouse=os.environ["SF_WH"],
        role=os.environ["SF_ROLE"],
    )
    cur = conn.cursor()
    try:
        for i, q in enumerate(QUERIES, 1):
            t0 = time.time()
            cur.execute(textwrap.dedent(q))
            n = len(cur.fetchall())
            print(f"  [{i}/{len(QUERIES)}] rows={n}  elapsed={time.time()-t0:.2f}s  qid={cur.sfqid}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
