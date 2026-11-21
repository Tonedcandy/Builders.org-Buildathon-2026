import glob
import json
import os
from datetime import datetime, timezone
import duckdb

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", "8"))


def parse_f1_ms(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_csv_field(field: str):
    if '"' not in field:
        return field
    if len(field) >= 2 and field[0] == '"' and field[-1] == '"' and '"' not in field[1:-1]:
        return field[1:-1]
    return None


def build_id_map(links_dir: str):
    parent = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: str, b: str):
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        ra, rb = find(a), find(b)
        if ra != rb:
            # Byte-wise smallest ID becomes the canonical root
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for path in sorted(glob.glob(os.path.join(links_dir, "*.csv"))):
        with open(path, "rb") as f:
            raw = f.read()
        lines = raw.split(b"\n")
        for raw_line in lines[1:]:  # skip header
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            parts = line.split(",")
            if len(parts) != 2:
                continue
            a = parse_csv_field(parts[0])
            b = parse_csv_field(parts[1])
            if a is None or b is None or a == "" or b == "":
                continue
            union(a, b)

    # Only store IDs that actually map to a different canonical ID
    return [(uid, find(uid)) for uid in parent if find(uid) != uid]


def main():
    with open(os.path.join(DATA, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    win_start = parse_f1_ms(manifest["window_start"])
    win_end = parse_f1_ms(manifest["window_end"])
    click_win_start = win_start - 7 * 86_400_000

    os.makedirs(os.path.join(OUT, "user_daily"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "campaign_daily"), exist_ok=True)
    tmp_dir = os.path.join(SCRATCH, "duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    id_pairs = build_id_map(os.path.join(DATA, "links"))

    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone = 'UTC';")
    con.execute(f"SET threads = {NCPU};")
    con.execute(f"SET temp_directory = '{tmp_dir}';")

    # Register ID mapping table
    con.execute("CREATE TABLE id_map (raw_id VARCHAR, canonical_id VARCHAR);")
    if id_pairs:
        con.executemany("INSERT INTO id_map VALUES (?, ?);", id_pairs)

    # Helper macros for strict CSV dialect and timestamp parsing
    con.execute("""
        CREATE MACRO valid_field(s) AS (
            NOT contains(s, '"')
            OR (
                length(s) >= 2
                AND starts_with(s, '"')
                AND ends_with(s, '"')
                AND NOT contains(substr(s, 2, length(s) - 2), '"')
            )
        );

        CREATE MACRO unquote(s) AS (
            CASE WHEN starts_with(s, '"') THEN substr(s, 2, length(s) - 2) ELSE s END
        );

        CREATE MACRO parse_ts(s) AS (
            CASE
                WHEN regexp_full_match(s, '^[0-9]+$') THEN
                    CASE
                        WHEN ltrim(s, '0') = '' THEN 0::BIGINT
                        ELSE try_cast(ltrim(s, '0') AS BIGINT)
                    END
                WHEN regexp_full_match(s, '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$') THEN
                    epoch_ms(try_strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
                WHEN regexp_full_match(s, '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}$')
                     AND CAST(substr(s, 21, 2) AS INT) <= 23
                     AND CAST(substr(s, 24, 2) AS INT) <= 59 THEN
                    CASE
                        WHEN substr(s, 20, 1) = '+' THEN
                            epoch_ms(try_strptime(substr(s, 1, 19), '%Y-%m-%dT%H:%M:%S'))
                            - (CAST(substr(s, 21, 2) AS BIGINT) * 3600 + CAST(substr(s, 24, 2) AS BIGINT) * 60) * 1000
                        ELSE
                            epoch_ms(try_strptime(substr(s, 1, 19), '%Y-%m-%dT%H:%M:%S'))
                            + (CAST(substr(s, 21, 2) AS BIGINT) * 3600 + CAST(substr(s, 24, 2) AS BIGINT) * 60) * 1000
                    END
                ELSE NULL
            END
        );

        CREATE MACRO parse_price(s) AS (
            CASE
                WHEN regexp_full_match(s, '^[0-9]+$') THEN
                    CASE
                        WHEN ltrim(s, '0') = '' THEN 0::BIGINT
                        ELSE try_cast(ltrim(s, '0') AS BIGINT)
                    END
                ELSE NULL
            END
        );
    """)

    # Load clean dimension tables
    con.execute(f"""
        CREATE TABLE products AS
        SELECT product_id, category
        FROM read_csv('{DATA}/products.csv', header=true, new_line='\\n', quote='"', escape='"', columns={{'product_id': 'VARCHAR', 'category': 'VARCHAR'}});

        CREATE TABLE campaigns AS
        SELECT campaign_id, channel
        FROM read_csv('{DATA}/campaigns.csv', header=true, new_line='\\n', quote='"', escape='"', columns={{'campaign_id': 'VARCHAR', 'channel': 'VARCHAR'}});
    """)

    # 1-4. Parse, dedup, validate, and drop late events
    con.execute(f"""
        CREATE TABLE valid_events AS
        WITH raw_lines AS (
            SELECT string_split(line, ',') AS p
            FROM (
                SELECT unnest(string_split(content, chr(10))[2:]) AS line
                FROM read_text('{DATA}/events/*.csv')
            )
            WHERE line != ''
        ),
        parsed AS (
            SELECT
                unquote(p[1]) AS event_id,
                unquote(p[2]) AS user_id,
                unquote(p[3]) AS ts_raw,
                unquote(p[4]) AS ingest_ts_raw,
                unquote(p[5]) AS event_type,
                unquote(p[6]) AS product_id,
                unquote(p[7]) AS price_raw
            FROM raw_lines
            WHERE len(p) = 7
              AND valid_field(p[1]) AND valid_field(p[2]) AND valid_field(p[3])
              AND valid_field(p[4]) AND valid_field(p[5]) AND valid_field(p[6])
              AND valid_field(p[7])
              AND unquote(p[1]) != ''
        ),
        deduped AS (
            SELECT *
            FROM parsed
            QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts_raw ASC) = 1
        ),
        typed AS (
            SELECT
                COALESCE(m.canonical_id, d.user_id) AS person_id,
                parse_ts(d.ts_raw) AS ts,
                epoch_ms(strptime(d.ingest_ts_raw, '%Y-%m-%dT%H:%M:%SZ')) AS ingest_ts,
                d.event_type,
                d.product_id,
                CASE
                    WHEN d.event_type = 'purchase' THEN parse_price(d.price_raw)
                    ELSE 0::BIGINT
                END AS price_cents
            FROM deduped d
            LEFT JOIN id_map m ON d.user_id = m.raw_id
            WHERE d.user_id != ''
              AND d.product_id != ''
              AND d.event_type IN ('view', 'cart', 'purchase')
        )
        SELECT person_id, ts, event_type, product_id, price_cents
        FROM typed
        WHERE ts IS NOT NULL
          AND ts >= {win_start} AND ts < {win_end}
          AND (event_type != 'purchase' OR (price_cents IS NOT NULL AND price_cents BETWEEN 1 AND 10000000))
          AND (ingest_ts - ts) <= 86400000;
    """)

    # 5-6. Sessionize events
    con.execute("""
        CREATE TABLE sessionized_events AS
        WITH gaps AS (
            SELECT
                person_id,
                ts,
                event_type,
                product_id,
                price_cents,
                CASE
                    WHEN LAG(ts) OVER (PARTITION BY person_id ORDER BY ts) IS NULL
                      OR ts - LAG(ts) OVER (PARTITION BY person_id ORDER BY ts) > 1800000
                    THEN 1 ELSE 0
                END AS is_new
            FROM valid_events
        ),
        numbered AS (
            SELECT
                person_id,
                ts,
                event_type,
                product_id,
                price_cents,
                SUM(is_new) OVER (
                    PARTITION BY person_id
                    ORDER BY ts
                    RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS session_num
            FROM gaps
        )
        SELECT
            person_id,
            session_num,
            ts,
            MIN(ts) OVER (PARTITION BY person_id, session_num) AS session_start,
            CAST(epoch_ms(MIN(ts) OVER (PARTITION BY person_id, session_num)) AS DATE) AS day,
            event_type,
            product_id,
            price_cents
        FROM numbered;

        CREATE TABLE sessions AS
        SELECT
            person_id,
            session_num,
            MIN(session_start) AS session_start,
            MIN(day) AS day,
            COUNT(*) AS events,
            SUM(price_cents) AS revenue_cents,
            MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS has_purchase
        FROM sessionized_events
        GROUP BY person_id, session_num;
    """)

    # Parse, dedup, and validate clicks
    con.execute(f"""
        CREATE TABLE valid_clicks AS
        WITH raw_lines AS (
            SELECT string_split(line, ',') AS p
            FROM (
                SELECT unnest(string_split(content, chr(10))[2:]) AS line
                FROM read_text('{DATA}/clicks/*.csv')
            )
            WHERE line != ''
        ),
        parsed AS (
            SELECT
                unquote(p[1]) AS click_id,
                unquote(p[2]) AS user_id,
                unquote(p[3]) AS campaign_id,
                unquote(p[4]) AS ts_raw
            FROM raw_lines
            WHERE len(p) = 4
              AND valid_field(p[1]) AND valid_field(p[2])
              AND valid_field(p[3]) AND valid_field(p[4])
              AND unquote(p[1]) != ''
        ),
        deduped AS (
            SELECT click_id, ANY_VALUE(user_id) AS user_id, ANY_VALUE(campaign_id) AS campaign_id, ANY_VALUE(ts_raw) AS ts_raw
            FROM parsed
            GROUP BY click_id
        )
        SELECT
            d.click_id,
            COALESCE(m.canonical_id, d.user_id) AS person_id,
            d.campaign_id,
            parse_ts(d.ts_raw) AS ts
        FROM deduped d
        JOIN campaigns c ON d.campaign_id = c.campaign_id
        LEFT JOIN id_map m ON d.user_id = m.raw_id
        WHERE d.user_id != ''
          AND parse_ts(d.ts_raw) IS NOT NULL
          AND parse_ts(d.ts_raw) >= {click_win_start}
          AND parse_ts(d.ts_raw) < {win_end};
    """)

    # Output 1: user_daily
    con.execute(f"""
        COPY (
            WITH user_agg AS (
                SELECT
                    person_id AS user_id,
                    day,
                    COUNT(*) AS sessions,
                    SUM(events) AS events,
                    SUM(revenue_cents) AS revenue_cents
                FROM sessions
                GROUP BY person_id, day
            ),
            cat_rev AS (
                SELECT
                    e.person_id AS user_id,
                    e.day,
                    COALESCE(p.category, 'unknown') AS category,
                    SUM(e.price_cents) AS cat_revenue
                FROM sessionized_events e
                LEFT JOIN products p ON e.product_id = p.product_id
                WHERE e.event_type = 'purchase'
                GROUP BY e.person_id, e.day, COALESCE(p.category, 'unknown')
            ),
            top_cat AS (
                SELECT user_id, day, category AS top_category
                FROM cat_rev
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY user_id, day
                    ORDER BY cat_revenue DESC, category ASC
                ) = 1
            )
            SELECT
                u.user_id,
                u.day,
                CAST(u.sessions AS BIGINT) AS sessions,
                CAST(u.events AS BIGINT) AS events,
                CAST(u.revenue_cents AS BIGINT) AS revenue_cents,
                t.top_category
            FROM user_agg u
            LEFT JOIN top_cat t ON u.user_id = t.user_id AND u.day = t.day
        ) TO '{OUT}/user_daily/part-0.parquet' (FORMAT PARQUET);
    """)

    # Output 2: campaign_daily
    con.execute(f"""
        COPY (
            WITH qualifying_clicks AS (
                SELECT
                    s.person_id,
                    s.session_num,
                    s.day,
                    c.click_id,
                    c.campaign_id,
                    c.ts AS click_ts
                FROM sessions s
                JOIN valid_clicks c
                  ON s.person_id = c.person_id
                 AND c.ts <= s.session_start
                 AND c.ts >= s.session_start - 604800000
            ),
            credited_click AS (
                SELECT person_id, session_num, campaign_id
                FROM qualifying_clicks
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY person_id, session_num
                    ORDER BY click_ts DESC, click_id ASC
                ) = 1
            ),
            credited_agg AS (
                SELECT
                    COALESCE(cc.campaign_id, 'organic') AS campaign_id,
                    s.day,
                    COUNT(*) AS sessions,
                    SUM(s.has_purchase) AS purchasing_sessions,
                    SUM(s.revenue_cents) AS revenue_cents
                FROM sessions s
                LEFT JOIN credited_click cc
                  ON s.person_id = cc.person_id AND s.session_num = cc.session_num
                GROUP BY COALESCE(cc.campaign_id, 'organic'), s.day
            ),
            assisted_agg AS (
                SELECT
                    campaign_id,
                    day,
                    COUNT(*) AS assisted_sessions
                FROM (
                    SELECT DISTINCT person_id, session_num, day, campaign_id
                    FROM qualifying_clicks
                )
                GROUP BY campaign_id, day
            )
            SELECT
                COALESCE(cr.campaign_id, a.campaign_id) AS campaign_id,
                CASE
                    WHEN COALESCE(cr.campaign_id, a.campaign_id) = 'organic' THEN 'organic'
                    ELSE c.channel
                END AS channel,
                COALESCE(cr.day, a.day) AS day,
                CAST(COALESCE(cr.sessions, 0) AS BIGINT) AS sessions,
                CAST(COALESCE(cr.purchasing_sessions, 0) AS BIGINT) AS purchasing_sessions,
                CAST(COALESCE(cr.revenue_cents, 0) AS BIGINT) AS revenue_cents,
                CAST(COALESCE(a.assisted_sessions, 0) AS BIGINT) AS assisted_sessions
            FROM credited_agg cr
            FULL OUTER JOIN assisted_agg a
              ON cr.campaign_id = a.campaign_id AND cr.day = a.day
            LEFT JOIN campaigns c
              ON COALESCE(cr.campaign_id, a.campaign_id) = c.campaign_id
        ) TO '{OUT}/campaign_daily/part-0.parquet' (FORMAT PARQUET);
    """)


if __name__ == "__main__":
    main()