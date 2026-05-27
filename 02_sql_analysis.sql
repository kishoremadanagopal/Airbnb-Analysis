-- =============================================================================
-- NYC Airbnb 2019 — SQL Analysis
-- =============================================================================
-- Run against the SQLite database built by 02_sql_demo.py
-- (which loads airbnb_clean.csv into a table called `listings`).
--
-- These queries demonstrate analytical SQL patterns:
--   - CTEs for readable multi-step logic
--   - Window functions for ranking, percentiles, and running comparisons
--   - Conditional aggregation (CASE WHEN inside SUM/AVG)
--   - Self-joins for cohort comparisons
--
-- Each query is independent. The 02_sql_demo.py runner executes them in
-- order and prints the results.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Q1. Borough summary: listings, mean/median price, availability
-- -----------------------------------------------------------------------------
-- Note: SQLite doesn't have a native PERCENTILE_CONT, so we compute the median
-- with a window-function trick (ROW_NUMBER over an ordered partition).
WITH ranked AS (
    SELECT
        neighbourhood_group,
        price,
        ROW_NUMBER() OVER (PARTITION BY neighbourhood_group ORDER BY price) AS rn,
        COUNT(*)    OVER (PARTITION BY neighbourhood_group)                 AS n
    FROM listings
),
medians AS (
    SELECT neighbourhood_group, AVG(price) AS median_price
    FROM ranked
    WHERE rn IN (n / 2, n / 2 + 1)
    GROUP BY neighbourhood_group
)
SELECT
    l.neighbourhood_group                       AS borough,
    COUNT(*)                                    AS listings,
    ROUND(AVG(l.price), 2)                      AS mean_price,
    ROUND(m.median_price, 2)                    AS median_price,
    ROUND(AVG(l.availability_365), 1)           AS avg_avail_days,
    ROUND(AVG(l.number_of_reviews), 1)          AS avg_reviews
FROM listings l
JOIN medians  m USING (neighbourhood_group)
GROUP BY l.neighbourhood_group, m.median_price
ORDER BY mean_price DESC;


-- -----------------------------------------------------------------------------
-- Q2. Top 10 hosts by listing count — who dominates the market?
-- -----------------------------------------------------------------------------
SELECT
    host_id,
    host_name,
    COUNT(*)                AS num_listings,
    ROUND(AVG(price), 2)    AS avg_price,
    ROUND(SUM(price * availability_365 / 365.0), 0) AS est_annual_revenue_capacity
FROM listings
GROUP BY host_id, host_name
ORDER BY num_listings DESC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- Q3. Price percentile rank within each borough
-- -----------------------------------------------------------------------------
-- Useful for the dashboard: given a listing, where does it sit in its borough?
-- Sample shows the top 3 and bottom 3 by percentile within Manhattan.
WITH ranked AS (
    SELECT
        id, name, neighbourhood_group, neighbourhood, room_type, price,
        PERCENT_RANK() OVER (PARTITION BY neighbourhood_group ORDER BY price) AS price_pct_rank
    FROM listings
)
SELECT id, name, neighbourhood, room_type, price,
       ROUND(price_pct_rank * 100, 1) AS price_percentile
FROM ranked
WHERE neighbourhood_group = 'Manhattan'
  AND (price_pct_rank >= 0.999 OR price_pct_rank <= 0.001)
ORDER BY price_pct_rank DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- Q4. Room-type mix by borough — conditional aggregation
-- -----------------------------------------------------------------------------
-- Shows the borough-level share of each room type in a single row per borough.
-- This is the kind of pivot you'd otherwise do in pandas with crosstab.
SELECT
    neighbourhood_group AS borough,
    COUNT(*) AS total,
    SUM(CASE WHEN room_type = 'Entire home/apt' THEN 1 ELSE 0 END) AS entire_home,
    SUM(CASE WHEN room_type = 'Private room'    THEN 1 ELSE 0 END) AS private_room,
    SUM(CASE WHEN room_type = 'Shared room'     THEN 1 ELSE 0 END) AS shared_room,
    ROUND(100.0 * SUM(CASE WHEN room_type = 'Entire home/apt' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_entire_home
FROM listings
GROUP BY neighbourhood_group
ORDER BY total DESC;


-- -----------------------------------------------------------------------------
-- Q5. Most reviewed neighborhoods (proxy for booking volume)
-- -----------------------------------------------------------------------------
-- Combines mean price with review activity to find "popular and pricey" vs
-- "popular and cheap" neighborhoods. Filtered to neighborhoods with at least
-- 100 listings to avoid noisy small-sample winners.
SELECT
    neighbourhood,
    neighbourhood_group AS borough,
    COUNT(*)                            AS listings,
    ROUND(AVG(price), 2)                AS avg_price,
    ROUND(SUM(number_of_reviews), 0)    AS total_reviews,
    ROUND(AVG(reviews_per_month), 2)    AS avg_reviews_per_month
FROM listings
GROUP BY neighbourhood, neighbourhood_group
HAVING COUNT(*) >= 100
ORDER BY total_reviews DESC
LIMIT 15;


-- -----------------------------------------------------------------------------
-- Q6. Commercial vs casual hosts — segmentation analysis
-- -----------------------------------------------------------------------------
-- Classify hosts by listing count and compare price/availability between them.
-- "Commercial" = 5+ listings (clear professional operation).
WITH host_segments AS (
    SELECT
        host_id,
        COUNT(*) AS host_listings,
        CASE
            WHEN COUNT(*) = 1 THEN '1 listing (casual)'
            WHEN COUNT(*) BETWEEN 2 AND 4 THEN '2-4 listings (multi-host)'
            ELSE '5+ listings (commercial)'
        END AS segment
    FROM listings
    GROUP BY host_id
)
SELECT
    hs.segment,
    COUNT(DISTINCT hs.host_id)         AS num_hosts,
    COUNT(*)                            AS num_listings,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM listings), 1) AS pct_of_market,
    ROUND(AVG(l.price), 2)              AS avg_price,
    ROUND(AVG(l.availability_365), 1)   AS avg_avail_days
FROM listings l
JOIN host_segments hs USING (host_id)
GROUP BY hs.segment
ORDER BY MIN(hs.host_listings);
