-- ============================================================
-- SMART E-COMMERCE - BUSINESS QUERIES
-- Database: SQLite (database/ecommerce.db)
-- Table principale: fact_sales
-- ============================================================


-- ------------------------------------------------------------
-- 1. KPIs GLOBAUX
-- ------------------------------------------------------------

-- 1.1 Total Revenue
SELECT
    ROUND(SUM(total_amount), 2) AS total_revenue
FROM fact_sales;


-- 1.2 Nombre total de commandes
SELECT
    COUNT(DISTINCT order_id) AS total_orders
FROM fact_sales;


-- 1.3 Nombre total de clients uniques
SELECT
    COUNT(DISTINCT customer_unique_id) AS total_customers
FROM fact_sales;


-- 1.4 Panier moyen (Average Order Value)
SELECT
    ROUND(SUM(total_amount) / COUNT(DISTINCT order_id), 2) AS avg_order_value
FROM fact_sales;


-- ------------------------------------------------------------
-- 2. ANALYSE DES VENTES
-- ------------------------------------------------------------

-- 2.1 Ventes mensuelles
SELECT
    year_month,
    COUNT(DISTINCT order_id)     AS orders,
    ROUND(SUM(total_amount), 2)  AS revenue
FROM fact_sales
GROUP BY year_month
ORDER BY year_month;


-- 2.2 Top 10 produits les plus vendus (par nombre de commandes)
SELECT
    product_id,
    COUNT(*)                    AS times_sold,
    COUNT(DISTINCT order_id)    AS unique_orders,
    ROUND(SUM(total_amount), 2) AS revenue
FROM fact_sales
GROUP BY product_id
ORDER BY times_sold DESC
LIMIT 10;


-- 2.3 Top 10 catégories par revenue
SELECT
    product_category_name_english AS category,
    COUNT(DISTINCT order_id)      AS orders,
    ROUND(SUM(total_amount), 2)   AS revenue
FROM fact_sales
GROUP BY product_category_name_english
ORDER BY revenue DESC
LIMIT 10;


-- 2.4 Top 10 villes par revenue
SELECT
    customer_city,
    customer_state,
    COUNT(DISTINCT order_id)     AS orders,
    ROUND(SUM(total_amount), 2)  AS revenue
FROM fact_sales
GROUP BY customer_city, customer_state
ORDER BY revenue DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 3. ANALYSE CLIENT (RFM)
-- ------------------------------------------------------------

-- 3.1 RFM par client
WITH snapshot AS (
    SELECT DATE(MAX(order_purchase_timestamp), '+1 day') AS snapshot_date
    FROM fact_sales
)
SELECT
    f.customer_unique_id,
    CAST(
        JULIANDAY((SELECT snapshot_date FROM snapshot)) -
        JULIANDAY(MAX(f.order_purchase_timestamp))
    AS INTEGER)                          AS recency_days,
    COUNT(DISTINCT f.order_id)           AS frequency,
    ROUND(SUM(f.total_amount), 2)        AS monetary
FROM fact_sales f
GROUP BY f.customer_unique_id
ORDER BY monetary DESC
LIMIT 20;


-- 3.2 Top 10 meilleurs clients (par revenue)
SELECT
    customer_unique_id,
    customer_city,
    customer_state,
    COUNT(DISTINCT order_id)     AS orders,
    ROUND(SUM(total_amount), 2)  AS total_spent
FROM fact_sales
GROUP BY customer_unique_id, customer_city, customer_state
ORDER BY total_spent DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 4. ANALYSE DES PAIEMENTS
-- ------------------------------------------------------------

-- 4.1 Répartition par type de paiement
SELECT
    payment_type,
    COUNT(*)                    AS transactions,
    ROUND(SUM(payment_value), 2) AS total_value
FROM payments
GROUP BY payment_type
ORDER BY total_value DESC;


-- 4.2 Nombre de versements (installments)
SELECT
    payment_installments,
    COUNT(*) AS transactions
FROM payments
GROUP BY payment_installments
ORDER BY payment_installments;


-- ------------------------------------------------------------
-- 5. ANALYSE DES AVIS
-- ------------------------------------------------------------

-- 5.1 Distribution des scores d'avis
SELECT
    review_score,
    COUNT(*)                                AS total,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM reviews
GROUP BY review_score
ORDER BY review_score;


-- 5.2 Score moyen par catégorie
SELECT
    f.product_category_name_english AS category,
    ROUND(AVG(r.review_score), 2)   AS avg_score,
    COUNT(*)                         AS num_reviews
FROM fact_sales f
JOIN reviews r ON f.order_id = r.order_id
GROUP BY f.product_category_name_english
HAVING COUNT(*) >= 100
ORDER BY avg_score DESC
LIMIT 15;


-- ------------------------------------------------------------
-- 6. ANALYSE DES LIVRAISONS
-- ------------------------------------------------------------

-- 6.1 Délai moyen de livraison (en jours)
SELECT
    ROUND(
        AVG(
            JULIANDAY(order_delivered_customer_date) -
            JULIANDAY(order_purchase_timestamp)
        ), 2
    ) AS avg_delivery_days
FROM orders
WHERE order_delivered_customer_date IS NOT NULL;


-- 6.2 Statut des commandes
SELECT
    order_status,
    COUNT(*)                                AS total,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM orders
GROUP BY order_status
ORDER BY total DESC;