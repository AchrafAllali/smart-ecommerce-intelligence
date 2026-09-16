"""
Prompts and schema description for the RAG assistant.
"""

SCHEMA_DESCRIPTION = """
DATABASE SCHEMA (SQLite):

1. fact_sales
   - order_id, customer_unique_id, product_id, seller_id
   - price, freight_value, total_amount
   - order_purchase_timestamp, year, month, quarter, year_month, day_of_week
   - payment_type, payment_installments, payment_value
   - product_category_name_english
   - customer_city, customer_state

2. dim_customers
   - customer_id, customer_unique_id, customer_city, customer_state

3. dim_products
   - product_id, product_category_name, product_category_name_english

4. dim_sellers
   - seller_id, seller_city, seller_state

5. orders
   - order_id, customer_id, order_status, order_purchase_timestamp,
     order_delivered_customer_date, order_estimated_delivery_date

6. reviews
   - review_id, order_id, review_score

7. payments
   - order_id, payment_type, payment_installments, payment_value

8. customer_segments
   - customer_unique_id, recency, frequency, monetary,
     R_score, F_score, M_score, segment

9. customer_churn
   - customer_unique_id, recency, frequency, monetary,
     churn, churn_probability, churn_prediction

10. popular_products
    - product_id, category, n_purchases, n_customers

11. recommendations
    - customer_unique_id, product_id, rank, score
"""