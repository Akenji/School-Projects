##   Code to load the transformed data into the MySQL data warehouse
import mysql.connector
import pandas as pd

# Paths to your CSV files
customer_csv = 'Customer_Dim.csv'
product_csv = 'Product_Dim.csv'
date_csv = 'Date_Dim.csv'
sales_csv = 'Sales_Fact.csv'

# Database connection details
# Connect to MySQL Database
try:
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        port=3307,
        password='',
        database='etlolapoperations'
    )
    cursor = conn.cursor()
   

    # Load and insert Customer_Dim data
    customer_df = pd.read_csv(customer_csv)
    for _, row in customer_df.iterrows():
        cursor.execute("""
            INSERT INTO customer_dim (customer_id, customer_name, customerAddress, city, Region, customerPhone, customerEmail)
            VALUES (varchar, varchar, text, varchar, text, varchar,varchar)
        """, (row['customer_id'], row['customer_name'], row['customerAddress'], row['city'], row['Region'], row['customerPhone'], row['customerEmail']))

    # Load and insert Product_Dim data
    product_df = pd.read_csv(product_csv)
    for _, row in product_df.iterrows():
        cursor.execute("""
            INSERT INTO product_dim (product_id, product_name, category, price)
            VALUES (varchar, varchar, varchar, decimal)
        """, (row['product_id'], row['product_name'], row['category'], row['price']))

    # Load and insert Date_Dim data
    date_df = pd.read_csv(date_csv)
    for _, row in date_df.iterrows():
        cursor.execute("""
            INSERT INTO date_dim (date_id, date, day_of_week, month, year,day,quateer)
            VALUES (int, date, varchar, varchar, int,text,int)
        """, (row['date_id'], row['date'], row['day_of_week'], row['month'], row['year'], row['day'],row['quater']))

    # Load and insert Sales_Fact data
    sales_df = pd.read_csv(sales_csv)
    for _, row in sales_df.iterrows():
        cursor.execute("""
            INSERT INTO sales_fact (sales_id,customer_id, product_id, date_id, quantity, total_amount)
            VALUES (varchar, varchar, varchar, int, int, decimal)
        """, (row['sales_id'],row['customer_id'], row['product_id'], row['date_id'], row['quantity'], row['total_amount']))

    # Commit the transactions
    conn.commit()
    print("Data inserted successfully.")

except mysql.connector.Error as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection is closed.")
