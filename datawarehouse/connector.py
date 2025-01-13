
import mysql.connector 
from sqlalchemy import create_engine
import pandas as pd



# Database connection details
# Connect to MySQL Database
try:
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        port=3307, #specify mysql port
        password='',
        database='etlolapoperations'
    )
    cursor = conn.cursor()
    print("Connected to the database.")

except mysql.connector.Error as e:
    print(f"Error while connecting to MySQL: {e}")

