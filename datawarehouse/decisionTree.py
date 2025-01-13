# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import mysql.connector
# import matplotlib.pyplot as plt

# # Database connection
# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     port=3307,
#     password="",
#     database="etlolapoperations"
# )

# # Simplified query to start with
# query = """
# SELECT 
#     s.customer_id,
#     p.category,
#     SUM(s.quantity) as total_quantity,
#     SUM(s.total_amount) as total_spent,
#     COUNT(DISTINCT s.sale_id) as number_of_transactions
# FROM sales_fact s
# JOIN Product_Dim p ON s.product_id = p.product_id
# GROUP BY s.customer_id, p.category
# """

# # Load data
# cursor = db.cursor()
# cursor.execute(query)
# results = cursor.fetchall()

# # Convert to DataFrame
# data = pd.DataFrame(results, columns=['customer_id', 'category', 'total_quantity', 
#                                     'total_spent', 'number_of_transactions'])

# # Data cleaning
# def clean_data(df):
#     # Replace infinite values with NaN
#     df = df.replace([np.inf, -np.inf], np.nan)
    
#     # Fill NaN values with median
#     for column in df.select_dtypes(include=[np.number]).columns:
#         df[column] = df[column].fillna(df[column].median())
    
#     # Remove extreme outliers (values beyond 3 standard deviations)
#     for column in df.select_dtypes(include=[np.number]).columns:
#         mean = df[column].mean()
#         std = df[column].std()
#         df = df[df[column].between(mean - 3*std, mean + 3*std)]
    
#     return df

# # Clean the data
# data = clean_data(data)

# # Prepare features and target
# X = data[['total_quantity', 'total_spent', 'number_of_transactions']]
# le = LabelEncoder()
# y = le.fit_transform(data['category'])

# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Train Decision Tree
# dt_classifier = DecisionTreeClassifier(
#     random_state=42,
#     max_depth=5,
#     min_samples_split=20,
#     min_samples_leaf=10
# )

# # Fit the model
# dt_classifier.fit(X_train, y_train)

# # Evaluate model
# train_accuracy = dt_classifier.score(X_train, y_train)
# test_accuracy = dt_classifier.score(X_test, y_test)

# print(f"Training Accuracy: {train_accuracy:.2f}")
# print(f"Testing Accuracy: {test_accuracy:.2f}")

# # Feature importance
# feature_importance = pd.DataFrame({
#     'feature': X.columns,
#     'importance': dt_classifier.feature_importances_
# }).sort_values('importance', ascending=False)

# print("\nFeature Importance:")
# print(feature_importance)

# # Visualize decision tree
# plt.figure(figsize=(10,5))
# plot_tree(dt_classifier, feature_names=X.columns, class_names=le.classes_, filled=True)
# plt.show()

# # Function to predict category for new customers
# def predict_category(quantity, amount, transactions):
#     # Scale the input features
#     features = scaler.transform([[quantity, amount, transactions]])
#     prediction = dt_classifier.predict(features)
#     return le.inverse_transform(prediction)[0]

# # Example prediction
# try:
#     example_prediction = predict_category(10, 1000, 5)
#     print(f"\nPredicted category: {example_prediction}")
# except Exception as e:
#     print(f"Prediction error: {e}")

# db.close()

import pymysql
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np  # Import NumPy

# Database connection details
db_config = {
    'host': 'localhost',  # Host where the database is running
    'user': 'root',  # Your database username
    'password': '',  # Your database password
    'port': 3307,
    'database': 'retaildw'  # The database you're connecting to
}

# Step 1: Connect to the MySQL database
try:
    # Establish connection
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    print("Connected to the database.")
   
    # Step 2: Load data from the RetailDW database
    query = """
    SELECT sales_fact.*, date_dim.date, product_dim.category
    FROM sales_fact
    JOIN Date_Dim ON sales_fact.date_id = date_dim.date_id
    JOIN product_dim ON sales_fact.product_id = product_dim.product_id
    """
   
    # Load the data into a pandas DataFrame
    data = pd.read_sql(query, conn)
   
    # Step 3: Display the first few rows of the dataset to check the data
    print("First few rows of the dataset:")
    print(data.head())

    # Step 4: Check for missing data
    print("Missing values per column:")
    print(data.isnull().sum())

    # Step 5: Fill missing values using forward fill method
    data.fillna(method='ffill', inplace=True)
    print("\nData after filling missing values:")
    print(data.isnull().sum())  # Check again for any remaining missing values
   
    # Step 6: Randomly assign product categories (if not available)
    # Check if there are missing categories and replace them
    missing_category_count = data['category'].isnull().sum()
    if missing_category_count > 0:
        print(f"Filling {missing_category_count} missing category values with random categories.")
        # Generate random categories for missing values
        random_categories = np.random.choice(['Electronics', 'Clothing', 'Food'], size=missing_category_count)
        # Assign the generated random categories to the missing entries
        data.loc[data['category'].isnull(), 'category'] = random_categories
   
    # Step 7: Define Features and Target Variable for Classification
    X = data[['customer_id', 'product_id', 'quantity']]  # Features: customer_id, product_id, and quantity
    y = data['category']  # Target: product category
   
    # Step 8: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 9: Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)  # Create Decision Tree Classifier
    clf.fit(X_train, y_train)  # Train the model on the training data

    # Step 10: Make Predictions
    y_pred = clf.predict(X_test)  # Predict categories for the test set

    print('------------')  # Print
    print('Tested category:')  # Print tested
    print(y_test)  # Print testing
    print('------------')  # Print
    print('Predicted category:')  # Print predicted
    print(y_pred)  # Print predicted
    print('------------')  # Print

    # Step 11: Calculate Accuracy and Print Results
    accuracy = metrics.accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(f'Accuracy: {accuracy:.2f}')  # Print accuracy as a percentage

    # Display a detailed classification report
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Optionally, display a confusion matrix for further analysis
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_matrix)

except pymysql.MySQLError as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    # Close the database connection if it was successfully opened
    if conn:
        conn.close()
        print("MySQL connection is closed.")