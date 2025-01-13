# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_and_prepare_data():
#     """
#     Load and prepare data from sales fact and customer dimension tables
#     """
#     # Example SQL query to get the data (you'll need to adapt this to your database connection)
#     query = """
#     SELECT 
#         s.customer_id,
#         c.age,
#         c.gender,
#         COUNT(DISTINCT s.sale_id) as number_of_transactions,
#         SUM(s.total_amount) as total_spent,
#         AVG(s.total_amount) as avg_transaction_value,
#         SUM(s.quantity) as total_items_purchased
#     FROM sales_fact s
#     JOIN customer_dim c ON s.customer_id = c.customer_id
#     GROUP BY s.customer_id, c.age, c.gender
#     """
    
#     # For demonstration, creating sample data (replace this with your actual data loading)
#     np.random.seed(42)
#     n_customers = 1000
    
#     df = pd.DataFrame({
#         'customer_id': range(n_customers),
#         'age': np.random.randint(18, 80, n_customers),
#         'gender': np.random.choice(['M', 'F'], n_customers),
#         'number_of_transactions': np.random.randint(1, 50, n_customers),
#         'total_spent': np.random.uniform(100, 10000, n_customers),
#         'avg_transaction_value': np.random.uniform(50, 500, n_customers),
#         'total_items_purchased': np.random.randint(1, 200, n_customers)
#     })
    
#     return df

# def preprocess_data(df):
#     """
#     Preprocess the data for clustering
#     """
#     # Convert gender to numeric
#     df['gender_numeric'] = df['gender'].map({'M': 0, 'F': 1})
    
#     # Select features for clustering
#     features = ['age', 'gender_numeric', 'number_of_transactions', 
#                'total_spent', 'avg_transaction_value', 'total_items_purchased']
#     X = df[features]
    
#     # Scale the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     return X_scaled, scaler, features

# def determine_optimal_clusters(X_scaled):
#     """
#     Use elbow method to find optimal number of clusters
#     """
#     inertias = []
#     K = range(1, 11)
    
#     for k in K:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(X_scaled)
#         inertias.append(kmeans.inertia_)
    
#     # Plot elbow curve
#     plt.figure(figsize=(10, 6))
#     plt.plot(K, inertias, 'bx-')
#     plt.xlabel('k')
#     plt.ylabel('Inertia')
#     plt.title('Elbow Method For Optimal k')
#     plt.savefig('elbow_curve.png')
#     plt.close()
    
#     return 5  # You can adjust this based on the elbow curve

# def perform_clustering(X_scaled, n_clusters):
#     """
#     Perform K-means clustering
#     """
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(X_scaled)
#     return kmeans, clusters

# def analyze_segments(df, clusters, features):
#     """
#     Analyze the characteristics of each segment
#     """
#     # Add cluster assignments to the dataframe
#     df['Segment'] = clusters
    
#     # Calculate segment profiles
#     segment_profiles = df.groupby('Segment').agg({
#         'age': ['mean', 'min', 'max'],
#         'gender': lambda x: x.value_counts().index[0],  # most common gender
#         'number_of_transactions': 'mean',
#         'total_spent': 'mean',
#         'avg_transaction_value': 'mean',
#         'total_items_purchased': 'mean'
#     }).round(2)
    
#     # Visualize age distribution by segment
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(x='Segment', y='age', data=df)
#     plt.title('Age Distribution by Segment')
#     plt.savefig('age_segments.png')
#     plt.close()
    
#     # Visualize spending patterns
#     plt.figure(figsize=(12, 6))
#     sns.scatterplot(data=df, x='total_spent', y='avg_transaction_value', 
#                    hue='Segment', palette='deep')
#     plt.title('Spending Patterns by Segment')
#     plt.savefig('spending_patterns.png')
#     plt.close()
    
#     # Gender distribution by segment
#     gender_dist = df.groupby(['Segment', 'gender']).size().unstack()
#     gender_dist_pct = gender_dist.div(gender_dist.sum(axis=1), axis=0) * 100
    
#     plt.figure(figsize=(10, 6))
#     gender_dist_pct.plot(kind='bar', stacked=True)
#     plt.title('Gender Distribution by Segment')
#     plt.xlabel('Segment')
#     plt.ylabel('Percentage')
#     plt.legend(title='Gender')
#     plt.tight_layout()
#     plt.savefig('gender_distribution.png')
#     plt.close()
    
#     return segment_profiles, gender_dist_pct

# def main():
#     # 1. Load and prepare data
#     print("Loading data...")
#     df = load_and_prepare_data()
    
#     # 2. Preprocess data
#     print("Preprocessing data...")
#     X_scaled, scaler, features = preprocess_data(df)
    
#     # 3. Determine optimal number of clusters
#     print("Determining optimal number of clusters...")
#     n_clusters = determine_optimal_clusters(X_scaled)
    
#     # 4. Perform clustering
#     print("Performing clustering...")
#     kmeans, clusters = perform_clustering(X_scaled, n_clusters)
    
#     # 5. Analyze segments
#     print("Analyzing segments...")
#     segment_profiles, gender_dist = analyze_segments(df, clusters, features)
    
#     # 6. Print results
#     print("\nSegment Profiles:")
#     print(segment_profiles)
#     print("\nGender Distribution by Segment (%):")
#     print(gender_dist)
    
#     # 7. Save segmented customer data
#     df.to_csv('customer_segments_demographics.csv', index=False)
    
#     print("\nAnalysis complete! Check the generated files for visualizations.")
    
#     return df, kmeans, scaler

# if __name__ == "__main__":
#     df, kmeans_model, scaler = main()


import pymysql
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')  # Or 'Qt5Agg', depending on your environment
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Database connection details
db_config = {
    'host': 'localhost',  # Host where the database is running
    'user': 'root',    # Your database username
    'port':3307,
    'password': '',    # Your database password
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
    SELECT sales_fact.customer_id, sales_fact.product_id, sales_fact.quantity, sales_fact.total_amount, date_dim.date
    FROM sales_fact
    JOIN Date_Dim ON sales_fact.date_id = date_dim.date_id
    """
   
    # Load the data into a pandas DataFrame
    data = pd.read_sql(query, conn)
   
    # Step 3: Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())

    # Step 4: Prepare the data for clustering
    # For clustering, we need customer-level data, so we'll aggregate the data by customer_id
    customer_data = data.groupby('customer_id').agg({
        'quantity': 'sum',
        'total_amount': 'sum'
    }).reset_index()

    print("Customer data aggregated by customer_id:")
    print(customer_data.head())

    # Step 5: Data Preprocessing
    # Normalize the data (scaling) to bring the features on the same scale
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data[['quantity', 'total_amount']])

    # Step 6: Elbow Method to determine the optimal number of clusters
    # We will test a range of cluster numbers to find the optimal one
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(customer_data_scaled)
        wcss.append(kmeans.inertia_)
   
    # Plotting the elbow graph to find the optimal number of clusters
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-cluster sum of squares)')
    plt.show()

    # Based on the elbow plot, choose the optimal number of clusters (let's assume it's 4 here)
    optimal_clusters = 4

    # Step 7: Apply K-Means Clustering
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

    # Step 8: Display the clustering results
    print("\nCustomer data with assigned clusters:")
    print(customer_data.head())

    # Step 9: Visualize the clustering (optional)
    plt.scatter(customer_data[customer_data['Cluster'] == 0]['quantity'], customer_data[customer_data['Cluster'] == 0]['total_amount'], s=100, c='red', label='Cluster 1')
    plt.scatter(customer_data[customer_data['Cluster'] == 1]['quantity'], customer_data[customer_data['Cluster'] == 1]['total_amount'], s=100, c='blue', label='Cluster 2')
    plt.scatter(customer_data[customer_data['Cluster'] == 2]['quantity'], customer_data[customer_data['Cluster'] == 2]['total_amount'], s=100, c='green', label='Cluster 3')
    plt.scatter(customer_data[customer_data['Cluster'] == 3]['quantity'], customer_data[customer_data['Cluster'] == 3]['total_amount'], s=100, c='yellow', label='Cluster 4')
   
    # Plot the centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
   
    plt.title('Customer Clusters based on Purchasing Behavior')
    plt.xlabel('Quantity Purchased')
    plt.ylabel('Total Amount Spent')
    plt.legend()
    plt.show()

except pymysql.MySQLError as e:
    print(f"Error while connecting to MySQL: {e}")


