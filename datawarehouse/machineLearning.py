# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# #import seaborn as sns

# def analyze_product_categories(df):
#     """
#     Train and evaluate the decision tree classifier
#     """
#     # Prepare features
#     features = ['sales_amount', 'customer_age', 'customer_income']
#     X = df[features]
    
#     # Encode product categories
#     le = LabelEncoder()
#     y = le.fit_transform(df['product_category'])
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     # Train classifier
#     clf = DecisionTreeClassifier(max_depth=5, random_state=42)
#     clf.fit(X_train, y_train)
    
#     # Visualize decision tree
#     plt.figure(figsize=(20,10))
#     plot_tree(clf, feature_names=features, 
#              class_names=le.classes_, filled=True)
#     plt.savefig('decision_tree.png')
    
#     # Calculate feature importance
#     feature_importance = pd.DataFrame({
#         'feature': features,
#         'importance': clf.feature_importances_
#     })
    
#     return clf, le, feature_importance


# def perform_customer_segmentation(customer_summary):
#     """
#     Perform customer segmentation using K-means
#     """
#     # Prepare features for clustering
#     cluster_features = ['total_sales', 'avg_sales', 'purchase_frequency']
#     X = customer_summary[cluster_features]
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Find optimal number of clusters
#     inertias = []
#     K = range(1, 10)
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
    
#     # Perform final clustering
#     optimal_k = 4  # Adjust based on elbow curve
#     kmeans = KMeans(n_clusters=optimal_k, random_state=42)
#     customer_summary['cluster'] = kmeans.fit_predict(X_scaled)
    
#     return kmeans, scaler, customer_summary

# def analyze_clusters(customer_summary):
#     """
#     Analyze and visualize customer segments
#     """
#     # Calculate cluster statistics
#     cluster_stats = customer_summary.groupby('cluster').agg({
#         'total_sales': 'mean',
#         'avg_sales': 'mean',
#         'purchase_frequency': 'mean'
#     }).round(2)
    
#     # Visualize clusters
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(
#         data=customer_summary,
#         x='total_sales',
#         y='purchase_frequency',
#         hue='cluster',
#         palette='deep'
#     )
#     plt.title('Customer Segments')
#     plt.savefig('customer_segments.png')
    
#     return cluster_stats

# # Main execution
# def main():
#     try:
#         # # 1. Load and validate data
#         # print("Loading data...")
#         # df = load_and_validate_data('your_data.csv')
        
#         # # 2. Preprocess data
#         # print("Preprocessing data...")
#         # df_processed, customer_summary = preprocess_data(df)
        
#         # 3. Train product category classifier
#         print("Training product category classifier...")
#         clf, le, feature_importance = analyze_product_categories(df processed)
#         print("\nFeature Importance:")
#         print(feature_importance)
        
#         # 4. Perform customer segmentation
#         print("\nPerforming customer segmentation...")
#         kmeans, scaler, segmented_customers = perform_customer_segmentation(
#             customer_summary
#         )
        
#         # 5. Analyze segments
#         print("\nAnalyzing customer segments...")
#         cluster_stats = analyze_clusters(segmented_customers)
#         print("\nCluster Statistics:")
#         print(cluster_stats)
        
#         # 6. Save results
#         segmented_customers.to_csv('customer_segments.csv', index=False)
        
#         print("\nAnalysis complete! Check the generated files for visualizations.")
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()



#import pymysql
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns
# #import mysql.connector 
# from sqlalchemy import create_engine 
# #from connector import conn, cursor 

# def get_data_from_warehouse():
#     """
#     Get data from your data warehouse
#     """
#     try:
#         # Create SQLAlchemy engine for pandas
#         engine = create_engine('mysql://root:@localhost:3307/etlolapoperations')
        
#     #     # Database connection details
#     #     connection = mysql.connector.connect(
#     #         host='localhost',
#     #         user='root',
#     #         port=3307, #specify mysql port
#     #         password='',
#     #         database='etlolapoperations'
#     #   )
       
        
#         # Your SQL query to get the required columns
#         query = """
#             SELECT sales_fact.*, customer_dim.Age, customer_dim.gender, product_dim.category
#             FROM sales_fact
#             JOIN customer_dim ON sales_fact.date_id = customer_dim.customer_id
#             JOIN product_dim ON sales_fact.product_id = product_dim.product_id
#             """
        
#         # Read into DataFrame using SQLAlchemy engine
#         with engine.connect() as connection:
#         # Read directly into a pandas DataFrame
#           df = pd.read_sql(query, con=connection)
#         #connection.close()
#         engine.dispose()
        
#         return df
    
#     except Exception as e:
#         print(f"Database connection error: {str(e)}")
#         raise

# def analyze_product_categories(df):
#     """
#     Analyze product categories using decision tree classifier
#     """
#     try:
#         # Prepare features (adjust column names according to your dataset)
#         features = ['Age', 'gender']  # Add or modify features based on your data
#         X = df[features]
        
#         # Prepare target variable (product category)
#         le = LabelEncoder()
#         y = le.fit_transform(df['category'])
        
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
        
#         # Train the classifier
#         clf = DecisionTreeClassifier(max_depth=5, random_state=42)
#         clf.fit(X_train, y_train)
        
#         # Calculate feature importance
#         feature_importance = pd.DataFrame({
#             'feature': features,
#             'importance': clf.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         # Create visualization
#         plt.figure(figsize=(15, 10))
#         plot_tree(clf, feature_names=features, 
#                  class_names=le.classes_, 
#                  filled=True, 
#                  rounded=True)
#         plt.savefig('decision_tree.png')
#         plt.close()
        
#         return clf, le, feature_importance
        
#     except Exception as e:
#         print(f"Error in analyze_product_categories: {str(e)}")
#         raise

# def perform_customer_segmentation(customer_summary):
#     """
#     Perform customer segmentation using KMeans
#     """
#     try:
#         # Prepare features for clustering
#         features_for_clustering = ['total_amount', 'avg_sales', 'purchase_frequency']
#         X = customer_summary[features_for_clustering]
        
#         # Scale the features
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Perform KMeans clustering
#         kmeans = KMeans(n_clusters=4, random_state=42)
#         customer_summary['Segment'] = kmeans.fit_predict(X_scaled)
        
#         # Create visualization
#         plt.figure(figsize=(10, 6))
#         sns.scatterplot(data=customer_summary, 
#                        x='total_amount', 
#                        y='purchase_frequency', 
#                        hue='Segment', 
#                        palette='deep')
#         plt.title('Customer Segments')
#         plt.savefig('customer_segments.png')
#         plt.close()
        
#         return kmeans, scaler, customer_summary
        
#     except Exception as e:
#         print(f"Error in perform_customer_segmentation: {str(e)}")
#         raise

# def analyze_clusters(segmented_customers):
#     """
#     Analyze the characteristics of each cluster
#     """
#     try:
#         # Calculate cluster statistics
#         cluster_stats = segmented_customers.groupby('Segment').agg({
#             'total_amount': ['mean', 'min', 'max', 'count'],
#             'avg_sales': ['mean', 'min', 'max'],
#             'purchase_frequency': ['mean', 'min', 'max']
#         }).round(2)
        
#         # Create visualization
#         plt.figure(figsize=(12, 6))
#         sns.boxplot(data=segmented_customers, x='Segment', y='total_amount')
#         plt.title('Sales Distribution by Segment')
#         plt.savefig('segment_sales_distribution.png')
#         plt.close()
        
#         return cluster_stats
        
#     except Exception as e:
#         print(f"Error in analyze_clusters: {str(e)}")
#         raise

# def main():
#     try:
#         # Get data from warehouse
#         print("Fetching data from warehouse...")
#         df = get_data_from_warehouse()

#         # Create customer summary
#         print("Creating customer summary...")
#         customer_summary = df.groupby('customer_id').agg({
#             'total_amount': ['sum', 'mean', 'count']
#         }).reset_index()
#         customer_summary.columns = ['customer_id', 'total_amount', 'avg_sales', 'purchase_frequency']

#         # Train product category classifier
#         print("Training product category classifier...")
#         clf, le, feature_importance = analyze_product_categories(df)
#         print("\nFeature Importance:")
#         print(feature_importance)

#         # Perform customer segmentation
#         print("\nPerforming customer segmentation...")
#         kmeans, scaler, segmented_customers = perform_customer_segmentation(customer_summary)

#         # Analyze segments
#         print("\nAnalyzing customer segments...")
#         cluster_stats = analyze_clusters(segmented_customers)
#         print("\nCluster Statistics:")
#         print(cluster_stats)

#         # Save results
#         segmented_customers.to_csv('customer_segments.csv', index=False)
#         print("\nAnalysis complete! Check the generated files for visualizations.")

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()




