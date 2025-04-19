import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load the 'heart.csv' dataset
heart_df = pd.read_csv('heart.csv')

# Display the structure and a brief summary of the dataset
heart_df.info()
heart_df.head()


# Correlation Analysis
plt.figure(figsize=(12, 8))
correlation_matrix = heart_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Matrix')
plt.show()

# Preparing data for classification
X = heart_df.drop(columns=['target'])  # Features
y = heart_df['target']                # Target

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Building a basic classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)



# Scaling the features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

# Performing K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
heart_df['cluster'] = kmeans.fit_predict(scaled_features)
kmeans_centers = kmeans.cluster_centers_

# Visualizing cluster distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], 
                hue=heart_df['cluster'], palette='viridis', s=60)
plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], c='red', s=200, alpha=0.8, marker='X', label='Centroids')
plt.title('Clusters of Heart Disease Risk')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.show()


# Visualizing cluster distribution with respect to two specific features, 'age' and 'chol'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=heart_df['age'], y=heart_df['chol'], hue=heart_df['cluster'], palette='coolwarm', s=60)
plt.title('Clusters based on Age and Cholesterol Levels')
plt.xlabel('Age')
plt.ylabel('Cholesterol Levels')
plt.legend(title='Cluster')
plt.show()

# Analyzing clustering results and their significance
cluster_analysis = heart_df.groupby('cluster').mean()

# Overall statistics of clusters
cluster_stats = heart_df['cluster'].value_counts()

# Structure for the detailed analysis output
analysis_output = {
    'Cluster Statistics': cluster_stats,
    'Mean Feature Values by Cluster': cluster_analysis
}


# 1. Visualizing Clusters by 'thal' and 'oldpeak'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=heart_df['thal'], y=heart_df['oldpeak'], hue=heart_df['cluster'], palette='viridis', s=60)
plt.title('Clusters based on Thalassemia and Oldpeak')
plt.xlabel('Thalassemia')
plt.ylabel('Oldpeak')
plt.legend(title='Cluster')
plt.show()

# 2. Boxplot of Cholesterol Levels across Clusters
plt.figure(figsize=(10, 6))
sns.boxplot(x=heart_df['cluster'], y=heart_df['chol'], palette='Set2')
plt.title('Cholesterol Levels across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Cholesterol Levels')
plt.show()

# 3. Distribution of Age within Each Cluster
plt.figure(figsize=(10, 6))
sns.histplot(data=heart_df, x='age', hue='cluster', multiple='stack', palette='husl', kde=True)
plt.title('Age Distribution across Clusters')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Cluster')
plt.show()

# 4. Pairplot for Key Features Highlighting Clusters
key_features = ['age', 'chol', 'thalach', 'oldpeak']
sns.pairplot(heart_df, vars=key_features, hue='cluster', palette='cool')
plt.suptitle('Pairplot of Key Features Highlighting Clusters', y=1.02)
plt.show()

# 5. Barplot showing the distribution of 'sex' across clusters
plt.figure(figsize=(8, 5))
sns.countplot(x='cluster', hue='sex', data=heart_df, palette='Pastel1')
plt.title('Gender Distribution within Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Sex', labels=['Female', 'Male'])
plt.show()

# 6. Scatterplot with Regression Line for Age vs Cholesterol across Clusters
plt.figure(figsize=(10, 6))
sns.lmplot(data=heart_df, x='age', y='chol', hue='cluster', palette='coolwarm', aspect=1.5, markers=['o', 's', 'D'])
plt.title('Age vs Cholesterol with Regression Lines for Each Cluster')
plt.xlabel('Age')
plt.ylabel('Cholesterol Levels')
plt.show()