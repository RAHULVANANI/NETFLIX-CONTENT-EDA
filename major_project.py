import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the dataset
df = pd.read_csv('netflix_content_50k.csv')

# Display first few rows
print(df.head())

# --- 🧹 Data Cleaning ---
df.dropna(inplace=True)
df = df[df['type'].isin(['Movie', 'TV Show'])]

# Encode categorical columns
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])
df['country_encoded'] = le.fit_transform(df['country'])
df['rating_encoded'] = le.fit_transform(df['rating'])

# --- 📊 Visualization 1: Type Distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df)
plt.title("Content Type Distribution")
plt.show()

# --- 📊 Visualization 2: Top 10 Countries by Content ---
top_countries = df['country'].value_counts().nlargest(10)
top_countries.plot(kind='bar', color='skyblue')
plt.title("Top 10 Countries with Most Content")
plt.ylabel("Number of Titles")
plt.show()

# --- 📊 Visualization 3: Content Added Over Years ---
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['year_added'].value_counts().sort_index().plot(kind='line', marker='o')
plt.title("Titles Added to Netflix Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.grid()
plt.show()

# --- 📊 Visualization 4: Rating Distribution ---
sns.countplot(y='rating', data=df, order=df['rating'].value_counts().index)
plt.title("Rating Distribution")
plt.show()

# 🎯 ML Task 1: Classification (Movie or TV Show)
features_cls = df[['country_encoded', 'rating_encoded', 'release_year']]
target_cls = df['type_encoded']

X_train, X_test, y_train, y_test = train_test_split(features_cls, target_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
preds_cls = clf.predict(X_test)
print("Classification Accuracy (Movie/TV Show):", accuracy_score(y_test, preds_cls))

# 🎯 ML Task 2: Regression (Predicting user_score)
features_reg = df[['country_encoded', 'rating_encoded', 'release_year']]
target_reg = df['user_score']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(features_reg, target_reg, test_size=0.2, random_state=42)
reg = RandomForestRegressor()
reg.fit(X_train_r, y_train_r)
preds_reg = reg.predict(X_test_r)
print("Regression MSE (User Score):", mean_squared_error(y_test_r, preds_reg))
