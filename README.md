# 🎬 Netflix Content Analysis & Prediction

An exploratory data analysis (EDA) and machine learning project that analyzes Netflix-style content data to uncover trends and predict content type and user ratings.

## 1. Introduction

**What is EDA?**
Exploratory Data Analysis is the process of examining a dataset to summarize its main characteristics, uncover patterns, spot anomalies, and test assumptions — often using visual methods.

**Why Netflix?**
Netflix is one of the world's most popular streaming platforms, with a diverse content library and rich user engagement trends, making it an ideal case study for content and audience analysis.

## 2. Project Objectives

- Analyze Netflix content trends
- Visualize key insights from the dataset
- Predict content type (**Movie** / **TV Show**)
- Predict **user ratings**

## 3. Dataset Overview

- **Size:** 50,000 rows × 15 columns
- **Nature:** Synthetic but realistic Netflix-style data
- **Key Columns:** `title`, `type`, `country`, `rating`, `duration`, `user_score`, `date_added`, and more

## 4. Tools & Technologies

- **Language:** Python
- **Libraries:** Pandas, Matplotlib, Seaborn, Scikit-learn
- **Optional:** WordCloud, Streamlit

## 5. Data Cleaning & Preprocessing

- Handling missing data
- Label encoding for categorical features
- Date parsing (e.g., `date_added`)

## 6. Visualizations

- Content type distribution (Movie vs TV)
- Top 10 countries by content count
- Year-wise content addition
- Rating distribution

## 7. Machine Learning Models

| Task | Model | Target |
|---|---|---|
| Classification | `RandomForestClassifier` | Predict `type` (Movie/TV) |
| Regression | `RandomForestRegressor` | Predict `user_score` |

## 8. Results

- **Classification Accuracy** — model performance on predicting content type
- **Regression MSE** — model performance on predicting user score
- **Feature Importance** — key drivers behind predictions (optional)

## 9. Conclusion

- Uncovered key content and viewership trends across the dataset
- Machine learning models performed well for both classification and regression tasks
- Insights are useful for **content planning** and **audience targeting**

## Getting Started

```bash
git clone https://github.com/RAHULVANANI/<repo-name>.git
cd <repo-name>
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud
python netflix-content-eda.py
```

## Author

**Rahul Vanani**
- [LinkedIn](https://www.linkedin.com/in/rahul-vanani-72ba60311/) · [GitHub](https://github.com/RAHULVANANI)


