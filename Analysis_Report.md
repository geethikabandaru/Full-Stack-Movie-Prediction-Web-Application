# Movie Success Analysis Report

## 1. Project Overview
This report summarizes the analysis of the IMDB Movie Dataset and the results of machine learning models trained to predict a movie's success. The project aims to identify the key factors that contribute to a movie being categorized as "Successful".

## 2. Dataset Summary
- **Source**: IMDB Movie Data (2006-2016)
- **Total Records**: 1,000 movies
- **Key Features**: Genre, Director, Rating, Votes, Revenue (Millions), Metascore, Runtime.
- **Target Variable**: `Success` (Binary: 1 for Successful, 0 for Unsuccessful)

## 3. Data Insights & Visualizations

### Genre Popularity
The dataset features a wide variety of genres. Drama, Action, and Comedy are the most frequent genres. However, high-revenue movies are often found in the **Adventure** and **Sci-Fi** categories.

### Financial Correlations
- **Rating vs Revenue**: There is a positive correlation between a movie's IMDB rating and its revenue. High-rated movies tend to generate significantly more revenue, though outliers exist.
- **Votes**: The number of user votes is a strong indicator of popular interest and is highly correlated with both Rating and Revenue.

### Key Success Drivers
Based on our analysis, the top factors driving a movie's success classification are:
1. **Votes**: Higher engagement usually translates to success.
2. **Revenue**: Financial performance is a primary component of the success metric.
3. **Rating**: Critical and audience reception play a significant role.

## 4. Model Performance
We implemented and validated three machine learning models on a 20% test split:

| Model | Accuracy (%) |
| :--- | :--- |
| **Naive Bayes** | ~85-88% |
| **Logistic Regression** | ~90-94% |
| **SVM (Support Vector Machine)** | ~92-95% |

*Note: The Logistic Regression and SVM models consistently outperformed Naive Bayes in terms of overall accuracy and robustness.*

## 5. Conclusions
- The movie's "Success" is heavily influenced by audience engagement (Votes) and financial return (Revenue).
- High Metascores and Ratings are good predictors but not absolute guarantees of commercial success.
- The **SVM model** provides the most reliable predictions for new movie entries.

---
*Report generated on: 2026-02-13*
