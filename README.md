# Restaurant Review Sentiment Analysis

## Overview
[Final Output - Streamlit App](https://cis509xfrogs.streamlit.app/)

This project proposes using textual analysis of customer reviews to uncover actionable insights for restaurants. By categorizing reviews into key dimensions such as **food quality, service, and ambiance**, restaurants can pinpoint areas for improvement, optimize operations, and enhance customer satisfaction. 

Using **sentiment analysis** and **word-matching** with datasets like Yelp, this approach generates **sub-rankings and visualizations** to track performance over time. These insights empower restaurants to drive profits, foster customer loyalty, and enable data-driven growth.

## Problem Statement
Many restaurants rely on overall review scores, missing valuable insights within customer feedback. By analyzing and categorizing reviews into specific areas like **customer satisfaction, food quality, and ambiance**, actionable insights can be extracted. This helps restaurants:
- Identify strengths and weaknesses
- Enhance customer experiences
- Improve operational efficiency
- Boost business performance

## Proposed Solution
We utilize **sentiment analysis** and **word-matching techniques** to categorize review content into meaningful subcategories. For example:
- **"waiter"** → categorized under **service**
- **"cheeseburger"** → categorized under **food**

This categorization allows for:
- **Sub-rankings and visualizations** to track performance trends over time
- **Targeted optimizations** to improve customer experiences
- **Efficient resource allocation** for long-term business success

## Data Sources

### **Primary Dataset**
- **Yelp Dataset:** [Yelp Open Dataset](https://www.yelp.com/dataset)

### **External Data Sources**
- **SQL Table Creation Script:** [GitHub Repository](https://github.com/veeragandhi/YelpDatasetSQLServer/blob/master/1_ImportYelpDataset.sql)  
  _Description:_ This repository enabled us to efficiently import JSON files into a **SQL Server database**, allowing for **faster attribute selection** compared to loading the large file into Python.

## Filtering & Selection Criteria
Using the above **SQL table creation repository**, we hosted a **local database** to query and select the specific cities and timeframes relevant to our study. Our selection criteria:

- **Cities:** Restaurants in **New Orleans** or **Nashville**
- **Timeframe:** Reviews **before 2020-01-01**
- **Business Status:** Restaurants that are **still open**

### SQL Query:
```sql
SELECT *
FROM dbo.Review r
JOIN dbo.Business b 
ON r.business_id = b.business_id
WHERE b.business_city IN ('New Orleans', 'Nashville')
AND r.review_date >= CONVERT(datetime, '2020-01-01')
AND b.is_open = 1
ORDER BY review_count DESC;
```

To refine our dataset further, we applied the following criteria:
- Non-food establishments removed
- Fast-food establishments removed
- Restaurants with more than 5 locations removed
- Restaurants with fewer than 110 reviews removed

## Aspect-Based Sentiment Analysis

To categorize and analyze restaurant reviews effectively, we implemented a **two-step NLP pipeline**:

### 1. Aspect Classification
We used **zero-shot classification** with the `facebook/bart-large-mnli` model to classify each review into the following predefined aspects:

- **Food Quality**
- **Service**
- **Ambiance**
- **Wait Time**
- **Price/Value**
- **Menu Variety**
- **Cleanliness**

If an aspect’s confidence score was **below 0.4**, it was considered **not present** in the review, and its sentiment was marked as **NaN** to prevent misclassification.

### 2. Sentiment Analysis
For sentiment evaluation, we applied **"sentiment-analysis"** using the `distilbert-base-uncased-finetuned-sst-2-english` model:

- A **1-5 rating** was assigned for each aspect.
- Only aspects with a confidence score **above 0.4** were included in the final sentiment analysis.

### Why This Approach?
This method ensures:
**Accurate categorization** of review content into meaningful aspects.  
**Reliable sentiment scoring**, filtering out weak predictions.  
**Actionable insights** by linking specific aspects to quantified sentiment ratings.  

By combining **zero-shot classification** with **fine-tuned sentiment analysis**, we enable **granular tracking** of restaurant performance over time.

## Conclusion

This project demonstrates how **textual analysis of customer reviews** can uncover actionable insights for restaurants. By leveraging **zero-shot classification** and **sentiment analysis**, we effectively categorized reviews into key aspects such as **Food Quality, Service, Ambiance, Wait Time, Price/Value, Menu Variety, and Cleanliness**.  

Our approach ensures:  
**Granular insights** beyond overall ratings by analyzing specific aspects of customer experience.  
**Accurate sentiment scoring**, filtering out weak predictions to maintain reliability.  
**Data-driven decision-making**, allowing restaurants to pinpoint strengths and address weaknesses.  

By tracking aspect-based sentiment over time, restaurants can **enhance customer satisfaction, optimize operations, and increase profitability**. This system enables **targeted improvements**, fostering **customer loyalty and long-term success**.  

Explore the insights in our **[Streamlit App](https://cis509xfrogs.streamlit.app/)**. 🚀  