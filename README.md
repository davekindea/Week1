# 📊 Financial News Sentiment and Stock Movement Analysis

## 🚀 Project Overview

This project focuses on analyzing a large corpus of financial news data to uncover correlations between news sentiment and stock market movements. The challenge combines Data Engineering (DE), Financial Analytics (FA), and Machine Learning Engineering (MLE) to provide actionable insights and predictive strategies for stock market trends.

## 🎯 Objectives

- Perform Sentiment Analysis on financial news headlines.
- Analyze the correlation between sentiment scores and stock price movements.
- Implement technical analysis indicators using TA-Lib.
- Utilize PyNance for advanced financial metrics.
- Visualize findings through interactive dashboards.
# 📊 Dataset Overview

The dataset consists of two main components:

### 1. Financial News Data:
- **Headline:** Text content of the financial news articles
- **Sentiment:** Sentiment score (positive, neutral, negative)
- **Publisher:** The source of the news article

### 2. Stock Market Data:
- **Date:** Stock trading date or news publication date
- **Stock Symbol:** Identifies the stock
- **Open:** Opening price for the stock on that day
- **High:** Highest price of the stock on that day
- **Low:** Lowest price of the stock on that day
- **Close:** Closing price of the stock on that day
- **Volume:** Number of shares traded
# 🛠️ Tools and Technologies

- **Python:** Primary programming language
- **Pandas & NumPy:** Data manipulation and analysis
- **TextBlob & VADER:** Sentiment analysis tools
- **Matplotlib, Seaborn, Plotly:** Data visualization libraries
- **TA-Lib:** Technical analysis for stock prices
- **Scikit-learn:** Machine learning models for predictive analysis
- **Jupyter Notebook:** Interactive analysis environment
## ⚙️ Methodology

### 1️⃣ Data Cleaning and Preprocessing
- Handle missing data
- Convert timestamps to a unified format
- Perform data normalization

### 2️⃣ Sentiment Analysis
- Use NLP techniques to analyze financial headlines
- Assign sentiment scores using libraries like TextBlob or NLTK

### 3️⃣ Correlation Analysis
- Align news data with stock data by matching publication dates and stock trading days
- Calculate daily stock returns
- Perform correlation tests between sentiment scores and stock price fluctuations

### 4️⃣ Exploratory Data Analysis (EDA)
- Investigate distributions of sentiment scores and stock price changes
- Visualize the impact of sentiment on stock performance
- Identify patterns in news cycles and stock movements

## 📚 Tasks Breakdown

### 📝 1️⃣ Task 1: Exploratory Data Analysis (EDA)

#### Descriptive Statistics:
- **Analyze headline lengths.**
- **Count publication frequencies.**
- **Identify active publishers.**

#### Text Analysis:
- **Perform sentiment analysis.**
- **Identify key topics.**

#### Time Series Analysis:
- **Examine publication trends.**
- **Analyze publication timing.**

#### Publisher Analysis:
- **Identify top publishers.**
- **Analyze reporting patterns.**

## 🗝️ KPIs:

- Proactivity in self-learning.
- Completeness of EDA insights.

## ⚙️ 2️⃣ Task 2: Quantitative Analysis with PyNance and TA-Lib

- **Data Preparation:** Load stock price data into a Pandas DataFrame.
- **Technical Indicators:** Calculate RSI, MACD, and Moving Averages using TA-Lib.
- **Financial Metrics:** Leverage PyNance for advanced analytics.
- **Data Visualization:** Create meaningful visualizations.

## 🗝️ KPIs:

- Accuracy of technical indicators.
- Completeness of analysis.



## 📊 3️⃣ Task 3: Correlation Between News and Stock Movements

- **Date Alignment:** Normalize dates between news and stock datasets.
- **Sentiment Analysis:** Assign sentiment scores to headlines.
- **Stock Movement Analysis:** Calculate daily returns.
- **Correlation Analysis:** Determine statistical correlation between sentiment scores and stock returns.

#### 🗝️ KPIs:
- Sentiment analysis accuracy.
- Correlation strength.



## 🛠️ Tools & Technologies

- **Python Libraries:** Pandas, NumPy, TA-Lib, PyNance, NLTK, TextBlob
- **Visualization Tools:** Matplotlib, Plotly
- **Version Control:** Git & GitHub
- **Environment:** Jupyter Notebook, Streamlit

## 📈 Key Deliverables

- Exploratory Data Analysis Report
- Sentiment Analysis Model and Results
- Correlation Analysis Findings
- Interactive Dashboards
- Final Report with Investment Recommendations

## 💼 How to Run the Project

### Clone the repository:

```bash
git clone https://github.com/davekindea/Week1.git
```

### Navigate to the project folder:
```bash
cd scripts
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run the dashboard:
```bash
streamlit run dashBoard.py
```
## 🚀 Live Dashboard

Explore the interactive dashboard for real-time insights and visualizations here:  
👉 [View Live Dashboard](https://novafinancial.streamlit.app/)
