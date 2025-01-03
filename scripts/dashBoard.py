import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from rake_nltk import Rake
from collections import Counter
import itertools
from wordcloud import WordCloud
import plotly.graph_objects as go





# Build the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../notebooks/data/raw_analyst_ratings.csv")
data_AAPL=os.path.join(current_dir, "../notebooks/data/yfinance_data/AAPL_historical_data.csv")
data_path_AMZN=os.path.join(current_dir, "../notebooks/data/yfinance_data/AMZN_historical_data.csv")
data_path_GOOG=os.path.join(current_dir, "../notebooks/data/yfinance_data/GOOG_historical_data.csv")
data_path_META=os.path.join(current_dir, "../notebooks/data/yfinance_data/META_historical_data.csv")
data_path_MSFT=os.path.join(current_dir, "../notebooks/data/yfinance_data/MSFT_historical_data.csv")
data_path_NAVD=os.path.join(current_dir, "../notebooks/data/yfinance_data/NAVD_historical_data.csv")
data_path_TSLA=os.path.join(current_dir, "../notebooks/data/yfinance_data/TSLA_historical_data.csv")

data1=pd.read_csv(data_AAPL)

st.title("📈 **Stock Price Analysis Dashboard**")
st.write("Explore the **Closing Price Trends** of the stock over time.")

# Plot with Plotly
st.subheader("📊 **Stock Closing Price Over Time**")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data1.index,
    y=data1['Close'],
    mode='lines+markers',
    name='Close Price',
    line=dict(color='#4B9CD3', width=2),
    marker=dict(size=4, color='#1F77B4', symbol='circle')
))

# Customize layout
fig.update_layout(
    title='Stock Analysis: Closing Price Over Time',
    xaxis_title='Date',
    yaxis_title='Close Price',
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    template='plotly_white',
    hovermode='x unified',
    margin=dict(l=50, r=50, t=50, b=50)
)

# Display Plot
st.plotly_chart(fig, use_container_width=True)



data1['SMA_50'] = data1['Close'].rolling(window=50).mean()
data1['SMA_200'] = data1['Close'].rolling(window=200).mean()

# Streamlit App Title
st.title("📈 **Stock Price with Moving Averages**")
st.write("Analyze **Stock Prices** along with **50-day** and **200-day Moving Averages**.")

# Plot with Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data1.index,
    y=data1['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='#1f77b4', width=2)
))

fig.add_trace(go.Scatter(
    x=data1.index,
    y=data1['SMA_50'],
    mode='lines',
    name='50-day SMA',
    line=dict(color='#ff7f0e', width=2, dash='dash')
))

fig.add_trace(go.Scatter(
    x=data1.index,
    y=data1['SMA_200'],
    mode='lines',
    name='200-day SMA',
    line=dict(color='#2ca02c', width=2, dash='dot')
))

# Customize layout
fig.update_layout(
    title='📊 Stock Price with Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Legend',
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    template='plotly_white',
    hovermode='x unified',
    margin=dict(l=50, r=50, t=50, b=50)
)

# Display Plot in Streamlit
st.plotly_chart(fig, use_container_width=True)



def calculate_RSI(data, window=14):
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(window), 'valid') / window
    avg_loss = np.convolve(loss, np.ones(window), 'valid') / window
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi = np.concatenate([np.full(window, np.nan), rsi])
    return rsi

# Add RSI to DataFrame
data1['RSI'] = calculate_RSI(data1['Close'].values)

# Streamlit App Title
st.title("📊 **Relative Strength Index (RSI)**")
st.write("Analyze the **RSI Indicator** to identify overbought and oversold conditions in stock prices.")

# Plot RSI with Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data1.index,
    y=data1['RSI'],
    mode='lines',
    name='RSI',
    line=dict(color='#1f77b4', width=2)
))

# Add Overbought and Oversold Lines
fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought (70)')
fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold (30)')

# Customize layout
fig.update_layout(
    title='📈 Relative Strength Index (RSI)',
    xaxis_title='Date',
    yaxis_title='RSI Value',
    legend_title='Legend',
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    template='plotly_white',
    hovermode='x unified',
    margin=dict(l=50, r=50, t=50, b=50)
)

# Display Plot in Streamlit
st.plotly_chart(fig, use_container_width=True)



def calculate_EMA(data, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[window-1] = np.mean(data[:window])
    for i in range(window, len(data)):
        ema[i] = alpha * (data[i] - ema[i-1]) + ema[i-1]
    return ema

# MACD Calculation Function
def calculate_MACD(data, short_window=12, long_window=26, signal_window=9):
    ema_short = calculate_EMA(data, short_window)
    ema_long = calculate_EMA(data, long_window)
    macd = ema_short - ema_long
    signal_line = calculate_EMA(macd, signal_window)
    macd_histogram = macd - signal_line
    return macd, signal_line, macd_histogram

# Add MACD to DataFrame
data1['MACD'], data1['Signal Line'], data1['MACD Histogram'] = calculate_MACD(data1['Close'].values)

# Streamlit App Title
st.title("📊 **MACD (Moving Average Convergence Divergence)**")
st.write("""
The **MACD Indicator** helps traders understand momentum and trend direction.
It consists of:
- **MACD Line:** Difference between short-term and long-term EMAs.
- **Signal Line:** EMA of the MACD Line.
- **Histogram:** Difference between MACD and Signal Line.
""")

# Plot MACD with Plotly
fig = go.Figure()

# Add MACD Line
fig.add_trace(go.Scatter(
    x=data1.index,
    y=data1['MACD'],
    mode='lines',
    name='MACD',
    line=dict(color='blue', width=2)
))

# Add Signal Line
fig.add_trace(go.Scatter(
    x=data1.index,
    y=data1['Signal Line'],
    mode='lines',
    name='Signal Line',
    line=dict(color='red', width=2, dash='dot')
))

# Add Histogram
fig.add_trace(go.Bar(
    x=data1.index,
    y=data1['MACD Histogram'],
    name='MACD Histogram',
    marker=dict(color='gray')
))

# Customize layout
fig.update_layout(
    title='📈 Moving Average Convergence Divergence (MACD)',
    xaxis_title='Date',
    yaxis_title='Value',
    legend_title='Legend',
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    template='plotly_white',
    hovermode='x unified',
    margin=dict(l=50, r=50, t=50, b=50)
)

# Display Plot in Streamlit
st.plotly_chart(fig, use_container_width=True)








st.set_page_config(page_title="Nova Financial Solutions", page_icon=":bar-chart:", layout="wide")
data=pd.read_csv(data_path)
st.title("Nova Financial Solutionn Dataset Analysis")
st.header("📝 Dataset Column Descriptions")

# Display column descriptions
st.markdown("""
- **`headline`**: Article release headline, the title of the news article, which often includes key financial actions like **stocks hitting highs, price target changes, or company earnings**.  
- **`url`**: The **direct link** to the full news article.  
- **`publisher`**: The **author/creator** of the article.  
- **`date`**: The **publication date and time**, including **timezone information (UTC-4 timezone)**.  

""")

data["date"]=pd.to_datetime(data["date"], errors="coerce")
data["Date"]=data["date"].dt.date
data.reset_index()
st.subheader("Raw Data")
st.dataframe(data.head(10))
st.divider()
st.subheader("Dataset Dimensions")
num_rows,num_cols=data.shape
col1, col2 = st.columns(2)

with col1:
    st.info(f"**📊 Number of Rows:** `{num_rows}`")

with col2:
    st.success(f"**📊 Number of Columns:** `{num_cols}`")

# Divider for clarity
st.divider()

# Title and Subheader

# Collapsible Section for Dataset Description
with st.expander("🔍 **View Dataset Description**"):
    st.write("Below is the statistical summary of the dataset:")
    st.dataframe(data.describe())
st.subheader("🚨 Missing Values in Dataset")

# Calculate missing values
missing_values = data.isnull().sum()
print(missing_values)
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Display missing values in an attractive way
if missing_values.empty:
    st.success("✅ No missing values found in the dataset!")
else:
    st.write("The table below shows the count of missing values per column:")
    st.dataframe(missing_values.rename('Missing Values').reset_index().rename(columns={'index': 'Column'}))

    # Visualization: Bar Plot for Missing Values
    st.write("### 📊 Visualization of Missing Values")
    fig, ax = plt.subplots()
    missing_values.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('Missing Values per Column')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Number of Missing Values')
    st.pyplot(fig)
st.subheader("Descriptive Statistics")
top_publishers = data['publisher'].value_counts()
publishers_data = top_publishers.reset_index()
publishers_data.columns = ['Publisher', 'Count']
top_10_publisher=publishers_data.head(10)
colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0', '#FFEB3B', '#673AB7', '#E91E63', '#00BCD4', '#FFC107', '#795548']

st.subheader("📚 **Top 10 Publisher Count**")
fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(top_10_publisher["Publisher"], top_10_publisher["Count"], color=colors, edgecolor='black')
ax.set_xlabel("Publisher Name", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Top 10 Publisher Count", fontsize=14)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

publishers_by_url = data['url'].value_counts()
publisher_count_url=publishers_by_url.reset_index()
publisher_count_url.columns=["Url","Count"]

top_10_url=publisher_count_url.head(10)
st.subheader("📚 **Top 10 Urls Count**")
fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(top_10_url["Url"], top_10_url["Count"], color=colors, edgecolor='black')
ax.set_xlabel("Url", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Top 10  Count", fontsize=14)
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
st.pyplot(fig)

data["date_format"] = data["date"].dt.date 
publication_date = data.groupby('date_format').size().reset_index(name='count')  
publication_date["date_format"] = pd.to_datetime(publication_date["date_format"], errors="coerce") 
publication_date["Year"] = publication_date["date_format"].dt.year
publication_date["Month"] = publication_date["date_format"].dt.month
publication_date["Day"] = publication_date["date_format"].dt.day
st.subheader("📅 **Trend of Publications Per Day**")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(publication_date['date_format'], publication_date['count'], marker='o', color='#1E88E5', linestyle='-', linewidth=2)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold')
ax.set_title('📈 Trend of Publications Per Day', fontsize=14, fontweight='bold', pad=20)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)


data.set_index('date', inplace=True)
monthly_counts = data.resample('M').size()
st.subheader("📅 **Number of Articles Published Per Month**")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_counts.index, monthly_counts.values, marker='o', linestyle='-', color='#43A047', linewidth=2, label='Monthly Count')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
ax.set_title('📊 Monthly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='upper left')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)


yearly_count = publication_date.groupby('Year').size().reset_index(name='count')
st.subheader("📆 **Number of Publications per Year**")
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(yearly_count['Year'], yearly_count['count'], color='#4CAF50', edgecolor='black')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold')
ax.set_title('📊 Yearly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontsize=10, color='black')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)


monthly_count = publication_date.groupby('Month').size().reset_index(name='count')
month_labels = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]
monthly_count['Month'] = monthly_count['Month'].apply(lambda x: month_labels[x-1])
st.subheader("📅 **Number of Publications per Month**")
fig, ax = plt.subplots(figsize=(15, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(monthly_count)))
bars = ax.bar(monthly_count['Month'], monthly_count['count'], color=colors, edgecolor='black')
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold')
ax.set_title('📊 Monthly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontsize=10, color='black')

# Customize X-axis for Month Names
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.set_xticks(range(len(monthly_count)))
ax.set_xticklabels(monthly_count['Month'])
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)


daily_count = publication_date.groupby('Day').size().reset_index(name='count')

st.subheader("📅 **Number of Publications per Day**")
fig, ax = plt.subplots(figsize=(15, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(daily_count)))
bars = ax.bar(daily_count['Day'], daily_count['count'], color=colors, edgecolor='black')
ax.set_xlabel('Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold')
ax.set_title('📊 Daily Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontsize=10, color='black')

# Customize X-axis for Month Names
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.set_xticks(range(len(daily_count)))
ax.set_xticklabels(daily_count['Day'])
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)
# Resetting the index of the DataFrame
data.reset_index(inplace=True)
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Set the 'date' column as the index
data.set_index('date', inplace=True)

# Group by the hour and count the occurrences
publication_date = pd.DataFrame()
publication_date['Hour'] = data.index.hour
hourly_data = publication_date.groupby("Hour").size().reset_index(name="count")
st.subheader("⏰ **Number of Publications per Hour**")

# Create Plot
fig, ax = plt.subplots(figsize=(15, 6))
colors = plt.cm.inferno(np.linspace(0.3, 0.8, len(hourly_data)))  # Use a nice color map
bars = ax.bar(hourly_data["Hour"], hourly_data["count"], color=colors, edgecolor='black')

# Add Labels and Title
ax.set_xlabel('Hour of the Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold')
ax.set_title('📊 Hourly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)

# Add Value Labels on Bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontsize=10, color='black')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)



st.subheader("Text Analysis(Sentiment analysis & Topic Modeling)")
sentiment_data=data.copy()
SIA=SentimentIntensityAnalyzer()
sentiment_data['sentiment'] = sentiment_data['headline'].apply(lambda x: SIA.polarity_scores(text=x)['compound'])
sentiment_data["sentiment_cata"]=pd.cut(sentiment_data["sentiment"], bins=[ -1,-0.5,-0.0001,0.5,1], labels=["very negative", "negative", "nutral","postive"])
sentiment_data_count=sentiment_data["sentiment_cata"].value_counts()
sentiment_data_count_data=sentiment_data_count.reset_index()
sentiment_data_count_data.columns=["sentiment_cata","Count"]
plt.figure(figsize=(15, 6))

# Plotting the bar chart
plt.bar(sentiment_data_count_data["sentiment_cata"], sentiment_data_count_data["Count"], color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel("Sentiment Category", fontsize=12, fontweight='bold')
plt.ylabel("Count", fontsize=12, fontweight='bold')
plt.title("Sentiment Category Count", fontsize=14, fontweight='bold')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Adjust layout for better spacing
plt.tight_layout()
st.pyplot(plt)


sentiment_data["year"] = sentiment_data.index.year
sentiment_data["month"] = sentiment_data.index.month
sentiment_data["day"] = sentiment_data.index.day
sentiment_data["WeekDay"] = sentiment_data.index.weekday
sentiment_data["Hour"] = sentiment_data.index.hour
sentiment_data["year_month"] = sentiment_data.index.to_period("M")

year_sentiment_counts = sentiment_data.groupby(["year_month", "sentiment_cata"]).size().reset_index(name="count")
pivot_table_yealy = year_sentiment_counts.pivot(index="year_month", columns="sentiment_cata", values="count").fillna(0)
pivot_table_yealy.head()
pivot_table_yealy.index = pivot_table_yealy.index.astype(str)

st.sidebar.header("📊 Sentiment Analysis Dashboard")
selected_view = st.sidebar.radio("Select View", ["Yearly Trends", "Monthly Trends", "Daily Trends"])

# ---- Yearly Trends ----
if selected_view == "Yearly Trends":
    st.subheader("📆 **Yearly Sentiment Trends**")
    
    yearly_sentiment_counts = sentiment_data.groupby(["year", "sentiment_cata"]).size().reset_index(name="count")
    pivot_table_year = yearly_sentiment_counts.pivot(index="year", columns="sentiment_cata", values="count").fillna(0)
    
    # Line Plot for Trends
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(pivot_table_year.index, pivot_table_year.get("very negative", 0), color="red", label="Very Negative")
    ax.plot(pivot_table_year.index, pivot_table_year.get("negative", 0), color="green", label="Negative")
    ax.plot(pivot_table_year.index, pivot_table_year.get("nutral", 0), color="yellow", label="Neutral")
    ax.plot(pivot_table_year.index, pivot_table_year.get("postive", 0), color="brown", label="Positive")
    
    ax.set_title("Sentiment Trends Over the Years", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(title="Sentiment", loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

    # Bar Plot
    st.write("### 📊 Sentiment Counts by Year")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_table_year.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_title("Sentiment Counts by Year", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ---- Monthly Trends ----
elif selected_view == "Monthly Trends":
    st.subheader("📅 **Monthly Sentiment Trends**")
    
    monthly_sentiment_counts = sentiment_data.groupby(["month", "sentiment_cata"]).size().reset_index(name="count")
    pivot_table_month = monthly_sentiment_counts.pivot(index="month", columns="sentiment_cata", values="count").fillna(0)
    
    st.write("### 📊 Sentiment Counts by Month")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_table_month.plot(kind='bar', ax=ax, colormap='coolwarm')
    ax.set_title("Sentiment Counts by Month", fontsize=14, fontweight='bold')
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ---- Daily Trends ----
elif selected_view == "Daily Trends":
    st.subheader("📆 **Daily Sentiment Trends**")
    
    daily_sentiment_counts = sentiment_data.groupby(["day", "sentiment_cata"]).size().reset_index(name="count")
    pivot_table_day = daily_sentiment_counts.pivot(index="day", columns="sentiment_cata", values="count").fillna(0)
    
    st.write("### 📊 Sentiment Counts by Day")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_table_day.plot(kind='bar', ax=ax, colormap='plasma')
    ax.set_title("Sentiment Counts by Day", fontsize=14, fontweight='bold')
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

publisher_sentiment = sentiment_data.groupby('publisher')['sentiment'].mean().sort_values()

st.title("📅 **Monthly Sentiment Trend Analysis**")
st.write("Explore how **average sentiment scores** fluctuate over months.")

# Resample data by Month
monthly_sentiment = sentiment_data['sentiment'].resample('M').mean().dropna()

# Insights for Highest and Lowest Sentiment Months
st.subheader("📊 **Key Insights**")
if not monthly_sentiment.empty:
    highest_month = monthly_sentiment.idxmax().strftime('%B %Y')
    lowest_month = monthly_sentiment.idxmin().strftime('%B %Y')
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"😊 **Highest Sentiment Month:** {highest_month} ({monthly_sentiment.max():.2f})")
    with col2:
        st.error(f"😞 **Lowest Sentiment Month:** {lowest_month} ({monthly_sentiment.min():.2f})")
else:
    st.warning("No data available for sentiment analysis.")

# Plot Monthly Sentiment Trend
st.write("### 📈 **Average Sentiment Score by Month**")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_sentiment.index, monthly_sentiment.values, marker='o', linestyle='-', color='#4B9CD3', label='Avg Sentiment')
ax.fill_between(monthly_sentiment.index, monthly_sentiment.values, color='#D6EAF8', alpha=0.4)

ax.set_title('Average Sentiment Score by Month', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Average Sentiment Score', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

st.pyplot(fig)



sentiment_data["length_of_heading"] = sentiment_data["headline"].apply(len)

# Calculate average headline length per sentiment category
sentiment_length = sentiment_data.groupby('sentiment_cata')['length_of_heading'].mean().sort_values()

# Streamlit Title
st.title("📰 **Average Headline Length by Sentiment Category**")
st.write("Explore how the **average length of headlines** varies across different sentiment categories.")

# Plot Configuration
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=sentiment_length.index, 
    y=sentiment_length.values, 
    palette="coolwarm",
    edgecolor='black'
)

# Add labels and title
ax.set_title('📏 Average Headline Length by Sentiment Category', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Sentiment Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Headline Length', fontsize=12, fontweight='bold')
ax.bar_label(ax.containers[0], fmt='%.1f', fontsize=10, label_type='edge', padding=3)

# Style the plot
sns.despine()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)




def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply text cleaning
sentiment_data['cleaned_text'] =sentiment_data['headline'].apply(clean_text)


r=Rake()

def extract_keywords(text):
    if pd.isnull(text) or text.strip() == "":
        return []
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

sentiment_data['keywords'] = sentiment_data['cleaned_text'].apply(extract_keywords)

all_keywords = itertools.chain.from_iterable(sentiment_data['keywords'])

# Count keywords
keyword_counts = Counter(all_keywords)

# Convert to a DataFrame and sort
keyword_df = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Frequency'])
keyword_df = keyword_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)


st.title("☁️ **Word Cloud of Keywords**")
st.write("Visualize the most frequently occurring **keywords** in your dataset with this interactive Word Cloud.")

# Generate the Word Cloud
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    colormap='coolwarm',
    max_words=100,
    contour_color='steelblue',
    contour_width=1
).generate_from_frequencies(keyword_counts)

# Plot Configuration
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)
st.markdown("### 📝 **Insights:**")
st.write("""
- Larger words indicate higher frequency.
- Use this Word Cloud to quickly identify the most important keywords in your dataset.
""")


data['publication_date'] = data['date'].str.split(' ').str[0]
data['publication_time'] = data['date'].str.split(' ').str[1]
data['publication_date'] = pd.to_datetime(data['publication_date'])
data['publication_hour'] = data['publication_time'].str[:2]
data.sort_values('date', inplace=True)

# Streamlit App Title
st.title("📰 **Article Publication Trends Dashboard**")
st.write("Explore trends in article publications over different time periods.")

# Sidebar Selection
st.sidebar.header("🔄 **Select Trend View**")
trend_option = st.sidebar.radio(
    "Choose Time Frame",
    ["Daily", "Monthly", "Yearly", "Hourly"]
)

# Plot Based on Selection
st.subheader(f"📊 **Number of Articles Published Over {trend_option}**")

fig, ax = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")

if trend_option == "Daily":
    particular_day = data.groupby(data['publication_date'].dt.date).size()
    sns.lineplot(x=particular_day.index, y=particular_day.values, marker='o', color='#4B9CD3', ax=ax)
    ax.set_title('Number of Articles Published Over Time (Daily)', fontsize=14, fontweight='bold')

elif trend_option == "Monthly":
    particular_month = data.groupby(data['publication_date'].dt.month).size()
    sns.lineplot(x=particular_month.index, y=particular_month.values, marker='o', color='#FFB347', ax=ax)
    ax.set_title('Number of Articles Published Over Each Month', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

elif trend_option == "Yearly":
    particular_year = data.groupby(data['publication_date'].dt.year).size()
    sns.lineplot(x=particular_year.index, y=particular_year.values, marker='o', color='#77DD77', ax=ax)
    ax.set_title('Number of Articles Published Per Year', fontsize=14, fontweight='bold')

elif trend_option == "Hourly":
    hourly_counts = data.groupby('publication_hour').size()
    sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o', color='#F28585', ax=ax)
    ax.set_title('Number of Articles Published Per Hour', fontsize=14, fontweight='bold')

# Labels and Grid
ax.set_xlabel(trend_option, fontsize=12)
ax.set_ylabel('Number of Articles', fontsize=12)
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)
