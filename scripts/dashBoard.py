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
import plotly.express as px
import plotly.figure_factory as ff
from PIL import Image
import matplotlib
import matplotlib.font_manager



# Build the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../notebooks/data/raw_analyst_ratings.csv")
data_path_keywords = os.path.join(current_dir, "../notebooks/data/keyword_df.csv")
data_path_AAPL=os.path.join(current_dir, "../notebooks/data/yfinance_data/AAPL_historical_data.csv")
data_path_AMZN=os.path.join(current_dir, "../notebooks/data/yfinance_data/AMZN_historical_data.csv")
data_path_GOOG=os.path.join(current_dir, "../notebooks/data/yfinance_data/GOOG_historical_data.csv")
data_path_META=os.path.join(current_dir, "../notebooks/data/yfinance_data/META_historical_data.csv")
data_path_MSFT=os.path.join(current_dir, "../notebooks/data/yfinance_data/MSFT_historical_data.csv")
data_path_NAVD=os.path.join(current_dir, "../notebooks/data/yfinance_data/NAVD_historical_data.csv")
data_path_TSLA=os.path.join(current_dir, "../notebooks/data/yfinance_data/TSLA_historical_data.csv")
sn_data_path=os.path.join(current_dir, "../notebooks/data/sentiment_data.csv")

st.set_page_config(page_title="Nova Financial Solutions", page_icon=":bar-chart:", layout="wide", initial_sidebar_state="expanded" )









data=pd.read_csv(data_path)
st.title("Nova Financial Solutionn Dataset Analysis")

image_path = os.path.join(current_dir,'../notebooks/data/predictive.jpg')

# Open image using PIL
image = Image.open(image_path)

# Display image using streamlit
st.image(image)



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


st.divider()


with st.expander("🔍 **View Dataset Description**"):
    st.write("Below is the statistical summary of the dataset:")
    st.dataframe(data.describe())

st.divider()
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
fig.subplots_adjust(top=0.9, bottom=0.1)

st.pyplot(fig)
st.divider()
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
fig.subplots_adjust(top=0.9, bottom=0.1)

st.pyplot(fig)
st.divider()
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
fig.subplots_adjust(top=0.9, bottom=0.1)

st.pyplot(fig)

st.divider()
data.set_index('date', inplace=True)
monthly_counts = data.resample('ME').size()
st.subheader("📅 **Number of Articles Published Per Month**")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_counts.index, monthly_counts.values, marker='o', linestyle='-', color='#43A047', linewidth=2, label='Monthly Count')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
ax.set_title('📊 Monthly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='upper left')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
fig.subplots_adjust(top=0.9, bottom=0.1)
 
st.pyplot(fig)

st.divider()
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
fig.subplots_adjust(top=0.9, bottom=0.1)
 
st.pyplot(fig)

st.divider()
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
fig.subplots_adjust(top=0.9, bottom=0.1)

st.pyplot(fig)

st.divider()
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


ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.set_xticks(range(len(daily_count)))
ax.set_xticklabels(daily_count['Day'])
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
fig.subplots_adjust(top=0.9, bottom=0.1)

st.pyplot(fig)
st.divider()
data.reset_index(inplace=True)
data['date'] = pd.to_datetime(data['date'], errors='coerce')


data.set_index('date', inplace=True)


publication_date = pd.DataFrame()
publication_date['Hour'] = data.index.hour
hourly_data = publication_date.groupby("Hour").size().reset_index(name="count")
st.subheader("⏰ **Number of Publications per Hour**")


fig, ax = plt.subplots(figsize=(15, 6))
colors = plt.cm.inferno(np.linspace(0.3, 0.8, len(hourly_data)))  # Use a nice color map
bars = ax.bar(hourly_data["Hour"], hourly_data["count"], color=colors, edgecolor='black')


ax.set_xlabel('Hour of the Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold')
ax.set_title('📊 Hourly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)


for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontsize=10, color='black')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
fig.subplots_adjust(top=0.9, bottom=0.1)
 
st.pyplot(fig)

st.divider()

st.subheader("Text Analysis(Sentiment analysis & Topic Modeling)")
sentiment_data=pd.read_csv(sn_data_path)
sentiment_data_count=sentiment_data["sentiment_cata"].value_counts()
sentiment_data_count_data=sentiment_data_count.reset_index()
sentiment_data_count_data.columns=["sentiment_cata","Count"]

plt.figure(figsize=(15, 6))


plt.bar(sentiment_data_count_data["sentiment_cata"], sentiment_data_count_data["Count"], color='skyblue', edgecolor='black')


plt.xlabel("Sentiment Category", fontsize=12, fontweight='bold')
plt.ylabel("Count", fontsize=12, fontweight='bold')
plt.title("Sentiment Category Count", fontsize=14, fontweight='bold')

plt.xticks(rotation=45, ha="right")


fig.subplots_adjust(top=0.9, bottom=0.1)

st.pyplot(plt)
st.dataframe(sentiment_data.head(10))
st.divider()

sentiment_data["date"]=pd.to_datetime(sentiment_data["date"], errors="coerce")
sentiment_data.set_index('date', inplace=True)
sentiment_data["year"] = sentiment_data.index.year
sentiment_data["month"] = sentiment_data.index.month
sentiment_data["day"] = sentiment_data.index.day
sentiment_data["WeekDay"] = sentiment_data.index.weekday
sentiment_data["Hour"] = sentiment_data.index.hour
sentiment_data["year_month"] = sentiment_data.index.tz_localize(None).to_period("M")

year_sentiment_counts = sentiment_data.groupby(["year_month", "sentiment_cata"]).size().reset_index(name="count")
pivot_table_yealy = year_sentiment_counts.pivot(index="year_month", columns="sentiment_cata", values="count").fillna(0)
pivot_table_yealy.head()
pivot_table_yealy.index = pivot_table_yealy.index.astype(str)
dataset_options = [
    'AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'TSLA', 'NAVD'
]
st.title("📊 Sentiment Analysis Dashboard")
selected_view = st.selectbox("Select View", ["Yearly Trends", "Monthly Trends", "Daily Trends"])


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
    fig.subplots_adjust(top=0.9, bottom=0.1)

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
    fig.subplots_adjust(top=0.9, bottom=0.1)
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
    fig.subplots_adjust(top=0.9, bottom=0.1)

    st.pyplot(fig)
st.divider()
publisher_sentiment = sentiment_data.groupby('publisher')['sentiment'].mean().sort_values()

st.title("📅 **Monthly Sentiment Trend Analysis**")
st.write("Explore how **average sentiment scores** fluctuate over months.")

monthly_sentiment = sentiment_data['sentiment'].resample('ME').mean().dropna()

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
fig.subplots_adjust(top=0.9, bottom=0.1)


st.pyplot(fig)

st.divider()

sentiment_data["length_of_heading"] = sentiment_data["headline"].apply(len)
sentiment_length = sentiment_data.groupby('sentiment_cata')['length_of_heading'].mean().sort_values()

st.title("📰 **Average Headline Length by Sentiment Category**")
st.write("Explore how the **average length of headlines** varies across different sentiment categories.")


fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=sentiment_length.index, 
    y=sentiment_length.values, 
    palette="coolwarm",
     hue='sentiment_cata'
    edgecolor='black'
)


ax.set_title('📏 Average Headline Length by Sentiment Category', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Sentiment Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Headline Length', fontsize=12, fontweight='bold')
ax.bar_label(ax.containers[0], fmt='%.1f', fontsize=10, label_type='edge', padding=3)


sns.despine()
plt.xticks(rotation=45, ha='right')
fig.subplots_adjust(top=0.9, bottom=0.1)

st.pyplot(fig)
st.divider()

keyword_df =pd.read_csv(data_path_keywords)
keyword_counts = keyword_df['Keyword'].value_counts()


st.title("☁️ **Word Cloud of Keywords**")
st.write("Visualize the most frequently occurring **keywords** in your dataset with this interactive Word Cloud.")


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
st.divider()
data.reset_index(inplace=True)
data['date'] = pd.to_datetime(data['date'], errors='coerce')


data['date_str'] = data['date'].astype(str)
data['publication_date'] = data['date_str'].str.split(' ').str[0]
data['publication_date'] = data['date_str'].str.split(' ').str[0]
data['publication_time'] = data['date_str'].str.split(' ').str[1]
data['publication_date'] = pd.to_datetime(data['publication_date'])
data['publication_hour'] = data['publication_time'].str[:2]
data.sort_values('date', inplace=True)
data.set_index('date', inplace=True)


st.title("📰 **Article Publication Trends Dashboard**")
st.write("Explore trends in article publications over different time periods.")


st.title("🔄 **Select Trend View**")
dataset_options=["Daily", "Monthly", "Yearly", "Hourly"]


trend_option = st.selectbox("Choose Time Frame", dataset_options)

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


ax.set_xlabel(trend_option, fontsize=12)
ax.set_ylabel('Number of Articles', fontsize=12)
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)
st.divider()


dataset_options = [
    'AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'TSLA', 'NAVD'
]


selected_option = st.selectbox("Select the dataset to load:", dataset_options)

if selected_option == 'AAPL':
    data1 = pd.read_csv(data_path_AAPL)
elif selected_option == 'AMZN':
    data1 = pd.read_csv(data_path_AMZN)
elif selected_option == 'GOOG':
    data1 = pd.read_csv(data_path_GOOG)
elif selected_option == 'META':
    data1 = pd.read_csv(data_path_META)
elif selected_option == 'MSFT':
    data1 = pd.read_csv(data_path_MSFT)
elif selected_option == 'TSLA':
    data1 = pd.read_csv(data_path_TSLA)
elif selected_option == 'NAVD':
    data1 = pd.read_csv(data_path_NAVD)

st.write(f"Data for {selected_option}:")
st.write("Explore the **Closing Price Trends** of the stock over time.")

st.subheader("📊 **Stock Closing Price Over Time**")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data1["Date"],
    y=data1['Close'],
    mode='lines+markers',
    name='Close Price',
    line=dict(color='#4B9CD3', width=2),
    marker=dict(size=4, color='#1F77B4', symbol='circle')
))


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


st.plotly_chart(fig, use_container_width=True)

st.divider()

data1['SMA_50'] = data1['Close'].rolling(window=50).mean()
data1['SMA_200'] = data1['Close'].rolling(window=200).mean()

st.title("📈 **Stock Price with Moving Averages**")
st.write("Analyze **Stock Prices** along with **50-day** and **200-day Moving Averages**.")


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data1["Date"],
    y=data1['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='#1f77b4', width=2)
))

fig.add_trace(go.Scatter(
    x=data1["Date"],
    y=data1['SMA_50'],
    mode='lines',
    name='50-day SMA',
    line=dict(color='#ff7f0e', width=2, dash='dash')
))

fig.add_trace(go.Scatter(
    x=data1["Date"],
    y=data1['SMA_200'],
    mode='lines',
    name='200-day SMA',
    line=dict(color='#2ca02c', width=2, dash='dot')
))


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


st.plotly_chart(fig, use_container_width=True)
st.divider()


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


data1['RSI'] = calculate_RSI(data1['Close'].values)

st.title("📊 **Relative Strength Index (RSI)**")
st.write("Analyze the **RSI Indicator** to identify overbought and oversold conditions in stock prices.")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data1["Date"],
    y=data1['RSI'],
    mode='lines',
    name='RSI',
    line=dict(color='#1f77b4', width=2)
))

fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought (70)')
fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold (30)')


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


st.plotly_chart(fig, use_container_width=True)
st.divider()


def calculate_EMA(data, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[window-1] = np.mean(data[:window])
    for i in range(window, len(data)):
        ema[i] = alpha * (data[i] - ema[i-1]) + ema[i-1]
    return ema


def calculate_MACD(data, short_window=12, long_window=26, signal_window=9):
    ema_short = calculate_EMA(data, short_window)
    ema_long = calculate_EMA(data, long_window)
    macd = ema_short - ema_long
    signal_line = calculate_EMA(macd, signal_window)
    macd_histogram = macd - signal_line
    return macd, signal_line, macd_histogram

data1['MACD'], data1['Signal Line'], data1['MACD Histogram'] = calculate_MACD(data1['Close'].values)


st.title("📊 **MACD (Moving Average Convergence Divergence)**")
st.write("""
The **MACD Indicator** helps traders understand momentum and trend direction.
It consists of:
- **MACD Line:** Difference between short-term and long-term EMAs.
- **Signal Line:** EMA of the MACD Line.
- **Histogram:** Difference between MACD and Signal Line.
""")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data1["Date"],
    y=data1['MACD'],
    mode='lines',
    name='MACD',
    line=dict(color='blue', width=2)
))


fig.add_trace(go.Scatter(
    x=data1["Date"],
    y=data1['Signal Line'],
    mode='lines',
    name='Signal Line',
    line=dict(color='red', width=2, dash='dot')
))

fig.add_trace(go.Bar(
    x=data1["Date"],
    y=data1['MACD Histogram'],
    name='MACD Histogram',
    marker=dict(color='gray')
))


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


st.plotly_chart(fig, use_container_width=True)

st.divider()

sns.set(style="whitegrid")
data1['ROC'] = data1['Close'].pct_change(periods=12) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data1["Date"], data1['ROC'], label='Rate of Change (ROC)', color='#FF6F61', linewidth=2)


ax.set_title('Rate of Change (ROC) Over Time', fontsize=18, fontweight='bold', color='#333333')
ax.set_xlabel('Date', fontsize=14, color='#555555')
ax.set_ylabel('ROC (%)', fontsize=14, color='#555555')

ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=12)

st.subheader('Rate of Change (ROC) - A Financial Indicator')
st.markdown("This plot shows the rate of change of the closing prices over a period of 12 time units (e.g., days, months).")

st.pyplot(fig)
st.divider()


sns.set(style="whitegrid")


def compute_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


data1['RSI'] = compute_rsi(data1, window=14)

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(data1["Date"], data1['RSI'], label='RSI', color='#800080', linewidth=2)

ax.axhline(70, color='red', linestyle='--', label='Overbought (70)', linewidth=1.5)
ax.axhline(30, color='green', linestyle='--', label='Oversold (30)', linewidth=1.5)

ax.set_title('Relative Strength Index (RSI)', fontsize=18, fontweight='bold', color='#333333')
ax.set_xlabel('Date', fontsize=14, color='#555555')
ax.set_ylabel('RSI', fontsize=14, color='#555555')

ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=12)

st.subheader('Relative Strength Index (RSI) - A Momentum Indicator')
st.markdown("""
The RSI is used to identify overbought and oversold conditions in a market. 
- Overbought: RSI > 70 (Red line)
- Oversold: RSI < 30 (Green line)
""")


st.pyplot(fig)
st.divider()


data1["daily_return"] = data1["Close"].pct_change().fillna(0)




st.title("📊 Daily Stock Returns Over Time")


st.write("""
This interactive plot shows the daily stock returns across the entire dataset. Hover over points to view specific values.
""")

# Plot the Data
st.subheader("📈 Interactive Daily Returns Plot")
fig = px.line(
    data1,
    x="Date",
    y="daily_return",
    title="Daily Stock Returns Over Time",
    labels={"daily_return": "Daily Return (%)", "Date": "Date"},
    template="plotly_dark"
)
fig.update_traces(line=dict(color='cyan', width=2))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Daily Return (%)",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0.1)"
)

# Display the Plot
st.plotly_chart(fig, use_container_width=True)

st.divider()

sentiment_data.reset_index(inplace=True)
sentiment_data['date'] = pd.to_datetime(sentiment_data['date'], errors='coerce')
sentiment_data["Date"]=sentiment_data['date'].dt.date
sentiment_data["Date"]=pd.to_datetime(sentiment_data["Date"], errors="coerce")
sentiment_data["sentiment"]=sentiment_data["sentiment"]
sentiment_data['sentiment_Average'] = sentiment_data.groupby('Date')['sentiment'].transform('mean')
data1["Date"]=pd.to_datetime(data1["Date"])
sentiment_data['Date'] =sentiment_data['Date'].dt.tz_localize(None)
data_merge = sentiment_data.merge(
    data1[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits","daily_return"]],
    on="Date"
)


data_correlation=data_merge["sentiment_Average"].corr(data_merge["daily_return"], method="pearson")

st.title("📊 Correlation Between Sentiment and Stock Returns")

# Description
st.write("""
Explore the relationship between sentiment scores and daily stock returns using an interactive scatter plot. 
This visualization displays the full dataset without any filtering applied.
""")

# Subheader for Scatter Plot
st.subheader("📈 Interactive Scatter Plot")
fig = px.scatter(
    data_merge,
    x='sentiment_Average',
    y='daily_return',
    color='daily_return',  # Color the points based on the daily return
    size=data_merge['daily_return'].abs(),  # Ensure size values are non-negative
    hover_data=['sentiment_Average', 'daily_return'],
    title='Correlation Between Sentiment and Stock Returns',
    template='plotly_dark'
)

# Update plot with styling options
fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(
    xaxis_title="Average Sentiment Score",
    yaxis_title="Daily Stock Return",
    hovermode="closest",
    plot_bgcolor="rgba(0,0,0,0.1)"
)

# Display the scatter plot in Streamlit
st.plotly_chart(fig, use_container_width=True)
st.divider()

st.title('Sentiment and Stock Returns Heatmap')


st.title("🔥 Heatmap of Sentiment and Daily Returns Correlation")
st.write("""
Explore the correlation between sentiment scores and daily stock returns using an interactive heatmap.
""")

correlation_matrix = data_merge[['sentiment_Average', 'daily_return']].corr()

# Ensure no NaN values in the correlation matrix
if correlation_matrix.isnull().sum().sum() > 0:
    print("There are NaN values in the correlation matrix")
    correlation_matrix = correlation_matrix.fillna(0)  # or handle as needed

# Create the annotated heatmap
st.subheader("📊 Correlation Heatmap")
fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns.tolist(),
    y=correlation_matrix.index.tolist(),
    colorscale='viridis',
    annotation_text=correlation_matrix.round(2).astype(str).values,
    showscale=True
)

fig.update_layout(
    title_text='Heatmap of Sentiment and Daily Returns Correlation',
    xaxis=dict(title='Metrics'),
    yaxis=dict(title='Metrics'),
    plot_bgcolor='rgba(0,0,0,0.1)',
    margin=dict(l=60, r=60, t=60, b=60)
)

# Display the heatmap
st.plotly_chart(fig, use_container_width=True)