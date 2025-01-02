import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import os




# Build the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../notebooks/data/raw_analyst_ratings.csv")

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

# --- 📊 Visualization in Streamlit ---
st.subheader("📅 **Number of Articles Published Per Month**")

# Create Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_counts.index, monthly_counts.values, marker='o', linestyle='-', color='#43A047', linewidth=2, label='Monthly Count')

# Add Labels and Title
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
ax.set_title('📊 Monthly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='upper left')
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)


yearly_count = publication_date.groupby('Year').size().reset_index(name='count')

# --- 📆 Visualization in Streamlit ---
st.subheader("📆 **Number of Publications per Year**")

# Create Plot
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(yearly_count['Year'], yearly_count['count'], color='#4CAF50', edgecolor='black')

# Add Labels and Title
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold')
ax.set_title('📊 Yearly Trend of Published Articles', fontsize=14, fontweight='bold', pad=20)

# Add Value Labels on Bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom', fontsize=10, color='black')

# Rotate X-axis labels for readability
ax.tick_params(axis='x', rotation=45, labelsize=10)

# Add Gridlines for better readability
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






