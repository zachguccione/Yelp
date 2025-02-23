import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Restaurant Sentiment Analysis Dashboard", layout="wide")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('data/aspect_sentiments.csv')
    # Convert review_date to datetime
    df['review_date'] = pd.to_datetime(df['review_date'])
    return df

# Load the data
df = load_data()

# Title
st.title("Restaurant Sentiment Analysis Dashboard")

# Sidebar for filters
st.sidebar.header("Filters")

# City selection
city_list = sorted(df['business_city'].unique())
selected_city = st.sidebar.selectbox(
    "Select a City",
    city_list
)

# Filter restaurants based on selected city
city_restaurants = df[df['business_city'] == selected_city]
restaurant_list = sorted(city_restaurants['business_name'].unique())

# Restaurant selection
selected_restaurant = st.sidebar.selectbox(
    "Select a Restaurant",
    restaurant_list
)

# Sentiment metric selection with clean names
sentiment_mapping = {
    'Food Quality_Sentiment': 'Food Quality', 
    'Service_Sentiment': 'Service',
    'Ambiance_Sentiment': 'Ambiance',
    'Wait Time_Sentiment': 'Wait Time',
    'Price/Value_Sentiment': 'Price/Value',
    'Menu Variety_Sentiment': 'Menu Variety',
    'Cleanliness_Sentiment': 'Cleanliness'
}

# Create reversed mapping for looking up original column names
reverse_mapping = {v: k for k, v in sentiment_mapping.items()}

# Use clean names in dropdown
selected_sentiment_clean = st.sidebar.selectbox(
    "Select Sentiment Metric",
    list(sentiment_mapping.values())
)

# Convert back to original column name for data processing
selected_sentiment = reverse_mapping[selected_sentiment_clean]

# Filter data based on selection
filtered_df = df[df['business_name'] == selected_restaurant]

# Function to format review text with line breaks
def format_review(text, width=50):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '<br>'.join(lines)

# Calculate trend line
def create_trendline(df, sentiment_col):
    try:
        # Remove any null values
        df_clean = df.dropna(subset=[sentiment_col, 'review_date'])
        
        if len(df_clean) < 2:  # Need at least 2 points for a trendline
            st.warning(f"Not enough reviews for trendline: {len(df_clean)} valid reviews")
            return None
            
        # Check if all sentiment values are the same
        if df_clean[sentiment_col].nunique() == 1:
            st.warning("All sentiment values are identical - cannot create trendline")
            return None
            
        # Convert dates to numeric values
        x_numeric = (df_clean['review_date'] - df_clean['review_date'].min()).dt.total_seconds()
        y_values = df_clean[sentiment_col].values
        
        # Fit the line
        z = np.polyfit(x_numeric, y_values, 1)
        p = np.poly1d(z)
        
        # Create trend line dates for plotting
        trend_x = [df_clean['review_date'].min(), df_clean['review_date'].max()]
        trend_x_numeric = [(d - df_clean['review_date'].min()).total_seconds() for d in trend_x]
        trend_y = p(trend_x_numeric)
        
        return go.Scatter(
            x=trend_x,
            y=trend_y,
            mode='lines',
            name='Trend',
            line=dict(color='red', width=3)
        )
    except Exception as e:
        st.error(f"Error creating trendline: {str(e)}")
        return None

# Create scatter plot first
scatter = go.Scatter(
    x=filtered_df['review_date'],
    y=filtered_df[selected_sentiment],
    mode='markers',
    name='Reviews',
    marker=dict(size=8, opacity=0.6),
    hovertemplate=(
        "<b>Date:</b> %{x|%Y-%m-%d}<br><br>" +
        "<b>Sentiment:</b> %{y:.2f}<br><br>" +
        "<b>Review:</b><br>" +
        "%{customdata[0]}<br>" +
        "<extra></extra>"
    ),
    customdata=[[format_review(text)] for text in filtered_df['review_text']]
)

# Create figure and add traces
fig = go.Figure(data=[scatter])

# Add trendline if it can be created
trendline = create_trendline(filtered_df, selected_sentiment)
if trendline is not None:
    fig.add_trace(trendline)

# Update layout
fig.update_layout(
    title=f'{selected_sentiment_clean} Trend for {selected_restaurant}',
    xaxis_title="Date",
    yaxis_title="Sentiment Score",
    hovermode='closest',
    showlegend=True,
    hoverlabel=dict(
        bgcolor="rgba(50, 50, 50, 0.95)",
        font_size=12,
        font=dict(color="white"),
        align="left"
    )
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Display some statistics
st.subheader("Restaurant Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    avg_sentiment = filtered_df[selected_sentiment].mean()
    st.metric("Average Sentiment", f"{avg_sentiment:.2f}")

with col2:
    review_count = len(filtered_df)
    st.metric("Total Reviews", review_count)

with col3:
    latest_review = filtered_df['review_date'].max()
    st.metric("Latest Review", latest_review.strftime('%Y-%m-%d'))