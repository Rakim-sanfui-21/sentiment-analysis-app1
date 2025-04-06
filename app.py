
import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already present
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Title
st.title("Advanced Sentiment Analysis App")
st.write("Upload a CSV file or enter text manually to analyze sentiments using VADER.")

# Function to get sentiment category
def get_sentiment(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        df['Sentiment'] = df['text'].apply(get_sentiment)
        # Count sentiment percentages
        sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
        sentiment_table = pd.DataFrame({
            "Sentiment": sentiment_counts.index,
            "Percentage": sentiment_counts.values.round(2)
        })
        st.subheader("Sentiment Breakdown (CSV Data)")
        st.write(df)
        st.write(sentiment_table)

# Manual input
st.subheader("Enter Text Manually")
user_input = st.text_area("Enter a sentence:")
if st.button("Analyze"):
    if user_input.strip() != "":
        sentiment = get_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
