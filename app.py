
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("Sentiment Analysis App")
st.write("Upload a CSV file or enter text manually to analyze sentiments.")

# Sample training data
train_data = pd.DataFrame({
    "text": [
        "I love this movie",
        "This is amazing",
        "I feel great today",
        "What a wonderful experience",
        "I hate this",
        "This is terrible",
        "I am very sad",
        "I am not happy with this",
        "It was okay",
        "Not bad",
        "It is average",
        "Nothing special"
    ],
    "sentiment": [
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "neutral",
        "neutral"
    ]
})

# Train the model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
model = MultinomialNB()
model.fit(X_train, train_data['sentiment'])

# Function to analyze sentiment
def analyze_sentiment(texts):
    X_input = vectorizer.transform(texts)
    predictions = model.predict(X_input)
    return predictions

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        df['Sentiment'] = analyze_sentiment(df['text'])
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
        result = analyze_sentiment([user_input])[0]
        st.success(f"Predicted Sentiment: **{result.capitalize()}**")
