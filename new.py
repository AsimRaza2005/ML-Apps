import streamlit as st
import pickle
from nltk.corpus import stopwords
import re
import string
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

def clean_text(text):
    if text is None:
        return ''
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

def predict_sentiment(review, model_filename, vectorizer_filename):
    try:
        # Load the model and vectorizer
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_filename, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
    except FileNotFoundError as e:
        return f"Error loading files: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
    
    if not review.strip():
        return "Review is empty"
    
    # Clean the review text
    cleaned_review = clean_text(review)
    
    # Vectorize the review text using the loaded vectorizer
    vectorized_review = vectorizer.transform([cleaned_review])
    
    try:
        # Predict the sentiment
        prediction = model.predict(vectorized_review)
    except Exception as e:
        return f"Error during prediction: {e}"
    
    # Decode the sentiment
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    
    return sentiment

def main():
    st.title('Sentiment Analysis App')
    st.write("Enter a review below and get the sentiment prediction.")

    # Input review from the user
    review = st.text_area("Review", "")

    if st.button('Predict Sentiment'):
        if review:
            sentiment = predict_sentiment(review, 'model.pkl', 'vectorizer.pkl')
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter a review.")

if __name__ == "__main__":
    main()
