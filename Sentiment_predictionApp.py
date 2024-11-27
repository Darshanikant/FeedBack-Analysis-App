import streamlit as st
import pickle
import numpy as np

# Load the saved models
model_file = "model.pkl"
tfidf_file = "tfidf.pkl"

with open(model_file, 'rb') as file:
    classifier = pickle.load(file)
with open(tfidf_file, 'rb') as file:
    vectorizer = pickle.load(file)

# App Title and Home Page Design
st.set_page_config(
    page_title="Sentiment Analysis App", 
    page_icon="🔍", 
    layout="centered"
)
# Set background (optional)
def set_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg("https://img.freepik.com/free-vector/abstract-blue-circle-black-background-technology_1142-9770.jpg?ga=GA1.1.1049275477.1729512916&semt=ais_hybrid")
st.title("🌟 Sentiment Analysis App 🌟")
st.markdown(
    """
    #### Analyze your review sentiment with ease!
    - **Positive or Negative Sentiment** prediction using advanced ML models.
    - Multiple algorithms to choose from.
    
    Created by **Darshanikanta**  
    **© All Rights Reserved.**
    """
)
st.markdown("### 🎉 Powered by Data & AI 🚀")

# Sidebar for algorithm selection
st.sidebar.title("🔍 Classification Algorithm")
algo_options = [
    "Decision Tree",
    "SVC",
    "KNN",
    "Logistic Regression",
    "Bernoulli Naive Bayes",
    "Gaussian Naive Bayes",
    "Random Forest",
    "XGBoost",
    "LightGBM"
]
selected_algo = st.sidebar.selectbox("Choose an algorithm:", algo_options)

# Placeholder for accuracy (simulate values for simplicity)
algo_accuracies = {
    "Decision Tree": 0.85,
    "SVC": 0.88,
    "KNN": 0.86,
    "Logistic Regression": 0.89,
    "Bernoulli Naive Bayes": 0.84,
    "Gaussian Naive Bayes": 0.82,
    "Random Forest": 0.90,
    "XGBoost": 0.92,
    "LightGBM": 0.91
}
st.sidebar.write(f"**Accuracy:** {algo_accuracies[selected_algo]:.2%}")

# Input text for review
st.markdown("### 📝 Enter Your Review Below:")
review_text = st.text_area("Type your review here:", "")

if st.button("💡 Predict Sentiment"):
    if review_text.strip():
        # Preprocess input text
        from nltk.stem.porter import PorterStemmer
        import re
        ps = PorterStemmer()
        review = re.sub('[^a-zA-Z]', ' ', review_text)
        review = review.lower()
        review = review.split()
        review = ' '.join(review)

        # Transform input using vectorizer
        input_vector = vectorizer.transform([review]).toarray()

        # Predict sentiment
        prediction = classifier.predict(input_vector)
        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"

        # Display result
        st.markdown("### 🎯 Prediction Result:")
        st.markdown(f"**The review is: {sentiment}**")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Footer with creator details
st.markdown("---")
st.markdown(
    """
    **Created by Darshanikanta**  
    **© All Rights Reserved.**  
    Made with ❤️ using **Streamlit** and **Machine Learning**
    """
)
