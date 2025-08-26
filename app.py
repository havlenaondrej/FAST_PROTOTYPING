



from dotenv import load_dotenv
import os

import streamlit as st
import pandas as pd
import openai


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Customer Review Analysis", layout="wide")
if not openai.api_key:
    st.warning("⚠️ OpenAI API key not found. Check your .env file.")
st.title("🛍️ Customer Review Analysis")
st.markdown("""
Analyze customer feedback across products using GenAI-powered sentiment detection.
Upload your CSV file, filter by product, and visualize the sentiment breakdown to uncover insights.
""")
st.markdown("### 📁 Upload Your CSV File")

uploaded_file = st.file_uploader("📁 Choose a CSV file", type=["csv"], help="Upload a CSV file containing customer reviews.")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Preview of uploaded data:")
    st.dataframe(df.head())

    # Step 2: Validate required columns
    required_cols = {"SUMMARY", "PRODUCT"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain 'SUMMARY' and 'PRODUCT' columns.")
    else:
        # Step 3: Filter by product
        product_list = df["PRODUCT"].dropna().unique()
        selected_product = st.selectbox("Filter by product", product_list)
        filtered_df = df[df["PRODUCT"] == selected_product]

        if st.button("🔍 Run Sentiment Analysis"):
            # Step 4: Run sentiment analysis using OpenAI
            def analyze_sentiment(text):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for classifying sentiment."},
                            {"role": "user", "content": f"What is the sentiment of this review: \"{text}\"? Respond with one word: Positive, Neutral, or Negative."}
                        ],
                        max_tokens=1
                    )
                    return response.choices[0].message["content"].strip()
                except Exception as e:
                    return "Error"

            with st.spinner("Analyzing reviews..."):
                filtered_df["sentiment"] = filtered_df["SUMMARY"].apply(analyze_sentiment)

            # Step 5: Visualize results
            st.subheader("📈 Sentiment Breakdown")
            sentiment_counts = filtered_df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            # Optional: Show data
            with st.expander("🔍 See full results"):
                st.dataframe(filtered_df[["PRODUCT", "SUMMARY", "sentiment"]])