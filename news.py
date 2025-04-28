import streamlit as st
import requests

st.set_page_config(page_title="BloomBERT News", layout="wide")
st.title("BloomBERT News")
st.write("Real-time Financial News")

API_KEY = "5aa0a79733ff44f9a2ef791dc59219e7"

country_options = {
    "United States": "us",
    "India": "in",
    "China (via keyword)": "cn",
    "United Kingdom": "gb",
}

selected_country = st.sidebar.selectbox("Select Country", list(country_options.keys()))
country_code = country_options[selected_country]

# Base URLs
top_headlines_url = "https://newsapi.org/v2/top-headlines"
everything_url = "https://newsapi.org/v2/everything"

# Decide which endpoint to use
if country_code == "us":
    # China: use keyword search because top-headlines does not support CN
    params = {
        "category": "business",
        "country": country_code,
        "pageSize": 10,
        "apiKey": API_KEY
    }
    url = top_headlines_url
else:
    # US and India: use top-headlines
    params = {
        "category": "business",
        "country": country_code,
        "pageSize": 10,
        "apiKey": API_KEY
    }
    url = everything_url

response = requests.get(url, params=params)

# Display results
if response.status_code == 200:
    data = response.json()
    articles = data.get("articles", [])

    if articles:
        for article in articles:
            st.subheader(article["title"])
            st.markdown(f"[Read more...]({article['url']})")
            st.markdown("---")
    else:
        st.info("No articles found for this selection.")
else:
    st.error("Unable to fetch news. Please check your API key or quota.")
