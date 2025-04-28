import streamlit as st
import pandas as pd
import os
import re

# Load stock symbols from indian_stock.csv
@st.cache_data
def load_data():
    file_path = "indian_stock.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=["Symbol", "Name", "Exchange"])  # Empty DataFrame

df = load_data()

# Streamlit UI
st.title("🔎 Stock Symbol Lookup")

search_query = st.text_input("Search for a stock by Name or Symbol:", "")

if search_query:
    # Compile regex pattern (case insensitive, escape special characters)
    try:
        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
        # Filter rows where any column matches the pattern
        results = df[df.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)]

        if not results.empty:
            st.write(results)
        else:
            st.warning("No matching stocks found.")
    except re.error:
        st.error("Invalid search pattern. Please try a simpler query.")
else:
    st.info("Type a company name or stock symbol to search.")

# Show full dataset option
if st.checkbox("Show Full Stock List"):
    st.write(df)

st.markdown("Made by Yahoo Finance and Mahua Dwivedi")
