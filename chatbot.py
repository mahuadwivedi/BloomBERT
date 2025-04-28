# add ai jailbreak restrictions...
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from restriction import check_restriction
import os
import faiss
import datetime
import numpy as np
import json
import yfinance as yf
import pandas as pd
import threading
import queue
import re
warnings.filterwarnings("ignore")

# Global variables for model instances
_mistral_pipeline = None
_embedding_model = None
_embedding_sentence_transformer = None
_model_loading_lock = threading.Lock()
_model_loading_queue = queue.Queue()

# Model configurations
emb_model = "sentence-transformers/all-MPNet-base-v2"  # dimension 768
HF_TOKEN = "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"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

def get_embedding_model():
    #Lazy loading for embedding model with caching.
    global _embedding_sentence_transformer
    
    if _embedding_sentence_transformer is None:
        with _model_loading_lock:
            if _embedding_sentence_transformer is None:
                print("[INFO] Loading sentence transformer embedding model...")
                _embedding_sentence_transformer = SentenceTransformer(emb_model)
                print("[INFO] Sentence transformer embedding model loaded.")
    
    return _embedding_sentence_transformer

def generate_embedding(text):
    """Generates embeddings for a given text."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype("float32")

def get_mistral_pipeline():
    """Lazy loading for Mistral pipeline with caching."""
    global _mistral_pipeline
    
    if _mistral_pipeline is None:
        with _model_loading_lock:
            if _mistral_pipeline is None:
                print("[INFO] Loading Mistral model...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        token=HF_TOKEN
                    )
                    _mistral_pipeline = pipeline(
                        'text-generation',
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.float16
                    )
                    print("[INFO] Mistral model loaded successfully.")
                except Exception as e:
                    print(f"[ERROR] Failed to load Mistral model: {e}")
                    return None
    
    return _mistral_pipeline

class StockVectorDB:
    FAISS_INDEX_FILE = "vectorstore.index"
    METADATA_FILE = "metadata.json"
    EMBEDDING_DIM = 512  # Matches your embedding dimension
    
    def __init__(self):
        self.index = self.load_faiss_index()
        self.metadata_store = self.load_metadata()

    def load_faiss_index(self):
        """Loads FAISS index or initializes a new one."""
        if os.path.exists(self.FAISS_INDEX_FILE):
            print("[INFO] Loading existing FAISS index...")
            return faiss.read_index(self.FAISS_INDEX_FILE)
        else:
            print("[INFO] No FAISS index found, creating a new one.")
            index_flat = faiss.IndexFlatL2(self.EMBEDDING_DIM)
            index = faiss.IndexIDMap(index_flat)
            return index

    def load_metadata(self):
        """Loads metadata from JSON file or initializes an empty dictionary."""
        if os.path.exists(self.METADATA_FILE):
            try:
                with open(self.METADATA_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                    print("[WARNING] Metadata file is corrupted or empty, resetting.")
        return {}

    def save_metadata(self):
        """Saves metadata dictionary to JSON file."""
        with open(self.METADATA_FILE, 'w') as f:
            json.dump(self.metadata_store, f, indent=4)
#Fetches stock data using yfinance with support for both NSE and BSE."""
        
    def fetch_stock_data(self, ticker):
        try:
            print(f"[INFO] Fetching stock data for {ticker}...")
            fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Add timestamp
            
            # Ensure proper suffix format
            original_ticker = ticker
            if not (ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")):
                ticker = ticker.upper() + ".NS"  # Default to NSE
                print(f"[INFO] No suffix found, trying with NSE suffix: {ticker}")
                
            stock = yf.Ticker(ticker)
            
            # Basic price data
            stock_info = stock.history(period="1d")
            
            # If NSE data is empty, try BSE
            if stock_info.empty and ticker.upper().endswith(".NS"):
                bse_ticker = ticker.upper().replace(".NS", ".BO")
                print(f"[INFO] NSE data not found, trying BSE: {bse_ticker}")
                stock = yf.Ticker(bse_ticker)
                stock_info = stock.history(period="1d")
                ticker = bse_ticker  # Update ticker to BSE version
                
            # If BSE data is empty, try NSE
            elif stock_info.empty and ticker.upper().endswith(".BO"):
                nse_ticker = ticker.upper().replace(".BO", ".NS") 
                print(f"[INFO] BSE data not found, trying NSE: {nse_ticker}")
                stock = yf.Ticker(nse_ticker)
                stock_info = stock.history(period="1d")
                ticker = nse_ticker  # Update ticker to NSE version
                
            # If still empty, try without any suffix
            if stock_info.empty and (ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")):
                base_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
                print(f"[INFO] Suffix data not found, trying without suffix: {base_ticker}")
                stock = yf.Ticker(base_ticker)
                stock_info = stock.history(period="1d")
                ticker = base_ticker  # Update ticker to base version
                
            if stock_info.empty:
                print(f"[INFO] No data found for ticker: {original_ticker}")
                return {}
            
        # Get additional information
            info = stock.info
                
            return {
                "stock_price": float(stock_info["Close"].iloc[-1]),  
                "open_price": float(stock_info["Open"].iloc[-1]),
                "high_price": float(stock_info["High"].iloc[-1]),
                "low_price": float(stock_info["Low"].iloc[-1]),
                "volume": int(stock_info["Volume"].iloc[-1]),
                "market_cap": info.get("marketCap", None),
                "pe_ratio": info.get("trailingPE", None),
                "dividend_yield": info.get("dividendYield", None),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector", None),
                "industry": info.get("industry", None),
                "fetch_timestamp": fetch_time  # Add timestamp to the returned data
            }
        except Exception as e:
            print(f"[ERROR] Failed to fetch stock data for {ticker}: {e}")
            return {}

    def add_stock_embedding(self, user_query):
        """Adds stock embedding based on user query."""
        # Extract potential ticker from user query
        words = user_query.split()
        possible_tickers = [word.upper() for word in words if len(word) <= 10 and word.isalnum()]
        added_stocks = []
        for ticker in possible_tickers:
            # Avoid duplicates
            existing_ids = [id for id, meta in self.metadata_store.items() if meta.get("ticker") == ticker]
            if existing_ids:
                print(f"[WARNING] {ticker} already exists in FAISS.")
                # Update stock data for existing ticker
                for id in existing_ids:
                    stock_data = self.fetch_stock_data(ticker)
                    if stock_data:
                        self.metadata_store[id]["stock_data"] = stock_data
                        self.save_metadata()
                        print(f"[INFO] Updated stock data for {ticker}.")
                        added_stocks.append(ticker)
                continue

            # Fetch stock data
            stock_data = self.fetch_stock_data(ticker)
            if not stock_data:
                print(f"[ERROR] No data found for {ticker}. Skipping.")
                continue

            # Generate embedding for the query
            embedding = generate_embedding(user_query)

            embedding_id = self.index.ntotal
            self.metadata_store[str(embedding_id)] = {
                "ticker": ticker,
                "stock_data": stock_data,
                "original_query": user_query
            }

            self.save_metadata()

            vector = np.array(embedding).astype("float32").reshape(1, -1)
            self.index.add_with_ids(vector, np.array([embedding_id]))
            faiss.write_index(self.index, self.FAISS_INDEX_FILE)
            print(f"[SUCCESS] Added {ticker}. FAISS now has {self.index.ntotal} embeddings.")
            added_stocks.append(ticker)

        return added_stocks

    def search_similar_stocks(self, query, top_k=3):
        """Search for similar stocks based on embedding similarity."""
        query_embedding = generate_embedding(query)
        query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
           
        # Perform similarity search
        D, I = self.index.search(query_vector, top_k)
            
        similar_stocks = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and str(idx) in self.metadata_store:  # Check for valid index
                stock_info = self.metadata_store[str(idx)]
                similar_stocks.append({
                    "ticker": stock_info.get("ticker"),
                    "stock_data": stock_info.get("stock_data"),
                    "distance": float(dist)
                })
            
        return similar_stocks
    
    def get_stock_by_ticker(self, ticker):
        ticker = ticker.upper()
        
        # Check if we already have this stock in our metadata
        for id, metadata in self.metadata_store.items():
            if metadata.get("ticker") == ticker:
                return metadata.get("stock_data", {})
        
        # If stock not found, try to fetch it
        stock_data = self.fetch_stock_data(ticker)
        
        # If data not found with current ticker and it doesn't have a suffix, try with .NS
        if not stock_data and not (ticker.endswith('.NS') or ticker.endswith('.BO')):
            print(f"[INFO] No data found for {ticker}, trying with .NS suffix")
            stock_data = self.fetch_stock_data(f"{ticker}.NS")
            if stock_data:
                ticker = f"{ticker}.NS"
        
        # If still no data and it doesn't have .BO suffix or has .NS suffix, try with .BO 
        if not stock_data:
            if ticker.endswith('.NS'):
                new_ticker = ticker.replace('.NS', '.BO')
                print(f"[INFO] No data found for {ticker}, trying with .BO suffix as {new_ticker}")
                stock_data = self.fetch_stock_data(new_ticker)
                if stock_data:
                    ticker = new_ticker
            elif not ticker.endswith('.BO'):
                new_ticker = f"{ticker}.BO"
                print(f"[INFO] No data found for {ticker}, trying with .BO suffix as {new_ticker}")
                stock_data = self.fetch_stock_data(new_ticker)
                if stock_data:
                    ticker = new_ticker
        
        if stock_data:
            # Add it to the database while we're at it
            embedding = generate_embedding(f"stock information for {ticker}")
            embedding_id = self.index.ntotal
            self.metadata_store[str(embedding_id)] = {
                "ticker": ticker,
                "stock_data": stock_data,
                "original_query": f"stock information for {ticker}"
            }
            vector = np.array(embedding).astype("float32").reshape(1, -1)
            self.index.add_with_ids(vector, np.array([embedding_id]))
            self.save_metadata()
            faiss.write_index(self.index, self.FAISS_INDEX_FILE)
            print(f"[SUCCESS] Added {ticker} during lookup.")
            return stock_data
        
        return {}

# Create the global stock_vector_db instance
stock_vector_db = StockVectorDB()

def extract_ticker_and_attribute(query):
    """Extract ticker symbol and requested attribute from query."""
    # Common patterns for stock queries
    ticker_pattern = r'([A-Za-z0-9\.]+)(?:\.ns|\.bo)?\s+(market cap|price|pe ratio|dividend|volume|high|low|open|close|52 week|sector|industry)'
    match = re.search(ticker_pattern, query, re.IGNORECASE)
    
    if match:
        ticker = match.group(1).upper()
        attribute = match.group(2).lower()
        
        # Check if .NS or .BO was mentioned elsewhere in the query
        if 'india' in query.lower() or 'nse' in query.lower() or 'irfc' in query.lower():
            ticker = ticker + '.NS' if not ticker.endswith('.NS') and not ticker.endswith('.BO') else ticker
        elif 'bse' in query.lower() or 'bombay' in query.lower():
            ticker = ticker + '.BO' if not ticker.endswith('.BO') and not ticker.endswith('.NS') else ticker
            
        return ticker, attribute
    
    # Check if it's just a ticker query
    words = query.split()
    for word in words:
        if len(word) >= 2 and len(word) <= 10 and any(c.isalpha() for c in word):
            # Potential ticker
            ticker = word.upper()
            # Handle suffixes based on keywords
            if 'india' in query.lower() or 'nse' in query.lower() or word.lower() == 'irfc':
                ticker = ticker + '.NS' if not ticker.endswith('.NS') and not ticker.endswith('.BO') else ticker
            elif 'bse' in query.lower() or 'bombay' in query.lower():
                ticker = ticker + '.BO' if not ticker.endswith('.BO') and not ticker.endswith('.NS') else ticker
            return ticker, "general"
    
    return None, None

def get_attribute_value(stock_data, attribute):
    """Get the specific attribute from stock data."""
    if attribute == "market cap" and stock_data.get("market_cap"):
        market_cap = stock_data.get("market_cap")
        if market_cap >= 1_000_000_000:
            return f"₹{market_cap/1_000_000_000:.2f} billion"
        else:
            return f"₹{market_cap/1_000_000:.2f} million"
    
    elif attribute == "price":
        return f"{stock_data.get('stock_price', 'N/A')}"
    
    elif attribute == "pe ratio":
        pe = stock_data.get("pe_ratio")
        return f"{pe:.2f}" if pe else "N/A"
    
    elif attribute == "dividend":
        div = stock_data.get("dividend_yield")
        return f"{div*100:.2f}%" if div else "N/A"
    
    elif attribute == "volume":
        return f"{stock_data.get('volume', 'N/A'):,}"
    
    elif attribute == "high":
        return f"{stock_data.get('high_price', 'N/A')}"
    
    elif attribute == "low":
        return f"{stock_data.get('low_price', 'N/A')}"
    
    elif attribute == "open":
        return f"{stock_data.get('open_price', 'N/A')}"
    
    elif attribute == "52 week":
        high = stock_data.get("fifty_two_week_high")
        low = stock_data.get("fifty_two_week_low")
        if high and low:
            return f"52-Week Range: ₹{low} - ₹{high}"
        return "52-Week data not available"
    
    elif attribute == "sector":
        return stock_data.get("sector", "N/A")
    
    elif attribute == "industry":
        return stock_data.get("industry", "N/A")
    
    elif attribute == "time" or attribute == "timestamp":
        return stock_data.get("fetch_timestamp", "N/A")
    
    elif attribute == "general":
        # Return general information about the stock
        name = stock_data.get("company_name", "")
        price = stock_data.get("stock_price", "N/A")
        change = stock_data.get("stock_price", 0) - stock_data.get("open_price", 0)
        change_pct = (change / stock_data.get("open_price", 1)) * 100 if stock_data.get("open_price") else 0
        timestamp = stock_data.get("fetch_timestamp", "N/A")
        
        return f"{name} ({price:.2f}, {change_pct:.2f}%) [Data as of {timestamp}]"
    
    return "Information not available"

def process_user_query(query):
    """Process natural language query about stocks."""
    ticker, attribute = extract_ticker_and_attribute(query)
    if ticker:
        # Get stock data for the ticker
        stock_data = stock_vector_db.get_stock_by_ticker(ticker)
        
        if not stock_data:
            return f"I couldn't find any data for {ticker}. Please check the ticker symbol and try again."
        
        if attribute:
            company_name = stock_data.get("company_name", ticker)
            value = get_attribute_value(stock_data, attribute)
            
            if attribute == "general":
                return f"{value}"
            else:
                return f"{company_name} {attribute}: {value}"
    
    # If we can't identify a specific query, try to find similar stocks
    similar_stocks = stock_vector_db.search_similar_stocks(query, top_k=1)
    if similar_stocks:
        stock = similar_stocks[0]
        ticker = stock["ticker"]
        stock_data = stock["stock_data"]
        company_name = stock_data.get("company_name", ticker)
        price = stock_data.get("stock_price", "N/A")
        
        return f"Based on your query, you might be interested in: {company_name} ({ticker}) - Current price: ₹{price:.2f}"
    
    # Try to find any stock symbol in the query and add it
    added_stocks = stock_vector_db.add_stock_embedding(query)
    if added_stocks:
        ticker = added_stocks[0]
        stock_data = stock_vector_db.get_stock_by_ticker(ticker)
        company_name = stock_data.get("company_name", ticker)
        price = stock_data.get("stock_price", "N/A")
        
        return f"I found information for {company_name} ({ticker}). Current price: ₹{price:.2f}"
    
    return "I couldn't understand your stock query. Please try asking about a specific stock by mentioning its ticker symbol, for example: 'TATAMOTORS.NS market cap' or 'What is the price of RELIANCE?'"

# Start model loading in the background thread
def background_model_loader():
    """Load models in the background to not block the main thread."""
    # Just trigger the loading by calling the getters
    get_embedding_model()
    get_mistral_pipeline()
    print("[INFO] Background model loading completed.")

# Command line interface for testing
def chatbot_interface():
    """Simple chatbot interface for testing."""
    print("\n=== Stock Information Chatbot ===")
    print("Ask about stocks like: 'TATAMOTORS.NS market cap' or 'What is the price of RELIANCE?'")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        response = process_user_query(user_input)
        print(f"Bot: {response}\n")

# Run this if the script is executed directly
if __name__ == "__main__":
    # Start background loading thread after creating StockVectorDB
    threading.Thread(target=background_model_loader, daemon=True).start()
    
    # Start the chatbot interface
    chatbot_interface()