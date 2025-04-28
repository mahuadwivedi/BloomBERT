import flask
import json
import os
import re
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import get_mistral_pipeline, stock_vector_db  # Import with new lazy loading
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import time
from ticker_buffer import TickerBuffer
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
from ipo_functions import fetch_ipo_data, handle_ipo_query, is_ipo_query
from restriction import check_restriction
from chatbot import get_embedding_model

app = Flask(__name__)
models_ready = False
CORS(app)
conversation_context=None

# In main.py, add this after app initialization
def initialize_models():
    global models_ready
    try:
        print("Starting model initialization...")
        start_time = time.time()
        
        # Initialize with timeout protection
        with ThreadPoolExecutor() as executor:
            # Run both model loading tasks with timeout
            embedding_future = executor.submit(get_embedding_model)
            mistral_future = executor.submit(get_mistral_pipeline)
            
            try:
                embedding_model = embedding_future.result(timeout=120)  # 2 minute timeout
                mistral_model = mistral_future.result(timeout=120)
                
                if embedding_model is not None and mistral_model is not None:
                    models_ready = True
                    print("All models successfully initialized and ready!")
                else:
                    print("WARNING: At least one model failed to load properly")
            except TimeoutError:
                print("ERROR: Model initialization timed out")
                models_ready = False
        
        elapsed = time.time() - start_time
        print(f"Model initialization took {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"ERROR initializing models: {str(e)}")
        models_ready = False

def fetch_stock_data(ticker):
    try:
        # Add .NS suffix for Indian stocks if not already present
        if ticker.lower() in ['tatamotors', 'reliance', 'tata', 'tcs'] and not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
            
        # Get stock info
        stock = yf.Ticker(ticker)
        data = stock.info
        
        # Extract relevant information
        price = data.get('currentPrice', data.get('regularMarketPrice', 'N/A'))
        change = data.get('regularMarketChangePercent', 'N/A')
        if change != 'N/A':
            change = round(change * 100, 2)  # Convert to percentage and round
            
        # Construct response data
        stock_data = {
            'price': price,
            'change': change,
            'day_low': data.get('dayLow', 'N/A'),
            'day_high': data.get('dayHigh', 'N/A'),
            'volume': data.get('volume', 'N/A'),
            'market_cap': data.get('marketCap', 'N/A'),
            'pe_ratio': data.get('trailingPE', 'N/A')
        }
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def search_company(query):
    """
    Attempt to identify a valid ticker from a full or partial company name.
    """
    try:
        # Extract words that are not part of financial query terms
        financial_terms = ['market cap', 'volume', 'price', 'change', 'range']
        lower_query = query.lower()
        user_query = next((term for term in financial_terms if term in lower_query), None)
        company_part = lower_query.replace(user_query, '').strip() if user_query else lower_query

        # Try searching with yfinance
        search_result = yf.Ticker(company_part)
        info = search_result.info

        if 'longName' in info or 'shortName' in info:
            return {
                "symbol": info.get("symbol", "N/A"),
                "longName": info.get("longName", info.get("shortName", "")),
                "sector": info.get("sector", "N/A"),
                "marketCap": info.get("marketCap", "N/A"),
                "volume": info.get("volume", "N/A"),
                "currentPrice": info.get("currentPrice", "N/A"),
                "user_query": user_query or "basic info"
            }
        else:
            return None
    except Exception as e:
        print("Error in search_company:", e)
        return None
    

def lookup_company_info(message):
    # Define keywords you want to detect
    keywords = ['market cap', 'volume', 'price', 'change', 'day range']
    
    detected_query = next((kw for kw in keywords if kw in message.lower()), None)
    if not detected_query:
        return None
    
    words = message.lower().split()
    
    # Remove known keywords to isolate possible company name
    for kw in keywords:
        message = message.replace(kw, '')
    company_name_guess = message.strip()
    
    # Use Yahoo Finance search to find a match
    search_results = yf.Ticker(company_name_guess)
    info = search_results.info

    if 'longName' in info:
        return {
            "longName": info.get("longName", ""),
            "symbol": info.get("symbol", ""),
            "sector": info.get("sector", ""),
            "marketCap": info.get("marketCap", "N/A"),
            "user_query": detected_query
        }
    else:
        return None

def search_company_by_name(company_name):
    """Search for a company by name and return possible ticker matches with improved handling"""
    # This is a generalized function that doesn't rely on hardcoded companies
    possible_matches = []
    search_terms = []
    
    # Clean and prepare search terms
    company_name_lower = company_name.lower()
    words = company_name_lower.split()
    
    # Original company name
    search_terms.append(company_name_lower)
    
    # Remove spaces
    search_terms.append(''.join(words))
    
    # First word only (useful for companies like Tata, Reliance)
    if len(words) >= 1 and len(words[0]) >= 3:
        search_terms.append(words[0])
    
    # First two words (common for company names)
    if len(words) >= 2:
        search_terms.append(f"{words[0]} {words[1]}")
        search_terms.append(f"{words[0]}{words[1]}")
    
    # First letters of multi-word names (e.g. "TCS" from "Tata Consultancy Services")
    if len(words) >= 2:
        initials = ''.join(word[0] for word in words if word)
        if len(initials) >= 2:
            search_terms.append(initials)
    
    # Common Indian exchange suffixes
    suffixes = ['.NS', '.BO', '', '.BSE', '.NSE']
    
    # Generate tickers for each pattern with suffixes
    potential_tickers = []
    for term in search_terms:
        term_clean = re.sub(r'[^\w\s]', '', term)  # Remove special chars
        
        # Add variations
        potential_tickers.append(term_clean)
        potential_tickers.append(term_clean.upper())
        
        for suffix in suffixes:
            potential_tickers.append(term_clean + suffix)
            potential_tickers.append(term_clean.upper() + suffix)
    
    # Add India-specific mappings
    indian_mappings = {
        'tata motors': ['TATAMOTORS', 'TATAMOTORS.NS'],
        'tata': ['TATAMOTORS', 'TCS', 'TATASTEEL', 'TATACHEM', 'TATAPOWER'],
        'reliance': ['RELIANCE', 'RELIANCE.NS'],
        'indusind bank': ['INDUSINDBK', 'INDUSINDBK.NS'],
        'hdfc bank': ['HDFCBANK', 'HDFCBANK.NS'],
        'state bank': ['SBIN', 'SBIN.NS'],
        'infosys': ['INFY', 'INFY.NS'],
        'icici bank': ['ICICIBANK', 'ICICIBANK.NS'],
        'axis bank': ['AXISBANK', 'AXISBANK.NS'],
        'kotak': ['KOTAKBANK', 'KOTAKBANK.NS'],
        'l&t': ['LT', 'LT.NS', 'LTIM', 'LTIM.NS'],
        'larsen': ['LT', 'LT.NS'],
        'bajaj': ['BAJFINANCE', 'BAJAJFINSV', 'BAJAJ-AUTO'],
        'maruti': ['MARUTI', 'MARUTI.NS'],
        'bharti airtel': ['BHARTIARTL', 'BHARTIARTL.NS'],
        'airtel': ['BHARTIARTL', 'BHARTIARTL.NS'],
        'hindustan unilever': ['HINDUNILVR', 'HINDUNILVR.NS'],
        'itc': ['ITC', 'ITC.NS'],
        'sun pharma': ['SUNPHARMA', 'SUNPHARMA.NS'],
        'wipro': ['WIPRO', 'WIPRO.NS'],
        'adani': ['ADANIPORTS', 'ADANIGREEN', 'ADANIENT'],
    }
    
    # Check if company name matches any known mappings
    for key, values in indian_mappings.items():
        if key in company_name_lower or company_name_lower in key:
            potential_tickers.extend(values)
    
    # Remove duplicates
    return list(set(potential_tickers))

class ConversationContext:
    def __init__(self):
        self.last_interaction = None
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges
        self.current_company = None
        self.company_info = {}
        self.recent_tickers = []  # Track recently mentioned tickers
        
    def add_interaction(self, user_query, bot_response):
        """Add an interaction to conversation history"""
        self.last_interaction = datetime.now()
        self.conversation_history.append({
            "user": user_query,
            "bot": bot_response,
            "timestamp": self.last_interaction
        })
        
        # Keep history to max length
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def set_company_context(self, ticker, company_name, description=None, sector=None, industry=None):
        """Set current company context"""
        self.current_company = ticker
        self.company_info = {
            "ticker": ticker,
            "name": company_name,
            "description": description,
            "sector": sector,
            "industry": industry,
            "last_mentioned": datetime.now()
        }
        
        # Add to recent tickers, keep only last 3
        if ticker not in self.recent_tickers:
            self.recent_tickers.insert(0, ticker)
            if len(self.recent_tickers) > 3:
                self.recent_tickers.pop()
    
    def get_company_context(self):
        """Get current company context if recent"""
        if not self.current_company:
            return None
            
        # Only return context if it's recent (within 10 minutes)
        cutoff = datetime.now() - timedelta(minutes=10)
        if self.company_info.get("last_mentioned", datetime.min) < cutoff:
            return None
            
        return self.company_info
    
    def is_follow_up_question(self, query):
        """Check if query appears to be a follow-up about the current company"""
        query_lower = query.lower()
        
        # Common follow-up indicators
        follow_up_terms = [
            "its", "it's", "it", "their", "they", "this company", "this stock",
            "the company", "the stock", "them", "what about", "and how about",
            "what's", "how about", "tell me more", "more info"
        ]
        
        # Check if query contains follow-up indicators
        has_follow_up_term = any(term in query_lower for term in follow_up_terms)
        
        # Check if query is very short (likely a follow-up)
        is_short_query = len(query_lower.split()) <= 4
        
        # Check for question without specific company mention
        is_question = any(q in query_lower for q in ["what", "how", "when", "why", "where", "?"])
        
        # Check if the company name or ticker is in the query
        company_info = self.get_company_context()
        if company_info:
            company_name = company_info.get("name", "").lower()
            ticker = company_info.get("ticker", "").lower()
            company_words = set(company_name.lower().split())
            
            # Check if query contains the ticker or full company name
            has_company_reference = ticker.lower() in query_lower
            
            # Check partial company name (more flexible matching)
            words_in_query = set(query_lower.split())
            company_word_overlap = len(company_words.intersection(words_in_query)) > 0
            
            if has_company_reference or company_word_overlap:
                return True
        
        # No direct company reference but has follow-up indicators or is a short question
        return (has_follow_up_term or 
                (is_short_query and is_question and 
                 not any(x in query_lower for x in ["hello", "hi", "bye", "thanks"])))


def get_market_news():
    """Get general market news as fallback"""
    try:
        # Use the Market index as a proxy for market news
        market_ticker = "^NSEI"  # NIFTY 50
        stock = yf.Ticker(market_ticker)
        news = stock.news
        
        if not news:
            return {
                "message": "BloomBERT >> \n\n No market news found at this time.",
                "source": ""
            }
            
        response = "## Latest Market News\n\n"
        
        for i, item in enumerate(news[:5], 1):
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown source')
            
            # Convert timestamp to readable date
            publish_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
            date_str = publish_time.strftime('%d %b %Y')
            
            response += f"### {i}. {title}\n"
            response += f"*{publisher} - {date_str}*\n\n"
            
            # Add a short summary if available
            if 'summary' in item and item['summary']:
                summary = item['summary']
                # Truncate long summaries
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                response += f"{summary}\n\n"
        
        response += "For more market news, visit financial news websites or market data providers."
        
        return {
            "message": response,
            "source": "https://finance.yahoo.com/quote/^NSEI"
        }
    except Exception as e:
        print(f"Error fetching market news: {e}")
        return {
            "message": "BloomBERT >> \n\n Sorry, I couldn't retrieve market news at this time.",
            "source": ""
        }
    
def get_name_to_ticker_mapping():
    """Build or load a mapping of company names to ticker symbols"""
    try:
        # Check if we have a saved mapping file
        if os.path.exists("company_name_mapping.json"):
            with open("company_name_mapping.json", "r") as f:
                return json.load(f)
        
        # If not, initialize an empty mapping
        return {}
    except Exception as e:
        print(f"Error loading company name mapping: {e}")
        return {}

def update_name_mapping(name, ticker):
    """Add or update the name-ticker mapping with improved full name handling"""
    try:
        mapping = get_name_to_ticker_mapping()
        
        # Store various forms of the name
        name_lower = name.lower()
        
        # Store the original full name
        mapping[name_lower] = ticker
        
        # Store without spaces
        mapping[name_lower.replace(" ", "")] = ticker
        
        # Store individual words and combinations
        words = name_lower.split()
        
        # Store the first word if it's significant (like "Tata" or "Reliance")
        if len(words) > 0 and len(words[0]) > 3:
            mapping[words[0]] = ticker
            
        # Store first two words combined (common company name pattern)
        if len(words) >= 2:
            two_word = f"{words[0]} {words[1]}"
            mapping[two_word] = ticker
            mapping[two_word.replace(" ", "")] = ticker
        
        # If the name has more than two words, also store without common terms
        if len(words) > 2:
            common_terms = ["limited", "ltd", "corporation", "corp", "inc", "industries", "company", "co"]
            filtered_words = [w for w in words if w.lower() not in common_terms]
            if filtered_words:
                filtered_name = " ".join(filtered_words)
                mapping[filtered_name] = ticker
                mapping[filtered_name.replace(" ", "")] = ticker
        
        # Save the updated mapping
        with open("company_name_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)
            
    except Exception as e:
        print(f"Error updating name mapping: {e}")

def check_company_name_match(query_text):
    """Check if any part of the query matches a known company name with improved partial matching"""
    mapping = get_name_to_ticker_mapping()
    query_lower = query_text.lower()
    query_nospace = re.sub(r'\s+', '', query_lower)
    found_tickers = []
    
    # Check for direct matches with company names
    for name, ticker in mapping.items():
        if name in query_lower or query_nospace == re.sub(r'\s+', '', name):
            if ticker not in found_tickers:
                found_tickers.append(ticker)
                
                # Also add with different suffixes
                # Prioritize .NS for Indian stocks
                if not ticker.endswith('.NS'):
                    found_tickers.append(ticker + '.NS')
                if not ticker.endswith('.BO'):
                    found_tickers.append(ticker + '.BO')
    
    # Check for partial matches (both beginning of name and significant words)
    name_parts = query_lower.split()
    for name, ticker in mapping.items():
        name_lower = name.lower()
        
        # Check if the query is at the beginning of a company name
        if name_lower.startswith(query_lower) or any(name_lower.startswith(part) for part in name_parts if len(part) >= 3):
            if ticker not in found_tickers:
                found_tickers.append(ticker)
                
                # Prioritize .NS for Indian stocks
                if not ticker.endswith('.NS'):
                    found_tickers.append(ticker + '.NS')
                if not ticker.endswith('.BO'):
                    found_tickers.append(ticker + '.BO')
        
        # Check for significant word matches
        words = query_lower.split()
        name_words = name_lower.split()
        
        # Check if important parts of the query match parts of the company name
        significant_match = False
        for word in words:
            if len(word) >= 4:  # Only check significant words
                if word in name_words or any(name_word.startswith(word) for name_word in name_words):
                    significant_match = True
                    break
        
        if significant_match and ticker not in found_tickers:
            found_tickers.append(ticker)
            
            # Prioritize .NS for Indian stocks
            if not ticker.endswith('.NS'):
                found_tickers.append(ticker + '.NS')
            if not ticker.endswith('.BO'):
                found_tickers.append(ticker + '.BO')
    
    return found_tickers

def handle_conversational_flow(user_message, context):
    """Handle basic conversational elements like greetings and farewells"""
    query_lower = user_message.lower()
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|greetings)\b', query_lower):
        return {
            "message": "Hello! I'm BloomBERT, your stock information assistant. How can I help you with stock information today?",
            "source": ""
        }
    
    # Farewells
    if re.search(r'\b(bye|goodbye|see you|talk to you later)\b', query_lower):
        return {
            "message": "BloomBERT >> \n\n Goodbye! Feel free to return anytime you need stock information.",
            "source": ""
        }
    
    # Thanks/Appreciation
    if re.search(r'\b(thanks|thank you|appreciate|helpful)\b', query_lower):
        return {
            "message": "BloomBERT >> \n\n You're welcome! Is there anything else you'd like to know about stocks?",
            "source": ""
        }
    
    # Help request
    if re.search(r'\b(help|what can you do|capabilities)\b', query_lower):
        help_message = """
I can help you with:
- Stock price information (e.g., "What's the price of TCS?")
- Company details (e.g., "Tell me about Reliance Industries")
- Market information (e.g., "How is HDFC Bank performing?")
- Recent stock news (e.g., "Any news about Tata Motors?")
- IPO information (e.g., "Any upcoming IPOs?")

You can ask follow-up questions about a company after I've provided information.
        """
        return {
            "message": help_message.strip(),
            "source": ""
        }
    
    # Default - no conversational match
    return None

def fetch_company_description(ticker, company_name):
    """Fetch company description from various sources with improved error handling"""
    try:
        # Try to get description from Yahoo Finance
        url = f"https://finance.yahoo.com/quote/{ticker}/profile"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        # Add timeout to prevent hanging
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the company description in the profile section
            description_section = soup.select_one('p[class*="description"]')
            if description_section and description_section.text.strip():
                return description_section.text.strip()
        
        # Fallback: Try to get from stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if 'longBusinessSummary' in info and info['longBusinessSummary']:
            return info['longBusinessSummary']
        
        # Second fallback: Create generic description
        sector = info.get('sector', 'various sectors')
        industry = info.get('industry', 'industries')
        country = info.get('country', 'the market')
        
        return f"{company_name} is a company operating in the {sector} sector, specifically in {industry}, based in {country}."
    except Exception as e:
        print(f"Error fetching company description: {e}")
        return f"{company_name} is a publicly traded company listed under the ticker {ticker}."

def create_company_info_template(ticker, info, description=None):
    """Create a standardized company information template with improved formatting"""
    template = f"## {info.get('longName', ticker)}\n\n"
    
    # Add company description if available
    if description:
        template += f"{description}\n\n"
    elif 'longBusinessSummary' in info:
        template += f"{info['longBusinessSummary']}\n\n"
    
    # Add basic company details
    template += "### Company Details\n"
    details = []
    
    if 'sector' in info and info['sector']:
        details.append(f"- **Sector:** {info['sector']}")
    if 'industry' in info and info['industry']:
        details.append(f"- **Industry:** {info['industry']}")
    if 'country' in info and info['country']:
        details.append(f"- **Country:** {info['country']}")
    if 'website' in info and info['website']:
        details.append(f"- **Website:** {info['website']}")
    if 'fullTimeEmployees' in info and info['fullTimeEmployees']:
        details.append(f"- **Employees:** {info['fullTimeEmployees']:,}")
    
    template += "\n".join(details) + "\n" if details else "No detailed information available.\n"
    
    # Add financial highlights if available
    template += "\n### Financial Highlights\n"
    financials = []
    
    if 'marketCap' in info and info['marketCap']:
        market_cap = info['marketCap']
        if market_cap >= 1_000_000_000:
            financials.append(f"- **Market Cap:** ₹{market_cap/1_000_000_000:.2f} billion")
        else:
            financials.append(f"- **Market Cap:** ₹{market_cap/1_000_000:.2f} million")
            
    if 'trailingPE' in info and info['trailingPE']:
        financials.append(f"- **P/E Ratio:** {info['trailingPE']:.2f}")
        
    if 'dividendYield' in info and info['dividendYield']:
        financials.append(f"- **Dividend Yield:** {info['dividendYield']*100:.2f}%")
        
    if 'fiftyTwoWeekHigh' in info and info['fiftyTwoWeekHigh']:
        financials.append(f"- **52-Week High:** ₹{info['fiftyTwoWeekHigh']:.2f}")
        
    if 'fiftyTwoWeekLow' in info and info['fiftyTwoWeekLow']:
        financials.append(f"- **52-Week Low:** ₹{info['fiftyTwoWeekLow']:.2f}")
    
    template += "\n".join(financials) + "\n" if financials else "No financial information available.\n"
    
    # Add data timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template += f"\nData as of: {current_time}"
    
    return template

def get_stock_news(ticker):
    """Get news for a specific stock with improved error handling and formatting."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return {
                "message": f"BloomBERT >> \n\n No recent news found for {ticker}.",
                "source": ""
            }
            
        # Format news items
        response = f"## Latest news for {ticker}\n\n"
        
        # Limit to 3 news items to keep response concise
        for i, item in enumerate(news[:3], 1):
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown source')
            
            # Convert timestamp to readable date
            publish_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
            date_str = publish_time.strftime('%d %b %Y')
            
            response += f"### {i}. {title}\n"
            response += f"*{publisher} - {date_str}*\n\n"
            
            # Add a short summary if available
            if 'summary' in item and item['summary']:
                summary = item['summary']
                # Truncate long summaries
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                response += f"{summary}\n\n"
        
        response += f"For more news, visit: https://finance.yahoo.com/quote/{ticker}"
        
        return {
            "message": response,
            "source": f"https://finance.yahoo.com/quote/{ticker}/news"
        }
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return {
            "message": f"BloomBERT >> \n\n Sorry, I couldn't retrieve news for {ticker}.",
            "source": ""
        }

def extract_possible_tickers(user_input, conversation_context):
    """Extract and generate possible ticker symbols from user input with improved company name handling"""
    query_lower = user_input.lower()
    possible_tickers = []
    
    # Check if this is a follow-up question about the current company
    if conversation_context.is_follow_up_question(user_input):
        company_context = conversation_context.get_company_context()
        if company_context:
            current_ticker = company_context['ticker']
            print(f"Detected follow-up question about {current_ticker}")
            possible_tickers.append(current_ticker)
            # Add it as first priority
            return possible_tickers
    
    # Clean input - remove common words that aren't part of company names
    common_words = ['price', 'share', 'stock', 'value', 'market', 'cap', 'info', 'about', 'tell', 'me', 'what', 'is', 'the', 'of']
    query_words = query_lower.split()
    filtered_words = [word for word in query_words if word not in common_words]
    cleaned_query = ' '.join(filtered_words).strip()
    
    # First try the whole phrase (to handle multi-word company names)
    # This prioritizes full company names like "tata motors" -> "TATAMOTORS.NS"
    if cleaned_query:
        possible_tickers.extend([
            cleaned_query.upper(),
            cleaned_query.upper() + '.NS',
            cleaned_query.replace(' ', '').upper(),
            cleaned_query.replace(' ', '').upper() + '.NS'
        ])
    
    # Try each possible combination of words in the query
    words = cleaned_query.split()
    if len(words) > 1:
        # Try combinations of words
        for i in range(1, len(words) + 1):
            for j in range(len(words) - i + 1):
                phrase = ' '.join(words[j:j+i])
                
                # Skip single common words
                if i == 1 and len(phrase) < 3:
                    continue
                    
                # Add with and without spaces
                no_space_phrase = phrase.replace(' ', '')
                
                # Add variations (with/without suffixes)
                possible_tickers.extend([
                    phrase.upper(),
                    phrase.upper() + '.NS',
                    phrase.upper() + '.BO',
                    no_space_phrase.upper(),
                    no_space_phrase.upper() + '.NS',
                    no_space_phrase.upper() + '.BO'
                ])
    
    # Specific handling for common Indian companies
    indian_companies = {
        'tata motors': ['TATAMOTORS', 'TATAMOTORS.NS', 'TATAMOTORS.BO'],
        'tata': ['TATAMOTORS', 'TCS', 'TATASTEEL', 'TATACHEM', 'TATAPOWER'],
        'reliance': ['RELIANCE', 'RELIANCE.NS', 'RELIANCEPP', 'RELINFRA'],
        'indusind bank': ['INDUSINDBK', 'INDUSINDBK.NS'],
        'indusind': ['INDUSINDBK', 'INDUSINDBK.NS'],
        'hdfc bank': ['HDFCBANK', 'HDFCBANK.NS'],
        'hdfc': ['HDFCBANK', 'HDFC', 'HDFCLIFE', 'HDFCAMC'],
        'state bank': ['SBIN', 'SBIN.NS'],
        'sbi': ['SBIN', 'SBIN.NS'],
        'zeel': ['ZEEL', 'ZEEL.BO'],
        'infosys': ['INFY', 'INFY.NS'],
        'icici bank': ['ICICIBANK', 'ICICIBANK.NS'],
        'NVIDIA': ['NVDA', 'NVIDIA'],
        'axis bank': ['AXISBANK', 'AXISBANK.NS'],
        'reliance power': ['RELIANCEPOWER', 'RELIANCEPOWER.NS'],
        'adani': ['ADANIPORTS', 'ADANIGREEN', 'ADANIENT'],
        'adani ports': ['ADANIPORTS', 'ADANIPORTS.NS'],
        'adani green': ['ADANIGREEN', 'ADANIGREEN.NS'],
        'adani enterprises': ['ADANIENT', 'ADANIENT.NS'],
        'adani wilmar': ['ADANIWILMAR', 'ADANIWILMAR.NS']
    }
    
    # Check if the query matches any known company
    for company_name, tickers in indian_companies.items():
        if company_name in query_lower:
            possible_tickers.extend(tickers)
    
    # Remove duplicates while preserving order
    unique_tickers = []
    seen = set()
    for ticker in possible_tickers:
        if ticker not in seen:
            unique_tickers.append(ticker)
            seen.add(ticker)
    
    return unique_tickers

def verify_ticker_with_company_name(ticker, user_input):
    """Verify if a ticker is valid and determine confidence based on name match with user input"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # If we get a longName, we have a valid ticker
        if 'longName' in info:
            company_name = info['longName'].lower()
            user_input_lower = user_input.lower()
            
            # Calculate name similarity
            user_input_words = set(user_input_lower.split())
            company_name_words = set(company_name.split())
            
            # If the company name is in the user input or vice versa
            if user_input_lower in company_name or any(word in company_name for word in user_input_words if len(word) > 3):
                return True, info, 2  # High confidence
            
            # Some word overlap
            if any(word in company_name_words for word in user_input_words if len(word) > 3):
                return True, info, 1  # Medium confidence
                
            # Valid ticker but low match confidence
            return True, info, 0  # Low confidence
                
    except Exception as e:
        print(f"Error verifying ticker {ticker}: {str(e)}")
        pass
    
    return False, None, 0

def identify_requested_attributes(query_lower):
    """Identify what stock attributes the user is asking about with improved precision"""
    requested_attrs = []
    
    # Price related attributes
    if any(term in query_lower for term in ['price', 'rate', 'value', 'cost', 'worth', 'current price', 'stock price']):
        requested_attrs.append('price')
    
    # Market cap
    if any(term in query_lower for term in ['market cap', 'marketcap', 'market capitalization', 'size', 'valuation']):
        requested_attrs.append('market_cap')
    
    # Opening price
    if any(term in query_lower for term in ['open', 'opening', 'start', 'began', 'morning price']):
        requested_attrs.append('open')
    
    # High/Low prices
    if any(term in query_lower for term in ['high', 'max', 'highest', 'peak', 'maximum']):
        requested_attrs.append('high')
    if any(term in query_lower for term in ['low', 'min', 'lowest', 'bottom', 'minimum']):
        requested_attrs.append('low')
    
    # Volume
    if any(term in query_lower for term in ['volume', 'traded', 'liquidity', 'trading volume']):
        requested_attrs.append('volume')
    
    # Sector/Industry
    if any(term in query_lower for term in ['sector', 'industry', 'field', 'business', 'company type']):
        requested_attrs.append('sector')
    
    # PE ratio
    if any(term in query_lower for term in ['pe', 'pe ratio', 'p/e', 'price to earning', 'earnings ratio', 'price-earnings']):
        requested_attrs.append('pe')
    
    # Dividend
    if any(term in query_lower for term in ['dividend', 'yield', 'payout', 'distribution']):
        requested_attrs.append('dividend')
    
    # General company info
    if any(term in query_lower for term in ['about', 'who is', 'what is', 'tell me about', 'details', 'information', 'profile']):
        requested_attrs.append('description')
    
    # Performance/History
    if any(term in query_lower for term in ['perform', 'history', 'chart', 'trend', 'growth', 'historical']):
        requested_attrs.append('performance')
    
    # 52 week high/low
    if '52' in query_lower or 'fifty two' in query_lower or 'year high' in query_lower or 'year low' in query_lower:
        requested_attrs.append('52week')
    
    # If no specific attributes requested, show summary
    if not requested_attrs:
        requested_attrs = ['summary']
    
    return requested_attrs

def identify_requested_attributes(query_lower):
    """Identify what stock attributes the user is asking about"""
    requested_attrs = []
    
    # Price related attributes
    if any(term in query_lower for term in ['price', 'rate', 'value', 'cost', 'worth']):
        requested_attrs.append('price')
    
    # Market cap
    if any(term in query_lower for term in ['market cap', 'marketcap', 'market capitalization', 'size']):
        requested_attrs.append('market_cap')
    
    # Opening price
    if any(term in query_lower for term in ['open', 'opening', 'start', 'began']):
        requested_attrs.append('open')
    
    # High/Low prices
    if any(term in query_lower for term in ['high', 'max', 'highest', 'peak']):
        requested_attrs.append('high')
    if any(term in query_lower for term in ['low', 'min', 'lowest', 'bottom']):
        requested_attrs.append('low')
    
    # Volume
    if any(term in query_lower for term in ['volume', 'traded', 'liquidity']):
        requested_attrs.append('volume')
    
    # Sector/Industry
    if any(term in query_lower for term in ['sector', 'industry', 'field', 'business']):
        requested_attrs.append('sector')
    
    # PE ratio
    if any(term in query_lower for term in ['pe', 'p/e', 'price to earning', 'earnings ratio', 'pe ratio', 'p/e ratio']):
        requested_attrs.append('pe')
    
    # Dividend
    if any(term in query_lower for term in ['dividend', 'yield', 'payout']):
        requested_attrs.append('dividend')
    
    # General company info
    if any(term in query_lower for term in ['about', 'who is', 'what is', 'tell me about', 'details', 'information']):
        requested_attrs.append('description')
    
    # Performance/History
    if any(term in query_lower for term in ['perform', 'history', 'chart', 'trend', 'growth']):
        requested_attrs.append('performance')
    
    # 52 week high/low
    if '52' in query_lower or '52-week' in query_lower or 'fifty two' in query_lower or 'year high' in query_lower or 'year low' in query_lower:
        requested_attrs.append('52week')
    
    # If no specific attributes requested, show summary
    if not requested_attrs:
        requested_attrs = ['summary']
    
    return requested_attrs

def process_query(user_message, conversation_context=None):
    """Process user stock queries with context handling and conversational flow."""
    try:
        # Initialize conversation_context if None
        if conversation_context is None:
            conversation_context = ConversationContext()
            
        # Clean up user input
        user_input = user_message.strip()
        query_lower = user_input.lower()
        
        # FIRST PRIORITY: Check restrictions before anything else
        restriction_check = check_restriction(user_input)
        if restriction_check != "Access Granted: No restricted keywords detected.":
            return {
                "message": restriction_check,
                "source": ""
            }
        
        # Check for conversational flow (greetings, farewells, etc.)
        conv_response = handle_conversational_flow(user_input, conversation_context)
        if conv_response:
            # Add to conversation history and return response
            conversation_context.add_interaction(user_input, conv_response["message"])
            return conv_response
        
        # Check if it's an IPO related query
        if is_ipo_query(query_lower):
            response = handle_ipo_query(user_input)
            conversation_context.add_interaction(user_input, response["message"])
            return response
            
        # Check if it's a news query
        if 'news' in query_lower:
            # Try to extract ticker if present
            possible_tickers = extract_possible_tickers(user_input, conversation_context)
            
            # If there's a possible ticker, try to get news for it
            if possible_tickers:
                for ticker in possible_tickers[:3]:  # Try the top 3 most likely tickers
                    news_result = get_stock_news(ticker)
                    if "couldn't retrieve news" not in news_result["message"]:
                        conversation_context.add_interaction(user_input, news_result["message"])
                        return news_result
            
            # General market news as fallback
            news_result = get_market_news()
            conversation_context.add_interaction(user_input, news_result["message"])
            return news_result
        
        # Identify what attributes the user is asking about
        requested_attrs = identify_requested_attributes(query_lower)
        print(f"Requested attributes: {requested_attrs}")

        # Generate possible tickers from user input and context
        possible_tickers = extract_possible_tickers(user_input, conversation_context)
        print(f"Possible tickers to check: {possible_tickers}")
        
        # Keep track of all valid tickers and their confidence scores
        valid_results = []
        
        # Try each possible ticker
        for ticker in possible_tickers:
            try:
                print(f"Attempting to fetch data for: {ticker}")
                # Verify ticker is valid with confidence score
                is_valid, info, confidence = verify_ticker_with_company_name(ticker, user_input)
                
                if is_valid and info:
                    valid_results.append((ticker, info, confidence))
            except Exception as e:
                print(f"Error with ticker {ticker}: {e}")
                continue
        
        # Sort valid results by confidence (highest first)
        valid_results.sort(key=lambda x: x[2], reverse=True)
        
        # Process the best match if any
        if valid_results:
            ticker, info, confidence = valid_results[0]
            try:
                stock = yf.Ticker(ticker)
                
                # Get price data
                hist = stock.history(period="1d")
                if not hist.empty:
                    company_name = info['longName']
                    
                    # Update the name mapping when we successfully identify a ticker
                    update_name_mapping(company_name, ticker)
                    
                    # Get current price data
                    current_price = hist["Close"].iloc[-1] if "Close" in hist else None
                    open_price = hist["Open"].iloc[-1] if "Open" in hist else None
                    
                    # Check if this is the first mention of this company
                    company_context = conversation_context.get_company_context()
                    is_first_mention = not company_context or company_context['ticker'] != ticker
                    
                    # Format the response based on what was asked
                    # Initial response always includes the company name and ticker
                    response = f"**{company_name} ({ticker})**"
                    
                    # For first mention, add brief company info
                    if is_first_mention:
                        sector = info.get('sector', None)
                        industry = info.get('industry', None)
                        
                        # Add brief company context
                        basic_info = []
                        if sector:
                            basic_info.append(f"Sector: {sector}")
                        if industry:
                            basic_info.append(f"Industry: {industry}")
                            
                        if basic_info:
                            response += "\n" + ", ".join(basic_info)
                    
                    # Now add the specifically requested information
                    response_sections = []
                    
                    # Handle description request separately as it needs longer format
                    if 'description' in requested_attrs:
                        description = fetch_company_description(ticker, company_name)
                        if description:
                            response += f"\n\n{description}"
                        
                    # Handle other focused attribute requests
                    if 'price' in requested_attrs and current_price is not None:
                        response_sections.append(f"Current Price: ₹{current_price:.2f}")
                    
                    if 'open' in requested_attrs and open_price is not None:
                        response_sections.append(f"Opening Price: ₹{open_price:.2f}")
                    
                    if 'high' in requested_attrs and "High" in hist:
                        response_sections.append(f"Day High: ₹{hist['High'].iloc[-1]:.2f}")
                    
                    if 'low' in requested_attrs and "Low" in hist:
                        response_sections.append(f"Day Low: ₹{hist['Low'].iloc[-1]:.2f}")
                    
                    if 'volume' in requested_attrs and "Volume" in hist:
                        volume = int(hist['Volume'].iloc[-1])
                        response_sections.append(f"Volume: {volume:,}")
                    
                    if 'market_cap' in requested_attrs and 'marketCap' in info and info['marketCap']:
                        market_cap = info['marketCap']
                        if market_cap >= 1_000_000_000:
                            response_sections.append(f"Market Cap: ₹{market_cap/1_000_000_000:.2f} billion")
                        else:
                            response_sections.append(f"Market Cap: ₹{market_cap/1_000_000:.2f} million")
                    
                    if 'sector' in requested_attrs:
                        if 'sector' in info and info['sector']:
                            response_sections.append(f"Sector: {info['sector']}")
                        if 'industry' in info and info['industry']:
                            response_sections.append(f"Industry: {info['industry']}")
                    
                    if 'pe' in requested_attrs and 'trailingPE' in info and info['trailingPE']:
                        response_sections.append(f"P/E Ratio: {info['trailingPE']:.2f}")
                    
                    if 'dividend' in requested_attrs and 'dividendYield' in info and info['dividendYield']:
                        response_sections.append(f"Dividend Yield: {info['dividendYield']*100:.2f}%")
                    
                    if '52week' in requested_attrs:
                        if 'fiftyTwoWeekHigh' in info and info['fiftyTwoWeekHigh']:
                            response_sections.append(f"52-Week High: ₹{info['fiftyTwoWeekHigh']:.2f}")
                        if 'fiftyTwoWeekLow' in info and info['fiftyTwoWeekLow']:
                            response_sections.append(f"52-Week Low: ₹{info['fiftyTwoWeekLow']:.2f}")
                    
                    # For summary, show price and change at minimum
                    if 'summary' in requested_attrs and current_price is not None:
                        if not any(x in requested_attrs for x in ['price']):
                            response_sections.append(f"Current Price: ₹{current_price:.2f}")
                            
                        if open_price is not None:
                            change = current_price - open_price
                            change_pct = (change / open_price) * 100
                            direction = "▲" if change >= 0 else "▼"
                            response_sections.append(f"Change: {direction} ₹{abs(change):.2f} ({change_pct:.2f}%)")
                    
                    # Add response sections if available
                    if response_sections:
                        response += "\n" + "\n".join(response_sections)
                    
                    # Add timestamp
                    current_time = datetime.now().strftime("%d %b %Y, %H:%M:%S")
                    response += f"\n\n*Data as of: {current_time}*"
                    
                    # Update the company context with this company
                    description = fetch_company_description(ticker, company_name) if 'longBusinessSummary' not in info else info['longBusinessSummary']
                    conversation_context.set_company_context(
                        ticker, 
                        company_name, 
                        description,
                        info.get('sector'),
                        info.get('industry')
                    )
                    
                    # Record this interaction
                    conversation_context.add_interaction(user_input, response)
                    
                    print(f"Successfully fetched data for: {ticker}")
                    return {
                        "message": response.strip(),
                        "source": f"https://finance.yahoo.com/quote/{ticker}"
                    }
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
            
        # If no ticker worked, give a helpful message
        company_context = conversation_context.get_company_context()
        #continue
        
        if conversation_context.is_follow_up_question(user_input) and company_context:
            response_message = f"BloomBERT >> \n\n I'm not sure what specific information you're asking about {company_context['name']}. You can ask about its price, sector, P/E ratio, dividend yield, or other financial metrics."
        else:
            response_message = "BloomBERT >> \n\n I couldn't find stock information for your query. Please provide a valid company name or stock ticker (e.g., 'Tata Motors' or 'TATAMOTORS')."
        
        conversation_context.add_interaction(user_input, response_message)
        return {
            "message": response_message,
            "source": ""
        }
                
    except Exception as e:
        print(f"Overall error in process_query: {e}")
        error_message = "BloomBERT >> \n\n Sorry, there was an error processing your request. Please try again with a valid company name or stock ticker."
        conversation_context.add_interaction(user_input, error_message)
        return {
            "message": error_message,
            "source": ""
        }

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return responses"""
    try:
        # Get the message from the request
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Initialize conversation context if not already done
        global conversation_context
        if conversation_context is None:
            conversation_context = ConversationContext()
        
        # Process the query
        response = process_query(user_message, conversation_context)
        
        return jsonify({
            "response": response["message"],
            "source": response["source"]
        })
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route('/api/stock/price/<ticker>')
def stock_price(ticker):
    """Get current price for a stock ticker"""
    try:
        # Sanitize ticker input
        ticker = ticker.strip().upper()
        
        # Get stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        
        if hist.empty:
            return jsonify({"error": "No data available for this ticker"}), 404
        
        # Get the latest price
        current_price = hist["Close"].iloc[-1] if "Close" in hist else None
        
        if current_price is None:
            return jsonify({"error": "Price data not available"}), 404
        
        # Get additional info if available
        info = stock.info
        company_name = info.get('longName', ticker) if 'longName' in info else ticker
        
        return jsonify({
            "ticker": ticker,
            "company": company_name,
            "price": float(current_price),
            "currency": "INR",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting stock price: {str(e)}")
        return jsonify({"error": "Failed to retrieve stock information"}), 500

@app.route('/api/market/news')
def market_news():
    """Get latest market news"""
    try:
        news_data = get_market_news()
        return jsonify({
            "news": news_data["message"],
            "source": news_data["source"]
        })
    except Exception as e:
        print(f"Error getting market news: {str(e)}")
        return jsonify({"error": "Failed to retrieve market news"}), 500

@app.route('/api/stock/news/<ticker>')
def stock_news(ticker):
    """Get news for a specific stock"""
    try:
        # Sanitize ticker input
        ticker = ticker.strip().upper()
        
        news_data = get_stock_news(ticker)
        return jsonify({
            "ticker": ticker,
            "news": news_data["message"],
            "source": news_data["source"]
        })
    except Exception as e:
        print(f"Error getting stock news: {str(e)}")
        return jsonify({"error": "Failed to retrieve stock news"}), 500

@app.route('/api/ipo/upcoming')
def upcoming_ipos():
    """Get upcoming IPO information"""
    try:
        ipo_data = fetch_ipo_data()
        return jsonify(ipo_data)
    except Exception as e:
        print(f"Error getting IPO data: {str(e)}")
        return jsonify({"error": "Failed to retrieve IPO information"}), 500

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        message = request.json.get('message')
        company_info = search_company(message)

        if company_info:
            ticker = company_info['symbol']
            name = company_info['longName']
            sector = company_info['sector']
            query = company_info['user_query']

            if query == "market cap":
                detail = company_info['marketCap']
                label = "Market Cap"
            elif query == "volume":
                detail = company_info['volume']
                label = "Volume"
            elif query == "price":
                detail = company_info['currentPrice']
                label = "Current Price"
            else:
                detail = f"Sector: {sector}"
                label = "Basic Info"

            response_text = (
                f"BloomBERT >> \n\n"
                f"{ticker} is owned by {name}, working in the {sector.lower()} sector.\n"
                f"\n {label}: {detail}"
            )

            return jsonify({
                "message": response_text,
                "source": f"https://finance.yahoo.com/quote/{ticker}"
            })

        else:
            return jsonify({
                #"message": "BloomBERT >> \n\n I couldn't identify the company. Please use a valid company name or ticker."
                "message": "BloomBERT >> \n\n I couldn't connect... Recheck connection."
            })

    except Exception as e:
        print("Error in /get_response:", e)
        return jsonify({
            "message": "BloomBERT >> \n\n Sorry, there was an error processing your request. Please try again."
        }), 500


@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "BloomBERT Stock Chatbot",
        "timestamp": datetime.now().isoformat()
    })

def start_initialization():
    global models_ready
    Thread(target=initialize_models, daemon=True).start()

# Then add this route that will trigger the initialization
@app.route('/initialize', methods=['GET'])
def trigger_initialization():
    """Endpoint to trigger model initialization"""
    global models_ready
    if not models_ready:
        start_initialization()
        return jsonify({"status": "Initialization started"})
    return jsonify({"status": "Models already initialized or initializing"})

# Finally, start the initialization when the app starts
with app.app_context():
    start_initialization()

if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=True, host="0.0.0.0", port=10000)