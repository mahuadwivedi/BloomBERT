import datetime
# Create a new file called ticker_buffer.py with this content:

class TickerBuffer:
    """Buffer to store information about tickers mentioned in a conversation."""
    
    def __init__(self):
        # Dictionary to store ticker info: {'TATAMOTORS.NS': {'description': '...', 'last_accessed': timestamp}}
        self.ticker_cache = {}
        # Track the most recently mentioned ticker for context in follow-up questions
        self.current_ticker = None
        
    def add_ticker(self, ticker, description, company_name):
        """Add or update a ticker in the buffer."""
        self.ticker_cache[ticker] = {
            'description': description,
            'company_name': company_name,
            'last_accessed': datetime.now()
        }
        self.current_ticker = ticker
        
    def get_description(self, ticker):
        """Get description for a ticker if available."""
        if ticker in self.ticker_cache:
            self.current_ticker = ticker  # Update the current ticker
            return self.ticker_cache[ticker]['description']
        return None
    
    def get_current_ticker(self):
        """Get the most recently mentioned ticker."""
        return self.current_ticker
    
    def is_first_mention(self, ticker):
        """Check if this is the first mention of the ticker in this conversation."""
        return ticker not in self.ticker_cache
    
    def get_company_info(self, ticker=None):
        """Get company info for the specified ticker or current ticker."""
        ticker_to_use = ticker or self.current_ticker
        if ticker_to_use and ticker_to_use in self.ticker_cache:
            return self.ticker_cache[ticker_to_use]
        return None