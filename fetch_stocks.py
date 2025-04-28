import os
import pandas as pd
import yfinance as yf

def get_all_indian_stock_symbols():
    """Fetch all NSE and BSE stock symbols and save as indian_stock.csv."""
    print("Fetching all Indian stock symbols, please wait...")
    
    # Fetch NSE stocks
    try:
        nse_tickers = pd.read_html("https://www.nseindia.com/market-data/securities-available-for-trading")[0]
        nse_symbols = nse_tickers["Symbol"].tolist()
        nse_symbols = [symbol + ".NS" for symbol in nse_symbols]  # Append .NS for Yahoo Finance
    except Exception as e:
        print(f"Failed to fetch NSE symbols: {e}")
        nse_symbols = []
    
    # Fetch BSE stocks
    try:
        bse_tickers = pd.read_html("https://www.bseindia.com/markets/equity/EQReports/Equitydebcopy.aspx")[0]
        bse_symbols = bse_tickers["Security Code"].astype(str).tolist()
        bse_symbols = [symbol + ".BO" for symbol in bse_symbols]  # Append .BO for BSE
    except Exception as e:
        print(f"Failed to fetch BSE symbols: {e}")
        bse_symbols = []
    
    # Combine unique stock symbols
    all_symbols = list(set(nse_symbols + bse_symbols))
    if not all_symbols:
        print("No stock symbols fetched. Please check the sources.")
        return
    
    # Fetch stock details using yfinance
    all_stocks = []
    for symbol in all_symbols:
        try:
            stock = yf.Ticker(symbol)
            stock_info = stock.info
            all_stocks.append({
                "Symbol": symbol,
                "Name": stock_info.get("shortName", "N/A"),
                "Exchange": "NSE" if symbol.endswith(".NS") else "BSE"
            })
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_stocks)
    file_path = os.path.join(os.getcwd(), "indian_stock.csv")
    df.to_csv(file_path, index=False)
    
    print(f"All Indian stock symbols saved to {file_path} ({len(df)} entries)")
    return file_path

if __name__ == "__main__":
    get_all_indian_stock_symbols()
