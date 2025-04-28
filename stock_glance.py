import streamlit as st
import yfinance as yf
import plotly.graph_objs as go

# Title
st.set_page_config(page_title="BloomBERT Glance", layout="wide")
st.title("BloomBERT")
st.write("Stock at a Glance")

# Get user input for the stock symbol
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, NVDA):", "")

# Display stock information and price history if a valid ticker is entered
if ticker:
    stock = yf.Ticker(ticker)
    def human_readable(number):
        if number is None:
            return "N/A"
        try:
            number = float(number)
            if abs(number) >= 1_000_000_000:
                return f"{number / 1_000_000_000:.2f} billion"
            elif abs(number) >= 1_000_000:
                return f"{number / 1_000_000:.2f} million"
            return str(number)
        except:
            return str(number)

    def estimate_ebitda(info):
        net_income = info.get('netIncome')
        interest = info.get('interestExpense')
        taxes = info.get('incomeTaxExpense')
        depreciation = info.get('depreciation') or info.get('depreciationAndAmortization')
        # Return "N/A" if core values are missing
        if None in (net_income, interest, taxes, depreciation):
            return "N/A"
        try:
            ebitda = net_income + interest + taxes + depreciation
            return human_readable(ebitda)
        except:
            return "N/A"

    try:
        stock_name = stock.info.get('longName', 'N/A')
        st.subheader(f"{ticker} - {stock_name}")

        # Display stock information
        st.subheader(f"{ticker} - Stock Info")
        st.write(f"**Company Name:** {stock_name}")
        st.write(f"**Company Sector:** {stock.info.get('sector', 'N/A')}")
        st.write(f"**Company Industry:** {stock.info.get('industry', 'N/A')}")
        st.write(f"**Current Price:** {stock.history(period='1d')['Close'].iloc[-1]:.2f}")
        st.write(f"**Market Cap:** {human_readable(stock.info.get('marketCap'))}")
        st.write(f"**Liquidity:** {human_readable(stock.info.get('averageDailyVolume10Day', 'N/A'))}")
        st.write(f"**EBITDA:** {human_readable(estimate_ebitda(stock.info))}")
        st.write(f"**RoE:** {stock.info.get('returnOnEquity', 'N/A')}")
        st.write(f"**RoCE:** {stock.info.get('returnOnCapitalEmployed', 'N/A')}")
        st.write(f"**52-Week High:** ₹{stock.info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.write(f"**52-Week Low:** ₹{stock.info.get('fiftyTwoWeekLow', 'N/A')}")
        st.write(f"**Sales growth (Quarter on quarter):** {stock.info.get('quarterlyRevenueGrowthYOY', 'N/A')}")
        st.write(f"**Sales growth (Year on year):** {stock.info.get('revenueGrowth', 'N/A')}")
        st.write(f"**Beta:** {stock.info.get('beta', 'N/A')}") # Dropdown for time period AFTER ticker is entered
        time_period = st.selectbox(
            "Select Time Period for Graph",
            options=["1d", "1wk", "1mo", "1y", "5y"],
            index=2  # Default to 1 month
        )
        
        # Toggle for graph type
        graph_type = st.radio(
            "Select Graph Type",
            options=["Line", "Candlestick"],
            horizontal=True
        )

        # Get historical market data based on selected period
        hist = stock.history(period=time_period)

        st.subheader(f"Price History ({time_period})")

        fig = go.Figure()

        if graph_type == "Line":
            # Add line trace
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00ff00', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:.2f}<extra></extra>'
            ))
        else:  # Candlestick
            # Add candlestick trace
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                increasing=dict(line=dict(color='#00ff00')),  # Green for increasing
                decreasing=dict(line=dict(color='#ff0000')),  # Red for decreasing
                name='Price'
            ))

        fig.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            xaxis=dict(
                title='Date',
                color='white',
                gridcolor='#444444',
                tickangle=90,
                rangeslider=dict(visible=False)  # Hide range slider for cleaner look
            ),
            yaxis=dict(
                title='Price',
                color='white',
                gridcolor='#444444'
            ),
            font=dict(color='white'),
            hovermode="x unified"
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Failed to load data: {e}")

# Footer
st.caption("Powered by Yahoo Finance & Mahua Dwivedi")