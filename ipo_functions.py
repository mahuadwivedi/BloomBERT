import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime

def fetch_ipo_data():
    #nse india IPO page is a reliable source for Indian IPOs
    try:
        fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        url = "https://finance.yahoo.com/calendar/ipo"
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse IPO information
        ipo_data = {
            "upcoming": [],
            "ongoing": [],
            "past": [],
            "fetch_time": fetch_time
        }
        
        # Process upcoming IPOs
        upcoming_section = soup.find('div', {'id': 'upcoming'})
        if upcoming_section:
            ipo_table = upcoming_section.find('table', class_='responsive')
            if ipo_table:
                rows = ipo_table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        company_name = cols[0].text.strip()
                        open_date = cols[1].text.strip()
                        close_date = cols[2].text.strip()
                        price_range = cols[3].text.strip()
                        lot_size = cols[4].text.strip()
                        issue_size = cols[5].text.strip()
                        
                        ipo_data["upcoming"].append({
                            "company": company_name,
                            "open_date": open_date,
                            "close_date": close_date,
                            "price_range": price_range,
                            "lot_size": lot_size,
                            "issue_size": issue_size
                        })
        
        # Process ongoing IPOs
        ongoing_section = soup.find('div', {'id': 'current'})
        if ongoing_section:
            ipo_table = ongoing_section.find('table', class_='responsive')
            if ipo_table:
                rows = ipo_table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        company_name = cols[0].text.strip()
                        open_date = cols[1].text.strip()
                        close_date = cols[2].text.strip()
                        price_range = cols[3].text.strip()
                        lot_size = cols[4].text.strip()
                        issue_size = cols[5].text.strip()
                        
                        ipo_data["ongoing"].append({
                            "company": company_name,
                            "open_date": open_date,
                            "close_date": close_date,
                            "price_range": price_range,
                            "lot_size": lot_size,
                            "issue_size": issue_size
                        })
        
        # Process past IPOs
        past_section = soup.find('div', {'id': 'justConcluded'})
        if past_section:
            ipo_table = past_section.find('table', class_='responsive')
            if ipo_table:
                rows = ipo_table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        company_name = cols[0].text.strip()
                        open_date = cols[1].text.strip()
                        close_date = cols[2].text.strip()
                        price_range = cols[3].text.strip()
                        lot_size = cols[4].text.strip()
                        issue_size = cols[5].text.strip()
                        
                        ipo_data["past"].append({
                            "company": company_name,
                            "open_date": open_date,
                            "close_date": close_date,
                            "price_range": price_range,
                            "lot_size": lot_size,
                            "issue_size": issue_size
                        })
        
        return ipo_data
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch IPO data: {e}")
        return None

def handle_ipo_query(query):
    # Handles user queries related to IPOs. Returns a formatted response with relevant IPO information.
    query = query.lower()
    timestamp = ipo_data.get("fetch_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Fetch IPO data
    ipo_data = fetch_ipo_data()
    if not ipo_data:
        return {
            "message": "I couldn't fetch the latest IPO information. Please try again later.",
            "source": "https://finance.yahoo.com/calendar/ipo"
        }
    
    # Check for specific company IPO
    company_match = re.search(r"(?:about|when is|details of|info on)\s+([a-zA-Z0-9\s]+)(?:'s)?\s+ipo", query)
    if company_match:
        company_name = company_match.group(1).strip().lower()
        
        # Search for the company in all categories
        for category in ["upcoming", "ongoing", "past"]:
            for ipo in ipo_data[category]:
                if company_name in ipo["company"].lower():
                    status = category
                    if status == "past":
                        status_message = f"{ipo['company']} IPO was open from {ipo['open_date']} to {ipo['close_date']}."
                    elif status == "ongoing":
                        status_message = f"{ipo['company']} IPO is currently open from {ipo['open_date']} to {ipo['close_date']}."
                    else:  # upcoming
                        status_message = f"{ipo['company']} IPO will open from {ipo['open_date']} to {ipo['close_date']}."
                    
                    response = f"{status_message}\n"
                    response += f"Price Range: {ipo['price_range']}\n"
                    response += f"Lot Size: {ipo['lot_size']}\n"
                    response += f"Issue Size: {ipo['issue_size']}"
                    
                    return {
                        "message": response,
                        "source": "https://finance.yahoo.com/calendar/ipo"
                    }
        
        return {
            "message": f"I couldn't find any information about {company_match.group(1)}'s IPO. It might not be listed yet or might have been listed under a different name.",
            "source": "https://finance.yahoo.com/calendar/ipo"
        }
    
    # Check for last/recent IPO query
    if any(term in query for term in ["last ipo", "recent ipo", "latest ipo", "previous ipo"]):
        if ipo_data["past"]:
            last_ipo = ipo_data["past"][0]
            response = f"The most recent IPO was {last_ipo['company']}.\n"
            response += f"It was open from {last_ipo['open_date']} to {last_ipo['close_date']}.\n"
            response += f"Price Range: {last_ipo['price_range']}\n"
            response += f"Issue Size: {last_ipo['issue_size']}"
            
            return {
                "message": response+ f"\n\nData as of: {timestamp}",
                "source": "https://finance.yahoo.com/calendar/ipo"
            }
    
    # Check for upcoming IPO query
    if any(term in query for term in ["upcoming ipo", "next ipo", "future ipo", "coming ipo"]):
        if ipo_data["upcoming"]:
            next_ipo = ipo_data["upcoming"][0]
            response = f"The next upcoming IPO is {next_ipo['company']}.\n"
            response += f"It will open from {next_ipo['open_date']} to {next_ipo['close_date']}.\n"
            response += f"Price Range: {next_ipo['price_range']}\n"
            response += f"Issue Size: {next_ipo['issue_size']}"
            
            return {
                "message": response,
                "source": "https://finance.yahoo.com/calendar/ipo"
            }
    
    # Check for ongoing IPO query
    if any(term in query for term in ["ongoing ipo", "current ipo", "open ipo", "active ipo"]):
        if ipo_data["ongoing"]:
            ongoing_ipos = ipo_data["ongoing"]
            if len(ongoing_ipos) == 1:
                ipo = ongoing_ipos[0]
                response = f"Currently, {ipo['company']} IPO is open.\n"
                response += f"It opened on {ipo['open_date']} and will close on {ipo['close_date']}.\n"
                response += f"Price Range: {ipo['price_range']}\n"
                response += f"Issue Size: {ipo['issue_size']}"
            else:
                response = "Currently open IPOs:\n\n"
                for ipo in ongoing_ipos:
                    response += f"- {ipo['company']}: Open till {ipo['close_date']}, Price Range: {ipo['price_range']}\n"
            
            return {
                "message": response,
                "source": "https://finance.yahoo.com/calendar/ipo"
            }
        else:
            return {
                "message": "There are no IPOs currently open for subscription.",
                "source": "https://finance.yahoo.com/calendar/ipo"
            }
    
    # General IPO list query
    if any(term in query for term in ["list ipo", "all ipo", "show ipo"]):
        response = "IPO Calendar Summary:\n\n"
        
        if ipo_data["ongoing"]:
            response += "Currently Open IPOs:\n"
            for ipo in ipo_data["ongoing"]:
                response += f"- {ipo['company']} (Till {ipo['close_date']})\n"
            response += "\n"
        
        if ipo_data["upcoming"]:
            response += "Upcoming IPOs:\n"
            for ipo in ipo_data["upcoming"][:3]:  # Show only first 3
                response += f"- {ipo['company']} (Opens {ipo['open_date']})\n"
            response += "\n"
        
        if ipo_data["past"]:
            response += "Recently Concluded IPOs:\n"
            for ipo in ipo_data["past"][:3]:  # Show only first 3
                response += f"- {ipo['company']} (Closed {ipo['close_date']})\n"
        
        return {
            "message": response,
            "source": "https://finance.yahoo.com/calendar/ipo"
        }
    
    # Default to a general IPO information response
    response = "I can help you with IPO information. Try asking me about:\n"
    response += "- Specific IPOs (e.g., 'When is Honasa's IPO?')\n"
    response += "- Recent IPOs (e.g., 'What was the last IPO?')\n"
    response += "- Upcoming IPOs (e.g., 'What are the upcoming IPOs?')\n"
    response += "- Currently open IPOs (e.g., 'Which IPOs are open now?')"
    
    return {
        "message": response,
        "source": "https://finance.yahoo.com/calendar/ipo"
    }

# Function to detect if a query is IPO-related
def is_ipo_query(query):
    """
    Detects if a user query is related to IPOs.
    """
    query = query.lower()
    ipo_keywords = [
        "ipo", "initial public offering", "public issue", 
        "new listing", "stock launch", "company launch",
        "going public", "subscribe", "gmp", "grey market premium"
    ]
    
    return any(keyword in query for keyword in ipo_keywords)