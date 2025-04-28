# Imp:To avoid AI Jailbreak creating restrictions for user input

import regex as re

restricted_keywords = [r"\b\w*where\s?you\s?get\s?information\s?from\w*\b",r"\b\w*where\s?you\s?get\s?data\s?from\w*\b",r"\b\w*where\s?does\s?the\s?AI\s?get\s?its\s?data\s?from\w*\b",r"\b\w*metadata\w*\b",r"\b\w*manipulat\w*\b",r"\b\w*bypass\w*\b",r"\b\w*insider\s?trading\w*\b",r"\b\w*front\s?running\w*\b",r"\b\w*price\s?fixing\w*\b",r"\b\w*pump\s?and\s?dump\w*\b",r"\b\w*spoofing\w*\b",r"\b\w*wash\s?trading\w*\b",r"\b\w*market\s?manipulation\w*\b",r"\b\w*arbitrage\s?abuse\w*\b",r"\b\w*pan\w*\b",r"\b\w*aadhaar\w*\b",r"\b\w*bank\s?account\s?details\w*\b",r"\b\w*demat\s?account\w*\b",r"\b\w*upi\w*\b",r"\b\w*ifsc\s?code\w*\b",r"\b\w*portfolio\s?details\w*\b",r"\b\w*annual\s?report\s?leaks\w*\b",r"\b\w*unpublished\s?financial\s?results\w*\b",r"\b\w*dividend\s?policy\s?before\s?announcement\w*\b",r"\b\w*stock\s?split\s?plans\s?before\s?disclosure\w*\b",r"\b\w*unauthorized\s?buy\s?\/\s?sell\w*\b",r"\b\w*margin\s?trading\s?misuse\w*\b",r"\b\w*options\s?trading\s?hacks\w*\b",r"\b\w*futures\s?manipulation\w*\b",r"\b\w*leverage\s?abuse\w*\b",r"\b\w*nse\w*\b",r"\b\w*bse\w*\b",r"\b\w*sebi\s?restricted\s?stocks\w*\b",r"\b\w*circuit\s?limits\s?breach\w*\b",r"\b\w*bulk\s?deals\w*\b",r"\b\w*block\s?deals\w*\b",r"\b\w*fake\s?ipo\s?alerts\w*\b",r"\b\w*unauthorized\s?broker\s?accounts\w*\b",r"\b\w*hacked\s?trading\s?accounts\w*\b",r"\b\w*fake\s?advisory\s?services\w*\b",r"\b\w*phishing\s?for\s?financial\s?data\w*\b",r"\b\w*sebi\s?rules\s?bypass\w*\b",r"\b\w*tax\s?evasion\s?techniques\w*\b",r"\b\w*gst\s?fraud\s?in\s?trading\w*\b",r"\b\w*aml\s?violations\w*\b",r"\b\w*high\s?frequency\s?trading\s?misuse\w*\b",r"\b\w*latency\s?arbitrage\w*\b",r"\b\w*algo\s?trading\s?manipulations\w*\b",r"\b\w*sql\s?injection\s?in\s?trading\s?platform\w*\b",r"\b\w*data\s?scraping\s?on\s?stock\s?markets\w*\b",r"\b\w*api\s?exploitation\s?in\s?trading\w*\b",r"\b\w*how\s?to\s?hack\s?trading\s?systems\w*\b",r"\b\w*bypass\s?stock\s?trading\s?limits\w*\b",r"\b\w*access\s?restricted\s?market\s?data\w*\b",r"\b\w*steal\s?trading\s?algorithms\w*\b",r"\b\w*spoof\s?sebi\s?regulations\w*\b"]

# Function to check user input for restricted keywords using regex
def check_restriction(user_input):
    for pattern in restricted_keywords:
        if re.search(pattern, user_input, re.IGNORECASE):
            return "I'm sorry, but your query contains terms that are restricted. Could you please rephrase or remove sensitive information? I'm here to assist with any compliant queries!"
    return "Access Granted: No restricted keywords detected."

# Function to check user input for restricted keywords using exact match
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    result = check_restriction(user_input)
    print(result)


# Works well!!
# regex: \b\w*scrape\w*\b

