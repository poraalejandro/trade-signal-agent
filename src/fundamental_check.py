"""Check SEC EDGAR for recent filings (earnings-related and annual) per ticker."""

from datetime import date

import requests

# SEC EDGAR requires a descriptive User-Agent identifying the requester,
# or it blocks the request. Reused from the rag-finance project.
HEADERS = {
    "User-Agent": "trade-signal-agent poraalejandro@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# Official ticker -> CIK mapping maintained by SEC.
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

# Form types that typically accompany an earnings release. 6-K is the
# foreign private issuer equivalent of 8-K/10-Q (e.g. TSM, NBIS).
EARNINGS_FORM_TYPES = ["8-K", "10-Q", "6-K"]

# Form types for the annual report. 20-F is the foreign private issuer
# equivalent of 10-K.
ANNUAL_FORM_TYPES = ["10-K", "20-F"]


def get_cik(ticker):
    """
    Look up a ticker's CIK (Central Index Key) via SEC's official mapping.

    Args:
        ticker: ticker symbol, e.g. "NVDA".

    Returns:
        The zero-padded CIK string (10 digits), as EDGAR endpoints expect it.
    """
    json_sec = requests.get(TICKER_CIK_URL, headers=HEADERS).json()
    for entry in json_sec.values():
        if entry["ticker"] == ticker.upper():
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker {ticker} not found.")


def get_recent_filings(cik):
    """
    Fetch the list of recent filings for a company from SEC EDGAR.

    Args:
        cik: zero-padded CIK string, as returned by get_cik.

    Returns:
        A list of (form_type, filing_date) tuples, newest filing first (EDGAR
        already returns them in that order).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()

    recent = data["filings"]["recent"]
    forms = recent["form"]
    dates = recent["filingDate"]

    return list(zip(forms, dates))


def check_recent_filings(ticker):
    """
    Check how many days have passed since a ticker's most recent
    earnings-related filing (EARNINGS_FORM_TYPES) and most recent annual
    filing (ANNUAL_FORM_TYPES) — covers both domestic and foreign private
    issuer form types (e.g. 6-K/20-F for tickers like TSM, NBIS).

    Args:
        ticker: ticker symbol, e.g. "NVDA".

    Returns:
        A dict with 'days_since_earnings' and 'days_since_annual' (each an
        int, or 'N/A' if no matching filing was found).
    """
    filings = get_recent_filings(get_cik(ticker))
    earnings_filing_date = None
    annual_filing_date = None

    for form, filing_date in filings:
        if form in EARNINGS_FORM_TYPES and earnings_filing_date is None:
            earnings_filing_date = filing_date
        elif form in ANNUAL_FORM_TYPES and annual_filing_date is None:
            annual_filing_date = filing_date
        if earnings_filing_date is not None and annual_filing_date is not None:
            break

    days_since_earnings = (
        (date.today() - date.fromisoformat(earnings_filing_date)).days
        if earnings_filing_date is not None
        else "N/A"
    )
    days_since_annual = (
        (date.today() - date.fromisoformat(annual_filing_date)).days
        if annual_filing_date is not None
        else "N/A"
    )

    return {
        "days_since_earnings": days_since_earnings,
        "days_since_annual": days_since_annual,
    }
