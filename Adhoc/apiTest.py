import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

api = "https://api.polygon.io/v2/aggs/ticker/WEED/range/1/minute/2023-07-01/2023-08-31?adjusted=true&sort=asc&limit=500000&apiKey=wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM"
api = "https://api.polygon.io/v3/reference/tickers?search=apple&limit=1000&apiKey=wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM"
api = "https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&apiKey=wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM"

api = "https://api.polygon.io/vX/reference/tickers/AAPL/events?apiKey=wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM"

api = "https://api.polygon.io/v3/reference/tickers/AAPL?date=2023-01-01&apiKey=wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM"


apiGET = requests.get(api)
JSON   = apiGET.json()

data     = pd.DataFrame(JSON["results"])

max(data["t"])

from datetime import datetime, timedelta

# Unix timestamp in milliseconds
timestamp_msec = 1695945300000
  # Replace this with your Unix timestamp in milliseconds

# Convert Unix timestamp to datetime object
timestamp_sec = timestamp_msec / 1000  # Convert milliseconds to seconds
est_time = datetime.utcfromtimestamp(timestamp_sec) - timedelta(hours=4)  # UTC to EST (UTC-5)

# Format the EST time including date
est_time_with_date = est_time.strftime('%Y-%m-%d %H:%M:%S')  # Format as 'YYYY-MM-DD HH:MM:SS'

print("Date and Time (EST):", est_time_with_date)

NASDAQ
TSX
NYSE

# CA
XTSE
XTSX

# US
XNYS
XNAS

tickers = yf.Tickers('msft appl tsla')
tickers.tickers["MSFT"].info


import yfinance as yf

ticker= yf.Ticker("aapl")

# show actions (dividends, splits)
# 
a = ticker.actions

b = ticker.balancesheet #use
c = ticker.quarterly_incomestmt

ticker.major_holders
ticker.institutional_holders
ticker.mutualfund_holders

s = ticker.get_shares_full(start="2022-01-01", end=None)
s["2023-11-09"]




x = ticker.basic_info.market_cap
ticker.basic_info.last_price
ticker.basic_info.market_cap
ticker.basic_info.shares
ticker.cash_flow
ticker.earnings_dates

ticker.financials
ticker.quarterly_balance_sheet

