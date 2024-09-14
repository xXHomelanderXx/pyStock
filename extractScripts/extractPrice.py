
###########################################################################
#  This code extracts the prices of all stocks in the ticker list
#  Handles API limits of 5 calls per minute
#
############################################################################

########################################
# 00 - Import needed modules ------------------------------------------------------------------------------------------------------
########################################

from datetime import datetime, timedelta
import os
import pandas as pd
import requests
import time
import numpy as np
import calendar

########################################
# 01 - Set Parameters--------------------------------------------------------------------------------------------------------------
########################################

multiplier     = "5"
timespan       = "minute"
apiKey         = "wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM" 
year           = 2022
ticketList     = pd.read_csv( os.getcwd() + "\\Data\\allTickers" + "_" + "2023-10-02" + ".csv")
ticker_start   = 11001
ticker_end     = 11300
sleepTime      = 12 # For limiting API calls, limit of 5 calls per minute

exportPath     = os.getcwd() + "\\Data\\TickerPrices_5m\\TickerPrices" + "_" + str(year) + "_" + str(ticker_start) + "_to_" + str(ticker_end) + ".csv"


########################################
# 02 - Define function to extract Prices for one ticker ----------------------------------------------------------------------------
########################################

def extractPrice(ticker, multiplier, timespan, start,end, apiKey):
    
    api = "https://api.polygon.io/v2/aggs/ticker/" + ticker + "/range/" + multiplier + "/" + timespan +"/" + start + "/" + end + "?adjusted=true&sort=asc&limit=50000&apiKey="+ apiKey
    time.sleep(sleepTime)
    
    getAPI = requests.get(api).json()
    prices = pd.DataFrame(getAPI["results"])
    prices["time_est"] = pd.to_datetime(prices["t"]/1000, unit = "s")
    prices["ticker"]   = ticker
    
    print("   Extracting "+ ticker + " from period " + start + " to " + end + " - " + str(len(prices)))
    
    return prices

########################################
# 03 - Extract start and end of months ----------------------------------------------------------------------------------------------
########################################

# Function to generate start and end dates for each month in a given year
def get_start_end_dates(year):
    start_end_dates = []
    for month in range(1, 13):  # Loop through 1 to 12 (months)
        _, last_day = calendar.monthrange(year, month)
        start_date = f"{year}-{month:02d}-01"  # Format: YYYY-MM-DD
        end_date = f"{year}-{month:02d}-{last_day}"  # Format: YYYY-MM-DD
        start_end_dates.append((start_date, end_date))
    return start_end_dates

dates_list = get_start_end_dates(year)

########################################
# 04 - Loop to extract prices of selected Tickers-------------------------------------------------------------------------------------
########################################

tickersToExtract = ticketList.head(ticker_end).tail(ticker_end-ticker_start + 1)["ticker"]

pricesList = []
start_time = time.time()

for t in tickersToExtract:
  
  # an API call is limited to 50,000 so need to break in chunks if using 1 minute intervals
  if (multiplier == "1") &  (timespan == "minute"):
    try:
      apiVolume = "https://api.polygon.io/v2/aggs/ticker/" + t +"/range/1/year/" + str(year) +"-01-01/2023-01-01?adjusted=true&sort=asc&limit=50000&apiKey=" + apiKey
      time.sleep(sleepTime)
      getAPI = requests.get(apiVolume).json()
      volume = pd.DataFrame(getAPI["results"])["n"][0]
  
    except Exception as e:
      print("  API Error:", e)
      continue
  
    print(t + " - has " + "{:,}".format(volume) + " volume in 2023")
  
    conditions = [
      (volume <    250000),  # 1 call per year
      (volume <   2500000),  # 2 calls per year
      (volume <  15000000),  # 3 calls per year
      (volume < 100000000) ] # 4 calls per year
  
    values = [12, 6, 4, 3]
  
    period = np.select(conditions, values, default=2) # defaults to 6 calls per year
  else:
    period = 12
    
  for i in range(int(12/period)):
    start = dates_list[period * i][0]
    end   = dates_list[period * (i+1)-1][1]
    
    try:
      prices = extractPrice(t, multiplier, timespan, start, end, "wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM")
      pricesList.append(prices)
    
    except Exception as e:
        print("  API Error:", e)
        continue

print((time.time() - start_time)/60)

########################################
# 05 - Create one dataset and extract  -------------------------------------------------------------------------------------
########################################

allPrices = pd.concat(pricesList)
if "otc" in allPrices.columns:
  del allPrices["otc"]

allPrices.to_csv(exportPath, index=False)

