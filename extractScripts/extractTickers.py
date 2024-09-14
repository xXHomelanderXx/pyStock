
###########################################################################
#  This code extracts the prices of all stocks in the ticker list
#  Handles API limits of 5 calls per minute
#
############################################################################

########################################
# 00 - Import needed modules ------------------------------------------------------------------------------------------------------
########################################

import requests
import pandas as pd
import time
from datetime import datetime
import os

########################################
# 01 - Set Parameters--------------------------------------------------------------------------------------------------------------
########################################

exportPath = os.getcwd() + "\\Data\\allTickers" + "_" + str(datetime.now().date()) + ".csv"

exchanges = ["all"]

########################################
# 02 - function for extract tickers based on starting and ending alphabet----------------------------------------------------------
########################################

def extractTickers(start,end,exchange):
    """
    extracts tickers for a specific Exchanger
    
    due to limit of 1000 at a time, use start and finish to set from what to what lettes to import
    
    """

    if exchange == "all":
         api = "https://api.polygon.io/v3/reference/tickers?market=stocks&ticker.gte="+ start + "&ticker.lte=" + end + "&limit=1000&apiKey=wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM"
    else:
         api = "https://api.polygon.io/v3/reference/tickers?market=stocks&exchange=" + exchange + "&ticker.gte="+ start + "&ticker.lte=" + end + "&limit=1000&apiKey=wg6AXbmzPqS0jPZHT_pyh2fG8z1ogpsM"

    getAPI = requests.get(api).json()

    results = pd.DataFrame(getAPI["results"])

    if len(results) == 1000:
      print("Warning - 1000 limit reached")

    return results

########################################
# 03 - Loop to extract all tickers one letter at a time ----------------------------------------------------------------------------
########################################

tickersList = []

for E in exchanges:
  for i in range(26):
    tickers = extractTickers(chr(65 + i), chr(65 + i + 1) , E)
    print("Extracting tickers starting with " + chr(65 + i) + " from " + E + " - " + str(len(tickers)))
    
    tickersList.append(tickers)
    time.sleep(12)

########################################
# 04 - Create one dataset and extract  ---------------------------------------------------------------------------------------------
########################################

alltickers = pd.concat(tickersList)
alltickers.to_csv(exportPath, index=False)
