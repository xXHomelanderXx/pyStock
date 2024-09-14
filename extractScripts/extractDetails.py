
###########################################################################
#  This code extracts ticker details forall stocks in the ticker list
#  using yfinace package
############################################################################

########################################
# 00 - Import needed modules ------------------------------------------------------------------------------------------------------
########################################

import pandas as pd
import yfinance as yf
import os
from datetime import datetime

########################################
# 01 - Set Parameters--------------------------------------------------------------------------------------------------------------
########################################

ticketTable   = pd.read_csv("C:\\Projects\\pyStock\\Tickers\\allTickers" + "_" + "2023-10-02" + ".csv")

tablesList  = ["balance_sheet", "cash_flow", "income_stmt"]

metricsList = [["Share Issued", "Stockholders Equity", "Total Debt"],
                ["Free Cash Flow", "Operating Cash Flow"],
                ["Basic EPS","Net Income","Total Revenue"]]

exportPath  = "C:\\Projects\\pyStock\\Tickers\\tickerDetails"  + "_" + str(datetime.now().date()) + ".csv"


########################################
# 02 - Define function to extract ticker details ----------------------------------------------------------------------------------
########################################

def get_table(ticker, table, metrics):
  
  all_ticker_info = yf.Ticker(ticker)
  
  df = getattr(all_ticker_info, table)
  df = df.loc[metrics]
  
  # Fix row and col names
  df.columns = pd.to_datetime(df.columns).strftime('%Y')
  df.index   = df.index.str.replace(" ","_")
  
  # Flatten the data and add proper column names
  df_flat = pd.DataFrame(df.values.flatten()).transpose()
  df_flat.columns = [f"{row}_{col}" for row in df.index for col in df.columns]
  
  duplicated_columns = df_flat.columns[df_flat.columns.duplicated(keep='first')]
  df_final = df_flat.loc[:, ~df_flat.columns.duplicated(keep='first')]

  return df_final

########################################
# 03 - Define function to combine all extracts ----------------------------------------------------------------------------------
########################################

def get_tables(ticker, tablesList, metricsList):
  all_tables = []
  for i in range(len(tablesList)):
   df = get_table(ticker,tablesList[i], metricsList[i])
   all_tables.append(df)
   
  df_final = pd.concat(all_tables, axis = 1)
  return df_final


########################################
# 03 - Define function to combine all extracts for all tickers ----------------------------------------------------------------------------------
########################################

def extract_info_tickers(tickerList,tablesList,metricsList):
  
  combined_table = pd.DataFrame()
  
  for ticker in tickerList:
    try:
      ticker_row = get_tables(ticker, tablesList, metricsList)
      print(f"Metrics successfully extracted for {ticker}")
    except Exception as e:
        print(f"Could not extract metrics for {ticker}")
        continue
    ticker_row["ticker"] = ticker
    combined_table = pd.concat([combined_table, ticker_row], ignore_index=True)

  return combined_table

########################################
# 04 - run the eXtract ----------------------------------------------------------------------------------
########################################

tickerList = ticketTable["ticker"].tolist()

final_table = extract_info_tickers(tickerList,tablesList,metricsList)


########################################
# 05 - run the etract ----------------------------------------------------------------------------------
########################################

final_table.to_csv(exportPath, index=False)
