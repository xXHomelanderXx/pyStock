  ###########################################################################
  #  This code prepares the data before being modeled
  #  
  #
  ############################################################################

def dataPrep(
    tickerPrices="I:\\Data\\pyStock\\TickerPrices_5m\\",
    tickerDetails="C:\\Projects\\pyStock\\Tickers\\tickerDetails_2024-09-06.csv",
    filter_missing_details=True,
    na_threshold=0.80,  # Remove variables with too many NAs, if % of NA is over this threshold
    nrows=100000,  # Put None for all rows
    outputPath="I:\\Data\\pyStock\\dataPrepped\\",
    vars_to_create=[  # Vars to create -> (start, end, time_unit)
        (-30,0,"m"),  # Target 1
        (-1, 0,"d"),  # target 2
        (-7, 0, "d"), # target 3
        (0, 5, "m"),
        (5, 10, "m"),
        (10, 15, "m"),
        (15, 20, "m"),
        (20, 25, "m"),
        (25, 30, "m"),
        (30, 60, "m"),
        (1, 2, "h"),
        (2, 3, "h"),
        (0, 10, "m"),
        (0, 30, "m"),
        (0, 1, "h"),
        (0, 2, "h"),
        (0, 1, "d"),
        (1, 2, "d"),
        (2, 3, "d"),
        (3, 4, "d"),
        (4, 5, "d"),
        (5, 6, "d"),
        (6, 7, "d")
    ],
    vars_to_remove=[
        'v', 'vw', 'o', 'c', 'h', 'l', 't', 'n',
        'time_stamp', 'current_time', 'time_est',
        'cumulative_volume', 'chg_open_close_pc',
        'ticker', 'year',
        #'day_of_month',
        #'month', 
        'chg_close_0_pc'
    ]
):

    """
    Prepares data for analysis by cleaning and transforming data.

    Parameters:
    - tickerPrices (str): Path to ticker prices data.
    - tickerDetails (str): Path to ticker details CSV file.
    - tickersList (str): Path to tickers list CSV file.
    - outputPath (str): Path to save the output.
    - filter_missing_details (bool): Whether to filter details only.
    - train_test_cut_dt (str): Date to split training and test data.
    - na_threshold (float): Threshold for NA values.
    - nrows (int): Number of rows to read.
    - vars_to_create (list): List of tuples defining variables to create.
    - target (str): Target variable name.
    - vars_to_remove (list): List of variables to remove.
    
    Returns:
    - Processed data.
    """

  ########################################
  # 00 - Import needed modules ------------------------------------------------------------------------------------------------------
  ########################################
    import pandas as pd
    import os
    import numpy as np
    import gc
    from datetime import datetime
    
  ########################################
  # 01 - load data ------------------------------------------------------------------------------------------------------------------
  ########################################

    dataFiles = os.listdir(tickerPrices)
    dataFiles = [file for file in dataFiles if file.endswith(".csv")]
    
    # load data
    df_list = []
    total_rows = 0

    for file in dataFiles:
      df_raw = pd.read_csv(tickerPrices + file, dtype = {'otc': str})

      ########################################
      # 03 - merge ticker details and filter------------------------------------------------------------------------------------------------
      ########################################

      df_tickerDetails = pd.read_csv(tickerDetails)

      if filter_missing_details:
        df = df_raw[df_raw["ticker"].isin(df_tickerDetails["ticker"].unique())]
      else:
        df = df_raw

      df =  df.merge(df_tickerDetails, on = "ticker", how = "left")

          # Filter out columns with more than threshold missing value
      na_percentage = df.isna().mean()
      df = df.loc[:, na_percentage <= na_threshold]

      ########################################
      # 04 - dataprep ----------------------------------------------------------------------------------------------------------------------
      ########################################

      # Create simple predictors
      df["time_stamp"]   = pd.to_datetime(df["time_est"])
      df["date"]         = df["time_stamp"].dt.date
      df["day_of_week"]  = df["time_stamp"].dt.strftime('%A')
      df["day_of_month"] = df["time_stamp"].dt.day
      df["month"]        = df["time_stamp"].dt.month
      df["year"]         = df["time_stamp"].dt.year
      df["current_time"] = df["time_stamp"].dt.time
      df["current_hour"] = df["time_stamp"].dt.hour

      # Define stock details for appropriate year
      metrics = df_tickerDetails.drop(columns=["ticker"]).columns
      metricsList = set([s[:-5] for s in metrics])
      
      df_fixed_list = []
      for y in set(df["year"]):
        df_tmp = df[df["year"] == y]
        
        for m in metricsList:
          df_tmp[f'{m}_last'] = df_tmp[f'{m}_{y-1}']
          df_tmp[f'{m}_pre_last'] = df_tmp[f'{m}_{y-2}']
        
        df_fixed_list.append(df_tmp)
    
      df_fixed = pd.concat(df_fixed_list, ignore_index=True)
      
      df = df_fixed.drop(columns=metrics, errors='ignore')
      
      # clear some memory
      del df_fixed, df_fixed_list, df_tmp, df_raw
      gc.collect()

      # Daily price changes
      group_by_vars = ["ticker", "date"]

      df_price_daily = df.groupby(group_by_vars).agg(
        open_price_day   = ('o',"first"),
        closed_price_day = ("c", "last")
      ).reset_index()


      df = df.merge(df_price_daily, on = group_by_vars, how = "left")

      df["chg_close_0_pc"] = df["closed_price_day"]/df["c"] - 1
      df["chg_0_open_pc"] = df["c"]/df["open_price_day"] - 1
      df["chg_open_close_pc"] = df["closed_price_day"]/df["open_price_day"] - 1

      df = df.drop(columns = ["open_price_day","closed_price_day"])

      del df_price_daily
      gc.collect()

      ########################################
      # 04 - define creating function ------------------------------------------------------------------------------------------------------
      ########################################

      # define function to create price change from y to x 

      def create_var(df,start, end, unit):
        
        target_time_start = df['time_stamp'] - pd.to_timedelta(start, unit=unit)
        target_time_end = df['time_stamp'] - pd.to_timedelta(end, unit=unit)
        
        # Use searchsorted for efficient lookups (finds index of where value fits in time_stamp vect or)
        time_stamps = df['time_stamp'].values
        
        # Get inices of start and finish
        indices_start = time_stamps.searchsorted(target_time_start.values, side='right') - 1
        indices_end   = time_stamps.searchsorted(target_time_end.values, side='right') - 1
        # small patch to make sure indexes don't go out of bounds (negative). Set them to first element 0
        indices_start = np.where(indices_start < 0, 0, indices_start)
        indices_end   = np.where(indices_end   < 0, 0, indices_end)

        # Calculate price change
        df[f"price_chg_{start}_{end}{unit}_pc"] = df['c'].iloc[indices_start].values/df['c'].iloc[indices_end].values - 1
        
        # Calculate trade volume amount
        df["cumulative_volume"] = df['v'].cumsum()
        day_avg = df['v'].mean()

        df[f"volume_chg_{start}_{end}{unit}_pc"] = (df['cumulative_volume'].iloc[indices_start].values-df['cumulative_volume'].iloc[indices_end].values)/day_avg
        
        return df

      # Create variables for each ticker
      unique_tickers = df["ticker"].unique()
      df_ticker_all = []
      
      ticker_tot = len(unique_tickers)

      for ticker in unique_tickers:

        df_tmp = df[df["ticker"] == ticker].sort_values(by='time_stamp')
        
        for start, end, unit in vars_to_create:
          df_tmp = create_var(df_tmp, start, end, unit)
        
        df_ticker_all.append(df_tmp)

        print(f"{ticker} - variables created")

      # merge all tickers into one df
      df_appended = pd.concat(df_ticker_all)

      df_appended = df_appended.drop(columns = vars_to_remove + ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n',
      'time_stamp','current_time', 'time_est',
      'cumulative_volume',
      'chg_open_close_pc',
      'chg_close_0_pc'])

      df_list.append(df_appended)

      # Stop loading files onces greater then sample size
      total_rows = total_rows + df_appended.shape[0]
      
      if (nrows != None):
        if (total_rows > nrows):
          break

    df_final = pd.concat(df_list, ignore_index=True)[:nrows]

    if outputPath != None:
      df_final.to_parquet(outputPath + "dataPrepped_" + str(datetime.now().date()) + ".parquet", index=False)

    return df_final
