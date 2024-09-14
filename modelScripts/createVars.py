
###########################################################################
#  This code extracts the prices of all stocks in the ticker list
#  Handles API limits of 5 calls per minute
#
############################################################################

########################################
# 00 - Import needed modules ------------------------------------------------------------------------------------------------------
########################################
import pandas as pd
import os
import numpy as np

########################################
# 01 - Set Parameters--------------------------------------------------------------------------------------------------------------
########################################

filter_details_only  = True 

dataPath           = "I:\\Data\\pyStock\\"

tickerPrices       = dataPath + "TickerPrices_5m\\"
tickerDetails      = "C:\\Projects\\pyStock\\Tickers\\tickerDetails_2024-09-06.csv"
tickersList        = "C:\\Projects\\pyStock\\Tickers\\allTickers_2023-10-02.csv"

outputPath         = "C:\\Projects\\pyStock\\models\\"

# Split between traning and test data
train_test_cut_dt = "2022-09-30"
# remove variables with too many NAs, if % of NA is over this threshold
na_threshold      = 0.80
# put None for all rows
nrows = None

########################################
# 02 - load data ------------------------------------------------------------------------------------------------------------------
########################################

dataFiles = os.listdir(tickerPrices)
dataFiles = [file for file in dataFiles if file.endswith(".csv")]

df_raw = pd.read_csv(tickerPrices + dataFiles[3], dtype = {'otc': str}, nrows = nrows)

########################################
# 03 - merge ticker details and filter------------------------------------------------------------------------------------------------
########################################

df_tickerDetails = pd.read_csv(tickerDetails)

if filter_details_only:
  df = df_raw[df_raw["ticker"].isin(df_tickerDetails["ticker"].unique())]
else:
  df = df_raw

df =  df.merge(df_tickerDetails, on = "ticker", how = "left")

########################################
# 04 - dataprep ----------------------------------------------------------------------------------------------------------------------
########################################

# Create simple predictors
df["time_stamp"]   = pd.to_datetime(df["time_est"])
df["date"]         = pd.to_datetime(df["time_est"]).dt.date
df["day_of_week"]  = df['date'].apply(lambda x: x.strftime('%A'))
df["day_of_month"] = pd.to_datetime(df["date"]).dt.day
df["month"]        = pd.to_datetime(df["date"]).dt.month
df["current_time"] = pd.to_datetime(df["time_est"]).dt.time
df["current_hour"] = pd.to_datetime(df["time_est"]).dt.hour

# Daily price change

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

# Filter out columns with more than threshold missing value
na_percentage = df.isna().mean()

df = df.loc[:, na_percentage <= na_threshold]

########################################
# 04 - define creating function ------------------------------------------------------------------------------------------------------
########################################
# set parameters
# start, end, unit

param_sets = [
  (-30,0,"m"), # target
  (0,5,"m"),
  (5,10,"m"),
  (10,15,"m"),
  (15,20,"m"),
  (20,25,"m"),
  (25,30,"m"),
  (0,10,"m"),
  (0,30,"m"),
  (0,60,"m"),
  (0,120,"m"),
  (1,2,"h"),
  (0,1,"d"),
  (1,2,"d"),
  (2,3,"d"),
  (3,4,"d")
]

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

for ticker in unique_tickers:
  
  df_tmp = df[df["ticker"] == ticker].sort_values(by='time_stamp')
  
  for start, end, unit in param_sets:
    df_tmp = create_var(df_tmp, start, end, unit)
  
  df_ticker_all.append(df_tmp)
  
  print(f"{ticker} - variables created")

# merge all tickers into one df
df_final = pd.concat(df_ticker_all)

####################################
# quick modeling test
###################################
import h2o
import numpy as np
from h2o.automl import H2OAutoML

# initialize h2o
h2o.init()

# data prep

df_train = df_final[df_final["date"] <= pd.to_datetime(train_test_cut_dt).date()]
df_test =  df_final[df_final["date"] > pd.to_datetime(train_test_cut_dt).date()]

train = h2o.H2OFrame(df_train)
valid  = h2o.H2OFrame(df_test)

# define response and predictors
response = "price_chg_-30_0m_pc"

vars_to_remove = [response,
  'v', 'vw', 'o', 'c', 'h', 'l', 't', 'n', 'time_stamp','current_time', 'time_est',  'date',
  'cumulative_volume',
  'chg_open_close_pc',
  "volume_chg_-30_0m_pc",
  'ticker',
  #'day_of_month',
  'month', 
  'chg_close_0_pc']

predictors = df_final.columns[~df_final.columns.isin(vars_to_remove)].tolist()

# AutoML Model
modelAML = H2OAutoML(max_runtime_secs = 0,
                     max_models= 10
                     seed = 42)


modelAML.train(x=predictors,
               y=response, 
               training_frame=train,
               leaderboard_frame=valid)

lb = h2o.automl.get_leaderboard(modelAML, extra_columns= "ALL")
lb

####################################
# Test model results + plots
###################################

import matplotlib.pyplot as plt
import seaborn as sns

model = modelAML.leader

predict = model.predict(valid).as_data_frame()
results = pd.concat([valid.as_data_frame(),predict], axis = 1)

# chg_-1_0d_pc

x_axis = "Free_Cash_Flow_2022"
n_bins = 30

# define percentile to view
lower_perc = 2.5
upper_perc = 97.5

if n_bins > 0:
  bin_edges = np.linspace(np.percentile(results[x_axis], lower_perc), np.percentile(results[x_axis], upper_perc), n_bins + 1)
  results['Bin'] = pd.cut(results[x_axis], bins=bin_edges, include_lowest=True)
else:
  results['Bin'] = results[x_axis]

# Calculate the average observed-to-predicted ratio for each bin

plot_stats = results.groupby('Bin').agg(
    Avg_Observed=(response, 'mean'),
    Avg_Predicted=('predict', 'mean'),
    Count=(response, 'count')
).reset_index()

# Plotting the results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the average observed and predicted values
ax1.plot(plot_stats['Bin'].astype(str), plot_stats['Avg_Predicted'], marker='o', linestyle='-', label='Avg Predicted', color='blue')
ax1.plot(plot_stats['Bin'].astype(str), plot_stats['Avg_Observed'], marker='x', linestyle='-', label='Avg Observed', color='red')
ax1.set_xlabel('Predicted Value Bins')
ax1.set_ylabel('Average Value')
ax1.set_title(f'Average Observed and Predicted Values by {x_axis}')
ax1.legend(loc='upper left')
ax1.tick_params(axis='x', rotation=45)

# Create a secondary y-axis for the number of observations
ax2 = ax1.twinx()
ax2.bar(plot_stats['Bin'].astype(str), plot_stats['Count'], alpha=0.3, color='gray', label='# Observations')
ax2.set_ylabel('# Observations')

# Add legends
fig.legend(loc='upper right')

plt.show()

model.varimp(use_pandas=True)

# Save model
model_path = h2o.saveModel(object = )

################### adhoc
results_ordered = results.sort_values(by = 'predict')
