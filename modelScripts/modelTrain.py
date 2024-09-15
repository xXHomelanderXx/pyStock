########################################
# 01 - Set Parameters--------------------------------------------------------------------------------------------------------------
########################################
import h2o
import numpy as np
import re
import pandas as pd
import gc
from h2o.automl import H2OAutoML
from modelScripts.dataPrep import dataPrep

#model params
input_df_path     = "I:\\Data\\pyStock\\dataPrepped\\dataPrepped_2024-09-15.parquet"
train_test_cut_dt = "2022-09-30" # Split point between traning and test data
target            = "price_chg_-30_0m_pc"
outputPath        = "C:\\Projects\\pyStock\\models\\"
exclude_algos     = [] # DeepLearning
vars_to_exclude   = None #['price_chg_-1_0d_pc', 'volume_chg_-1_0d_pc', 'price_chg_-7_0d_pc', 'volume_chg_-7_0d_pc']

####################################
# 02 - prepare data for modeling 
####################################

# load preprocessed datta

dfPrepped = pd.read_parquet(input_df_path)

#dfPrepped.to_parquet("I:\\Data\\pyStock\\dataPrepped\\dataPrepped_2024-09-15.parquet")
#pd.read_parquet("I:\\Data\\pyStock\\dataPrepped\\dataPrepped_2024-09-15.parquet")

# data prep to create processed data from raw (run if needed)
dfPrepped = dataPrep(
  tickerPrices           ="I:\\Data\\pyStock\\TickerPrices_5m\\",
  tickerDetails          = "C:\\Projects\\pyStock\\Tickers\\tickerDetails_2024-09-06.csv",
  filter_missing_details = True,
  na_threshold           = 0.80,
  nrows                  = 20000000,
  outputPath= "I:\\Data\\pyStock\\dataPrepped\\", #"I:\\Data\\pyStock\\dataPrepped\\",
  vars_to_create =[  # Vars to create -> (start, end, time_unit)
        (-30,0,"m"),  # Target 1
        #(-1, 0,"d"),  # target 2
        #(-7, 0, "d"), # target 3
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
        (2, 3, "d")
        #(3, 4, "d"),
        #(4, 5, "d"),
        #(5, 6, "d"),
        #(6, 7, "d")
    ],
  vars_to_remove = [
    'ticker',
    'year',
    #'day_of_month',
    #'month' 
    ]
)

####################################
# Automl model
###################################

# initialize h2o
h2o.init()

df_train = dfPrepped[pd.to_datetime(dfPrepped["date"]).dt.date <= pd.to_datetime(train_test_cut_dt).date()]
df_test =  dfPrepped[pd.to_datetime(dfPrepped["date"]).dt.date > pd.to_datetime(train_test_cut_dt).date()]

del dfPrepped
gc.collect()

train = h2o.H2OFrame(df_train)
test  = h2o.H2OFrame(df_test)

# define response and predictors
response = target

remove_response = ['date',
                    response,
                    re.sub(pattern="price", repl= "volume",string=response),
                    vars_to_exclude
                    ]

predictors = dfPrepped.columns[~dfPrepped.columns.isin(remove_response)].tolist()

# AutoML Model
modelAML = H2OAutoML(max_runtime_secs = 0,
                     max_models= 5,
                     seed = 42)

modelAML.train(x=predictors,
               y=response, 
               training_frame=train,
               leaderboard_frame=test)

lb = h2o.automl.get_leaderboard(modelAML, extra_columns= "ALL")
lb

modelAML.leader.varimp(use_pandas=True)

# Save model
model_path = h2o.save_model(model =modelAML.leader, path =outputPath, force=True)


####################################
# Test model results + plots
###################################

import matplotlib.pyplot as plt
import seaborn as sns

model = modelAML.leader
model = modelAML.get_best_model(algorithm="GBM")

predict = model.predict(test).as_data_frame()
results = pd.concat([test.as_data_frame(),predict], axis = 1)

# chg_-1_0d_pc

x_axis = "predict"
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

################### adhoc
results_ordered = results.sort_values(by = 'predict')
