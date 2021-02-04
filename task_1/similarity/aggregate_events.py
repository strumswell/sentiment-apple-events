#%% Import modul
import pandas as pd

# %% Read documents
events = ['2020_Nov_Post', '2020_Nov', '2020_Nov_Pre', '2020_Oct_Post', '2020_Oct', '2020_Oct_Pre', '2020_Sept_Post', '2020_Sept', '2020_Sept_Pre', '2020_Jun_Post', '2020_Jun', '2020_Jun_Pre', '2019_Sept_Post', '2019_Sept', '2019_Sept_Pre', '2019_Jun', '2019_Jun_Pre', '2019_Mar_Post', '2019_Mar', '2019_Mar_Pre', '2018_Oct_Post', '2018_Oct', '2018_Oct_Pre', '2018_Sept_Post', '2018_Sept', '2018_Sept_Pre', '2018_Jun_Post', '2018_Jun', '2018_Jun_Pre', '2018_Mar']
dfs = []

for event in events:
    dfs.append(pd.read_csv("../../data/cleaned/"+event+"_CleanedData.csv"))
    current_df = dfs[-1]
    rows_current_df = len(current_df.index)
    event_column = pd.DataFrame({'Event': [event for number in range(0, rows_current_df)]})
    current_df = current_df.join(event_column)
    current_df = current_df.drop(current_df.columns[[0, 1]], axis=1) # löscht unnamed spalte
    dfs[-1] = current_df

# %%
# test = dfs[5]

# pd.to_csv("../data/working/"+event+"_working.csv")

#%% Create df containing pre event post
#
event_comments = {}

for event in dfs:
    name_parts = event['Event'][0].split('_')
    event_name = name_parts[0] + name_parts[1]
    if event_name in event_comments:
        event_comments[event_name] = event_comments[event_name].append(event, ignore_index = True)
    else:
        event_comments[event_name] = event

# %%
# test_dict = event_comments['2018Sept']

# %% Cleaning nan
for df in dfs:
    print(df.isnull().sum())
    df = df.dropna(how='any', axis='rows')
    print(df.isnull().sum())

# %% speichert dict in working
for event in event_comments:
    event_comments[event].to_csv("../data/aggregated_events/" + event + "_aggregated_Data.csv")
