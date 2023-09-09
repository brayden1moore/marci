import nfl_data_py.nfl_data_py as nfl
import build
import datetime as dt
import numpy as np
import pandas as pd
pd.set_option('chained_assignment',None)
pd.set_option('display.max_columns',None)
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')

# get current season
year = dt.datetime.now().year
month = dt.datetime.now().month
season = year if month in [8,9,10,11,12] else year-1

# update current season
gbg = build.build_gbg_data(get_seasons=[2023], overwrite_seasons=[2023])
gbg_and_odds = build.add_odds_data(gbg)
gbg_and_odds_this_year = gbg_and_odds.loc[gbg_and_odds['Season']==season]

file_path = os.path.join(data_directory, 'gbg_and_odds_this_year.csv')
gbg_and_odds_this_year.to_csv(file_path)
