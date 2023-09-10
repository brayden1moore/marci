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
current_season = year if month in [8,9,10,11,12] else year-1

# update current season
build.build_gbg_data(get_seasons=[current_season])
build.add_odds_data()

