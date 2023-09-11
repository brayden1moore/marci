from operator import index
import nfl_data_py.nfl_data_py as nfl
import build
import datetime as dt
import numpy as np
import pandas as pd
pd.set_option('chained_assignment',None)
pd.set_option('display.max_columns',None)
import os
import pickle as pkl

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')
pickle_directory = os.path.join(parent_directory, 'Pickles')

# get team abbreviations
file_path = os.path.join(pickle_directory, 'team_name_to_abbreviation.pkl')
with open(file_path, 'rb') as f:
    team_name_to_abbreviation = pkl.load(f)
file_path = os.path.join(pickle_directory, 'team_abbreviation_to_name.pkl')
with open(file_path, 'rb') as f:
    team_abbreviation_to_name = pkl.load(f)

# get current season
year = dt.datetime.now().year
month = dt.datetime.now().month
current_season = year if month in [8,9,10,11,12] else year-1

# update current season
build.build_gbg_data(get_seasons=[current_season])
build.add_odds_data()

# get winners
pbp = build.get_pbp_data([2023])
pbp = pbp.drop_duplicates(subset='game_id')
pbp[['season','week','away','home']] = pbp['game_id'].str.split('_', expand=True)
games = pbp[['game_id','away_score','home_score','season','week','away','home']]
games[['away_score','home_score','season','week']] = games[['away_score','home_score','season','week']].astype(int)

games['away_team'] = games['away'].map(team_abbreviation_to_name)
games['home_team'] = games['home'].map(team_abbreviation_to_name)

games['total'] = games['away_score'] + games['home_score']
games['winner'] = [a if a_s>h_s else h if h_s>a_s else 'Tie' for a,h,a_s,h_s in games[['away_team','home_team','away_score','home_score']].values]

file_path = os.path.join(data_directory, 'results.csv')
games[['game_id','total','winner']].to_csv(file_path, index=False)


