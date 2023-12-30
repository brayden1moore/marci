import xgboost as xgb
import numpy as np
import pandas as pd
import pickle as pkl
import os
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

# set dirs for other files
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')
model_directory = os.path.join(parent_directory, 'Models')
pickle_directory = os.path.join(parent_directory, 'Pickles')

file_path = os.path.join(data_directory, 'gbg_this_year.csv')
gbg = pd.read_csv(file_path, low_memory=False)

file_path = os.path.join(data_directory, 'results.csv')
results = pd.read_csv(file_path, low_memory=False)

# get team abbreviations
file_path = os.path.join(pickle_directory, 'team_name_to_abbreviation.pkl')
with open(file_path, 'rb') as f:
    team_name_to_abbreviation = pkl.load(f)

file_path = os.path.join(pickle_directory, 'team_abbreviation_to_name.pkl')
with open(file_path, 'rb') as f:
    team_abbreviation_to_name = pkl.load(f)

# get schedule
file_path = os.path.join(pickle_directory, 'schedule.pkl')
with open(file_path, 'rb') as f:
    schedule = pkl.load(f)

# get current week
file_path = os.path.join(pickle_directory, 'the_week.pkl')
with open(file_path, 'rb') as f:
    the_week = pkl.load(f)

# load models
# moneyline
model = 'xgboost_ML_no_odds_71.4%'
file_path = os.path.join(model_directory, f'{model}.json')
xgb_ml = xgb.Booster()
xgb_ml.load_model(file_path)

# over/under
model = 'xgboost_OU_no_odds_59.8%'
file_path = os.path.join(model_directory, f'{model}.json')
xgb_ou = xgb.Booster()
xgb_ou.load_model(file_path)


def get_week():
    week = the_week['week']
    year = the_week['year']
    return int(week), int(year)


def get_games(week):
    df = schedule[week-1]
    df['Away Team'] = [' '.join(i.split('\xa0')[1:]) for i in df['Away TeamAway Team']]
    df['Home Team'] = [' '.join(i.split('\xa0')[1:]) for i in df['Home TeamHome Team']]
    df['Date'] = pd.to_datetime(df['Game TimeGame Time'])
    df['Date'] = df['Date'].dt.strftime('%A %d/%m %I:%M %p')
    df['Date'] = df['Date'].apply(lambda x: f"{x.split()[0]} {int(x.split()[1].split('/')[1])}/{int(x.split()[1].split('/')[0])} {x.split()[2]}".capitalize())
    return df[['Away Team','Home Team','Date']]


def get_one_week(home,away,season,week):
    try:
        max_GP_home = gbg.loc[((gbg['home_team'] == home) | (gbg['away_team'] == home)) & (gbg['GP'] < week)]['GP'].max()
        max_GP_away = gbg.loc[((gbg['home_team'] == away) | (gbg['away_team'] == away)) & (gbg['GP'] < week)]['GP'].max()

        home_df = gbg.loc[((gbg['away_team']==home) | (gbg['home_team']==home)) & (gbg['Season']==season) & (gbg['GP']==max_GP_home)]
        gbg_home_team = home_df['home_team'].item()
        home_df.drop(columns=['game_id','home_team','away_team','Season','game_date'], inplace=True)
        home_df = home_df[[i for i in home_df.columns if '.Away' not in i] if gbg_home_team==home else [i for i in home_df.columns if '.Away' in i]]
        home_df.columns = [i.replace('.Away','') for i in home_df.columns]

        away_df = gbg.loc[((gbg['away_team']==away) | (gbg['home_team']==away)) & (gbg['Season']==season) & (gbg['GP']==max_GP_away)]
        gbg_home_team = away_df['home_team'].item()
        away_df.drop(columns=['game_id','home_team','away_team','Season','game_date'], inplace=True)
        away_df = away_df[[i for i in away_df.columns if '.Away' not in i] if gbg_home_team==away else [i for i in away_df.columns if '.Away' in i]]
        away_df.columns = [i.replace('.Away','') + '.Away' for i in away_df.columns]

        df = home_df.reset_index(drop=True).merge(away_df.reset_index(drop=True), left_index=True, right_index=True)
        return df
    except ValueError:
        return pd.DataFrame()


def predict(home,away,season,week,total):
    global results

    # finish preparing data
    if len(home)>4:
        home_abbrev = team_name_to_abbreviation[home]
    else:
        home_abbrev = home

    if len(away)>4:
        away_abbrev = team_name_to_abbreviation[away]
    else:
        away_abbrev = away

    data = get_one_week(home_abbrev,away_abbrev,season,week)
    data['Total Score Close'] = total
    matrix = xgb.DMatrix(data.astype(float).values)

    # create game id 
    if week < 10:
        game_id = str(season) + '_0' + str(int(week)) + '_' + away_abbrev + '_' + home_abbrev
    else:
        game_id = str(season) + '_' + str(int(week)) + '_' + away_abbrev + '_' + home_abbrev

    try:
        moneyline_result = results.loc[results['game_id']==game_id, 'winner'].item()
    except:
        moneyline_result = 'N/A'

    try:
        ml_predicted_proba = xgb_ml.predict(matrix)[0][1]
        winner_proba = max([ml_predicted_proba, 1-ml_predicted_proba]).item()
        moneyline = {'Winner': [home if ml_predicted_proba>0.5 else away if ml_predicted_proba<0.5 else 'Toss-Up'],
                     'Probabilities':[winner_proba],
                     'Result': moneyline_result}
    except:
        moneyline = {'Winner': 'NA',
                     'Probabilities':['N/A'],
                     'Result': moneyline_result}
    
    try:
        result = results.loc[results['game_id']==game_id, 'total'].item()
        over_under_result = 'Over' if float(result)>float(total) else 'Push' if float(result)==float(total) else 'Under'

    except:
        over_under_result = 'N/A'
    
    try:
        ou_predicted_proba = xgb_ou.predict(matrix)[0][1]
        ou_proba = max([ou_predicted_proba, 1-ou_predicted_proba]).item()

        over_under = {'Over/Under': ['Over' if ou_predicted_proba>0.5 else 'Under'],
                      'Probability': [ou_proba],
                      'Result': over_under_result}
    except:
        over_under = {'Over/Under': 'N/A',
                      'Probability': ['N/A'],
                      'Result': over_under_result}
    
    return game_id, moneyline, over_under
