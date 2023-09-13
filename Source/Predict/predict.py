import xgboost as xgb
import numpy as np
import pandas as pd
import pickle as pkl
import os
import requests
from bs4 import BeautifulSoup

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

def get_week():
    headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Dnt': '1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    url = 'https://www.nfl.com/schedules/'
    resp = requests.get(url,headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    h2_tags = soup.find_all('h2')
    year = h2_tags[0].getText().split(' ')[0]
    week = h2_tags[0].getText().split(' ')[-1]
    return int(week), int(year)


def get_games(week):
    # pull from NBC
    url = 'https://www.nbcsports.com/nfl/schedule'
    df = pd.read_html(url)[week-1]
    df['Away Team'] = [' '.join(i.split('\xa0')[1:]) for i in df['Away TeamAway Team']]
    df['Home Team'] = [' '.join(i.split('\xa0')[1:]) for i in df['Home TeamHome Team']]
    df['Date'] = pd.to_datetime(df['Game TimeGame Time'])
    df['Date'] = df['Date'].dt.strftime('%A %d/%m %I:%M %p')
    df['Date'] = df['Date'].apply(lambda x: f"{x.split()[0]} {int(x.split()[1].split('/')[1])}/{int(x.split()[1].split('/')[0])} {x.split()[2]}".capitalize())

    return df[['Away Team','Home Team','Date']]


def get_one_week(home,away,season,week):
    try:
        home_df = gbg.loc[((gbg['away_team']==home) | (gbg['home_team']==home)) & (gbg['Season']==season) & (gbg['GP']==week-1)]
        gbg_home_team = home_df['home_team'].item()
        home_df.drop(columns=['game_id','home_team','away_team','Season','game_date'], inplace=True)
        home_df = home_df[[i for i in home_df.columns if '.Away' not in i] if gbg_home_team==home else [i for i in home_df.columns if '.Away' in i]]
        home_df.columns = [i.replace('.Away','') for i in home_df.columns]
        print(home_df)

        away_df = gbg.loc[((gbg['away_team']==away) | (gbg['home_team']==away)) & (gbg['Season']==season) & (gbg['GP']==week-1)]
        gbg_home_team = away_df['home_team'].item()
        away_df.drop(columns=['game_id','home_team','away_team','Season','game_date'], inplace=True)
        away_df = away_df[[i for i in away_df.columns if '.Away' not in i] if gbg_home_team==away else [i for i in away_df.columns if '.Away' in i]]
        away_df.columns = [i.replace('.Away','') + '.Away' for i in away_df.columns]
        print(away_df)

        df = home_df.merge(away_df, left_on='GP', right_on='GP.Away')
        print(df.columns)
        return df
    except ValueError:
        return pd.DataFrame()


def predict(home,away,season,week,total):
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
    game_id = str(season) + '_0' + str(week) + '_' + away_abbrev + '_' + home_abbrev

    # moneyline
    model = 'xgboost_ML_no_odds_71.4%'
    file_path = os.path.join(model_directory, f'{model}.json')
    xgb_ml = xgb.Booster()
    xgb_ml.load_model(file_path)

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

    # over/under
    model = 'xgboost_OU_no_odds_59.8%'
    file_path = os.path.join(model_directory, f'{model}.json')
    xgb_ou = xgb.Booster()
    xgb_ou.load_model(file_path)
    
    try:
        result = results.loc[results['game_id']==game_id, 'total'].item()
        over_under_result = 'Over' if float(result)>float(total) else 'Under'
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
    
    print(moneyline)
    return game_id, moneyline, over_under
