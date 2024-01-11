import nfl_data_py.nfl_data_py as nfl
import build
import datetime as dt
import numpy as np
import io
import pandas as pd
pd.set_option('chained_assignment',None)
pd.set_option('display.max_columns',None)
import os
import pickle as pkl
import requests
from bs4 import BeautifulSoup

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

# get schedule
print('Getting schedule.\n')
url = 'https://www.nbcsports.com/nfl/schedule'
df = pd.read_html(url)
file_path = os.path.join(pickle_directory, 'schedule.pkl')
with open(file_path, 'wb') as f:
    pkl.dump(df, f)

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

def get_lines(season):
    url = 'https://www.sportsbettingdime.com/nfl/public-betting-trends/'
    response = requests.get(url)
    html = BeautifulSoup(response.text)
    week = html.find_all('h2')[0].get_text().split(' ')[-1]
    df = pd.read_html(io.StringIO(response.text))

    columns = list(df[0].loc[0,:].values)
    columns = columns[:2] + columns[3:] 

    data_list = []
    for data in df[1:-1]:
        data.columns = columns
        data['Matchup'] = data['Matchup'].str.extract('([A-Z]+)[^A-Za-z]*$')
        data_dict = {
                'season' : season,
                'week' : week,
                'home_team' : data['Matchup'][1],
                'away_team' : data['Matchup'][0],
                'away_spread' : float(data.iloc[0,4].replace('+','')),
                'money_on_away_ats' : int(data.iloc[0,5].replace('%',''))/100,
                'bets_on_away_ats' : int(data.iloc[0,6].replace('%',''))/100,
                'away_moneyline' : int(data['moneyline'][0].replace('+','')),
                'money_on_away_ml' : int(data.iloc[0,8].replace('%',''))/100,
                'bets_on_away_ml' : int(data.iloc[0,9].replace('%',''))/100,
                'over_under' : data['total'].str.replace('o','').str.replace('u','').astype(float).mean(),
                'money_on_over' : int(data.iloc[0,11].replace('%',''))/100,
                'bets_on_over' : int(data.iloc[0,12].replace('%',''))/100
            }
        data_list.append(data_dict)

    betting_data = pd.DataFrame(data_list)
    betting_data['key'] = [f'{season}_{week}_{away}_{home}' for season, week, away, home in betting_data[['season','week','away_team','home_team']].values]
    return betting_data

current_week = get_week()[0]
the_week = {'week':current_week,
            'year':current_season}
file_path = os.path.join(pickle_directory, 'the_week.pkl')
with open(file_path, 'wb') as f:
    pkl.dump(the_week, f)

# update current season
build.build_gbg_data(get_seasons=[current_season])
build.add_odds_data()

# get winners
pbp = build.get_pbp_data([current_season])
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



