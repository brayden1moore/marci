import xgboost as xgb
import numpy as np
import pandas as pd
import pickle as pkl
import os
import requests
from bs4 import BeautifulSoup

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')
model_directory = os.path.join(parent_directory, 'Models')
pickle_directory = os.path.join(parent_directory, 'Pickles')

file_path = os.path.join(data_directory, 'pbp_this_year.csv')
pbp = pd.read_csv(file_path, index_col=0, low_memory=False)

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


def get_games():
    # pull from NBC
    url = 'https://www.nbcsports.com/nfl/schedule'
    df = pd.read_html(url)[0]
    df['Away Team'] = [' '.join(i.split('\xa0')[1:]) for i in df['Away TeamAway Team']]
    df['Home Team'] = [' '.join(i.split('\xa0')[1:]) for i in df['Home TeamHome Team']]
    df['Date'] = pd.to_datetime(df['Game TimeGame Time'])
    df['Date'] = df['Date'].dt.strftime('%A %d/%m %I:%M %p')
    df['Date'] = df['Date'].apply(lambda x: f"{x.split()[0]} {int(x.split()[1].split('/')[1])}/{int(x.split()[1].split('/')[0])} {x.split()[2]}".capitalize())

    return df[['Away Team','Home Team','Date']]


def get_one_week(team_name,season,week):
    # create columns
    team = pbp.loc[((pbp['home_team']==team_name) | (pbp['away_team']==team_name)) & (pbp['season']==season)] 
    team['GP'] = team['week']
    team['W'] = [1 if r>0 and team_name==h else 1 if r<0 and team_name==a else 0 for r,a,h in team[['result','away_team','home_team']].values]
    team['L'] = [0 if r>0 and team_name==h else 0 if r<0 and team_name==a else 1 for r,a,h in team[['result','away_team','home_team']].values]
    team['W_PCT'] = team['W']/team['GP']
    team['TOP'] = [t if team_name==p else 0 for t,p in team[['TOP_seconds','posteam']].values]
    team['FGA'] = [1 if team_name==p and f==1 else 0 for p,f in team[['posteam','field_goal_attempt']].values]
    team['FGM'] = [1 if team_name==p and f=='made' else 0 for p,f in team[['posteam','field_goal_result']].values]
    team['FG_PCT'] = team['FGM']/team['FGA']
    team['PassTD'] = np.where((team['posteam'] == team_name) & (team['pass_touchdown'] == 1), 1, 0)
    team['RushTD'] = np.where((team['posteam'] == team_name) & (team['rush_touchdown'] == 1), 1, 0)
    team['PassTD_Allowed'] = np.where((team['defteam'] == team_name) & (team['pass_touchdown'] == 1), 1, 0)
    team['RushTD_Allowed'] = np.where((team['defteam'] == team_name) & (team['rush_touchdown'] == 1), 1, 0)
    team['PassYds'] = [y if p==team_name else 0 for p,y in team[['posteam','passing_yards']].values]
    team['RushYds'] = [y if p==team_name else 0 for p,y in team[['posteam','rushing_yards']].values]
    team['PassYds_Allowed'] = [y if d==team_name else 0 for d,y in team[['defteam','passing_yards']].values]
    team['RushYds_Allowed'] = [y if d==team_name else 0 for d,y in team[['defteam','rushing_yards']].values]
    team['Fum'] = np.where((team['defteam'] == team_name) & (team['fumble_lost'] == 1), 1, 0)
    team['Fum_Allowed'] = np.where((team['posteam'] == team_name) & (team['fumble_lost'] == 1), 1, 0)
    team['INT'] = np.where((team['defteam'] == team_name) & (team['interception'] == 1), 1, 0)
    team['INT_Allowed'] = np.where((team['posteam'] == team_name) & (team['interception'] == 1), 1, 0)
    team['Sacks'] = np.where((team['defteam'] == team_name) & (team['sack'] == 1), 1, 0)
    team['Sacks_Allowed'] = np.where((team['posteam'] == team_name) & (team['sack'] == 1), 1, 0)
    team['Penalties'] = np.where((team['penalty_team'] == team_name), 1, 0)
    team['FirstDowns'] = [1 if team_name==p and f==1 else 0 for p,f in team[['posteam','first_down']].values]
    team['3rdDownConverted'] = [1 if p==team_name and t==1 else 0 for p,t in team[['posteam','third_down_converted']].values]
    team['3rdDownFailed'] = [1 if p==team_name and t==1 else 0 for p,t in team[['posteam','third_down_failed']].values]
    team['3rdDownAllowed'] = [1 if d==team_name and t==1 else 0 for d,t in team[['defteam','third_down_converted']].values]
    team['3rdDownDefended'] = [1 if d==team_name and t==1 else 0 for d,t in team[['defteam','third_down_failed']].values]
    team['PTS'] = [ap if at==team_name else hp if ht==team_name else None for ht,at,hp,ap in team[['home_team','away_team','home_score','away_score']].values]
    team['PointDiff'] = [r if team_name==h else -r if team_name==a else 0 for r,a,h in team[['result','away_team','home_team']].values]

    # aggregate from play-by-play to game-by-game
    features = {
        'GP':'mean',
        'W':'mean',
        'L':'mean',
        'W_PCT':'mean',
        'TOP':'sum',
        'FGA':'sum',
        'FGM':'sum',
        'FG_PCT':'mean',
        'PassTD':'sum',
        'RushTD':'sum',
        'PassTD_Allowed':'sum',
        'RushTD_Allowed':'sum',
        'PassYds':'sum',
        'RushYds':'sum',
        'PassYds_Allowed':'sum',
        'RushYds_Allowed':'sum',
        'Fum':'sum',
        'Fum_Allowed':'sum',
        'INT':'sum',
        'INT_Allowed':'sum',
        'Sacks':'sum',
        'Sacks_Allowed':'sum',
        'Penalties':'sum',
        'FirstDowns':'sum',
        '3rdDownConverted':'sum',
        '3rdDownFailed':'sum',
        '3rdDownAllowed':'sum',
        '3rdDownDefended':'sum',
        'PTS':'mean',
        'PointDiff':'mean'
    }
    game = team.groupby('game_id').agg(features).reset_index()
    game[['W','L']] = game[['W','L']].expanding().sum()
    game[game.columns[4:]] = game[game.columns[4:]].expanding().mean()
    game['TEAM'] = team_name
    game['Season'] = season
    return game.loc[game['GP']==week]


def get_one_week_home_and_away(home,away,season,week):
    home = get_one_week(home,season,week)
    away = get_one_week(away,season,week)
    away.columns = [f'{i}.Away' for i in away.columns]
    gbg = home.merge(away,left_index=True,right_index=True)
    gbg.drop(columns=['TEAM','TEAM.Away','Season.Away','game_id.Away'], inplace=True)
    return gbg


def predict(home,away,season,week,total):
    # finish preparing data
    home_abbrev = team_name_to_abbreviation[home]
    away_abbrev = team_name_to_abbreviation[away]
    gbg = get_one_week_home_and_away(home_abbrev,away_abbrev,season,week)
    gbg['Total Score Close'] = total

    print(gbg)
    matrix = xgb.DMatrix(gbg.drop(columns=['game_id','Season']).astype(float).values)

    # moneyline
    model = 'xgboost_ML_75.4%'
    file_path = os.path.join(model_directory, f'{model}.json')
    xgb_ml = xgb.Booster()
    xgb_ml.load_model(file_path)
    try:
        ml_predicted_proba = xgb_ml.predict(matrix)[0][1]
        winner_proba = max([ml_predicted_proba, 1-ml_predicted_proba]).item()
        moneyline = {'Winner': [home if ml_predicted_proba>0.6 else away if ml_predicted_proba<0.4 else 'Toss-Up'],
                     'Probabilities':[winner_proba]}
    except:
        moneyline = {'Winner': 'NA',
                     'Probabilities':['N/A']}

    # over/under
    model = 'xgboost_OU_59.3%'
    file_path = os.path.join(model_directory, f'{model}.json')
    xgb_ou = xgb.Booster()
    xgb_ou.load_model(file_path)
    try:
        ou_predicted_proba = xgb_ou.predict(matrix)[0][1].item()
        over_under = {'Over/Under': ['Over' if ou_predicted_proba>0.5 else 'Under'],
                      'Probability': [ou_predicted_proba]}
    except:
        over_under = {'Over/Under': 'N/A',
                      'Probabilities': ['N/A']}
    
    return moneyline, over_under


def update_past_predictions():
    file_path = os.path.join(data_directory, 'gbg_and_odds_this_year.csv')
    gbg_and_odds_this_year = pd.read_csv(file_path, index_col=0, low_memory=False)
    total_dict = dict(gbg_and_odds_this_year[['game_id','Total Score Close']])
    games = pbp.drop_duplicates(subset='game_id')

    predictions = {}
    for _, i in games.iterrows():
        game_id = i['game_id']
        home = i['home_team']
        away = i['away_team']
        week = i['week']
        season = i['season']
        total = total_dict[game_id]
        predictions[game_id] = predict(home,away,season,week,total)

    predictions_df = pd.DataFrame(predictions)
    file_path = os.path.join(data_directory, 'predictions_this_year.csv')
    predictions_df.to_csv(file_path)