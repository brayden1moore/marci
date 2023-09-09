import nfl_data_py.nfl_data_py as nfl
from tqdm import tqdm
import numpy as np
import pandas as pd
pd.set_option('chained_assignment',None)
pd.set_option('display.max_columns',None)
import os
import datetime as dt

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')

def get_pbp_data(get_seasons=[], overwrite_seasons=[]):
    """
    Pull data from nflFastR's Github repo. 
    If you choose to overwrite, it will replace the existing pbp data with the data you pull.

    """
    pbp = nfl.import_pbp_data(get_seasons)
    pbp['TOP_seconds'] = pbp['drive_time_of_possession'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if pd.notnull(x) else 0)
    
    if overwrite_seasons:
        file_path = os.path.join(data_directory, 'pbp.csv')
        old = pd.read_csv(file_path, index_col=0, low_memory=False)
        old = old.loc[~old['season'].isin(overwrite_seasons)]
        pbp = pd.concat([old,pbp])
        pbp.to_csv(file_path)

        year = dt.datetime.now().year
        month = dt.datetime.now().month
        season = year if month in [8,9,10,11,12] else year-1
        pbp_this_year = pbp.loc[pbp['season']==season]
        file_path = os.path.join(data_directory, 'pbp_this_year.csv')
        pbp_this_year.to_csv(file_path)

    return pbp


def build_gbg_data(get_seasons=[], overwrite_seasons=[]):
    """
    Using pbp.csv, build a game-by-game dataset to use for prediction models.
    Populate update_seasons with the current year to only update this season's data while preserving historical data.

    """
    print('Loading play-by-play data.')

    if overwrite_seasons:
        print('Overwriting data for', overwrite_seasons)
        pbp = get_pbp_data(get_seasons, overwrite_seasons)

    if not overwrite_seasons:
        file_path = os.path.join(data_directory, 'pbp.csv')
        pbp = pd.read_csv(file_path, index_col=0)

    pbp = pbp.loc[pbp['season'].isin(get_seasons)]
    game_date_dict = dict(pbp[['game_id','game_date']].values)
    teams = list(set(list(pbp['home_team'].unique()) + list(pbp['away_team'].unique())))
    print(teams)
    seasons = pbp['season'].unique()
    
    print('Building game-by-game data.')
    data = pd.DataFrame()
    for season in seasons:
        print(season)
        for team_name in tqdm(teams):
            # create features
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
            game[game.columns[1:]] = game[game.columns[1:]].shift()
            game['TEAM'] = team_name
            game['Season'] = season

            data = pd.concat([data,game])

    # separate home and away data and merge
    data = data.merge(pbp[['game_id','home_team','away_team']].drop_duplicates())
    home = data.loc[data['home_team']==data['TEAM']]
    away = data.loc[data['away_team']==data['TEAM']]
    away.columns = [f'{i}.Away' for i in away.columns]
    gbg = home.merge(away,left_on='game_id',right_on='game_id.Away')
    gbg.drop(columns=['TEAM','TEAM.Away','home_team.Away','away_team.Away','Season.Away','game_id.Away'], inplace=True)
    gbg['game_date'] = gbg['game_id'].map(game_date_dict)

    if overwrite_seasons:
        file_path = os.path.join(data_directory, 'gbg.csv')
        old = pd.read_csv(file_path, index_col=0, low_memory=False)
        old = old.loc[~old['Season'].isin(overwrite_seasons)]
        gbg = pd.concat([old,gbg])
        file_path = os.path.join(data_directory, 'gbg.csv')
        gbg.to_csv(file_path)

        year = dt.datetime.now().year
        month = dt.datetime.now().month
        season = year if month in [8,9,10,11,12] else year-1
        gbg_this_year = gbg.loc[gbg['Season']==season]
        file_path = os.path.join(data_directory, 'gbg_this_year.csv')
        gbg_this_year.to_csv(file_path)

    return gbg


def add_odds_data(gbg, overwrite=False):
    """
    Get odds from Australian Sports Betting's free online dataset and merge it with game-by-game data.

    """
    
    # get team abbreviations
    team_descriptions = nfl.import_team_desc()
    team_abbreviation_dict = dict(team_descriptions[['team_name','team_abbr']].values)
    
    # get odds
    odds = pd.read_excel('https://www.aussportsbetting.com/historical_data/nfl.xlsx')
    odds['Home Team'] = odds['Home Team'].str.replace('Washington Redskins','Washington Commanders').str.replace('Washington Football Team','Washington Commanders')
    odds['Away Team'] = odds['Away Team'].str.replace('Washington Redskins','Washington Commanders').str.replace('Washington Football Team','Washington Commanders')
    odds['Season'] = [i.year if i.month in [8,9,10,11,12] else i.year-1 for i in odds['Date']]
    odds['Home Team Abbrev'] = odds['Home Team'].map(team_abbreviation_dict)
    odds['Away Team Abbrev'] = odds['Away Team'].map(team_abbreviation_dict)
    odds = odds[['Date','Home Score','Away Score','Home Team Abbrev','Away Team Abbrev','Home Odds Close','Away Odds Close','Total Score Close']]
    odds['Key'] = odds['Date'].astype(str) + odds['Home Team Abbrev'] + odds['Away Team Abbrev']
    odds = odds.drop(columns=['Date','Home Team Abbrev','Away Team Abbrev']).dropna()
    odds['Home Odds'] = [round((i-1)*100) if i>= 2 else round(-100/(i-1)) for i in odds['Home Odds Close']]
    odds['Away Odds'] = [round((i-1)*100) if i>= 2 else round(-100/(i-1)) for i in odds['Away Odds Close']]
    odds['Home Winnings'] = [ho-1 if h>a else -1 if a>h else 0 for ho,h,a in odds[['Home Odds Close','Home Score','Away Score']].values]
    odds['Away Winnings'] = [ao-1 if a>h else -1 if h>a else 0 for ao,h,a in odds[['Away Odds Close','Home Score','Away Score']].values]

    # merge with gbg
    gbg['Key'] = gbg['game_date'].astype(str) + gbg['home_team'] + gbg['away_team']
    gbg_and_odds = gbg.merge(odds, left_on='Key', right_on='Key')
    gbg_and_odds['Home-Team-Win'] = (gbg_and_odds['Home Score']>gbg_and_odds['Away Score']).astype(int)
    gbg_and_odds['Over'] = ((gbg_and_odds['Home Score'] + gbg_and_odds['Away Score'])>gbg_and_odds['Total Score Close']).astype(int)
    
    if overwrite:
        file_path = os.path.join(data_directory, 'gbg_and_odds.csv')
        gbg_and_odds.to_csv(file_path)
    
    return gbg_and_odds

