from datetime import datetime
import numpy as np
import pandas as pd
pd.set_option('chained_assignment',None)
pd.set_option('display.max_columns',None)
import os
import pickle as pkl
from Source.Predict.predict import predict

# get team abbreviations
with open('Source/Pickles/team_abbreviation_to_name.pkl', 'rb') as f:
    team_abbreviation_to_name = pkl.load(f)

# get this year's odds and results
gbg_and_odds_this_year = pd.read_csv('Source/Data/gbg_and_odds_this_year.csv')
results = pd.read_csv('Source/Data/results.csv')

# make predictions
from tqdm import tqdm
print("Predicting games and getting record")
predictions = {}
for game_id,home,away,season,week,total in tqdm(gbg_and_odds_this_year[['game_id','home_team','away_team','Season','GP','Total Score Close']].values):
    if week!=1:
        predictions[game_id] = predict(home,away,season,week,total)

# merge data
predictions_df = pd.DataFrame(predictions).T
predictions_df['predicted_winner'] = [i['Winner'][0] if type(i['Winner'])==list else None for i in predictions_df[1]]
predictions_df['predicted_winner'] = predictions_df['predicted_winner'].map(team_abbreviation_to_name)
predictions_df['predicted_winner_probability'] = [i['Probabilities'][0] if type(i['Probabilities'])==list else None for i in predictions_df[1]]
predictions_df['predicted_over_under'] = [i['Over/Under'][0] if type(i['Over/Under'])==list else None for i in predictions_df[2]]
predictions_df['predicted_over_under_probability'] = [i['Probability'][0] if type(i['Probability'])==list else None for i in predictions_df[2]]
predictions_df = predictions_df.merge(results, left_index=True, right_on='game_id').merge(gbg_and_odds_this_year[['game_id','Total Score Close','home_team','away_team','game_date','Home Odds Close','Away Odds Close']]).dropna(subset=['predicted_winner'])
predictions_df['over_under'] = ['Over' if t>tsc else 'Under' if t<tsc else 'Push' for t,tsc in predictions_df[['total','Total Score Close']].values]
predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date'])

# get returns
predictions_df['home'] = predictions_df['home_team'].map(team_abbreviation_to_name)
predictions_df['away'] = predictions_df['away_team'].map(team_abbreviation_to_name)
predictions_df['picked_home'] = (predictions_df['home']==predictions_df['predicted_winner'])
predictions_df['picked_away'] = (predictions_df['away']==predictions_df['predicted_winner'])

predictions_df['winner_correct'] = (predictions_df['predicted_winner']==predictions_df['winner'])
predictions_df['winner_incorrect'] = ((predictions_df['predicted_winner']!=predictions_df['winner']) & (predictions_df['winner']!='Tie'))
predictions_df['winner_tie'] = (predictions_df['winner']=='Tie')
predictions_df['over_under_correct'] = (predictions_df['predicted_over_under']==predictions_df['over_under'])
predictions_df['over_under_incorrect'] = ((predictions_df['predicted_over_under']!=predictions_df['over_under']) & (predictions_df['over_under']!='Push'))
predictions_df['over_under_push'] = (predictions_df['over_under']=='Push')

predictions_df['winner_return'] = [ao-1 if (pa and wc) else ho-1 if (ph and wc) else -1 for ao,ho,pa,ph,wc in predictions_df[['Away Odds Close','Home Odds Close','picked_away','picked_home','winner_correct']].values]
predictions_df['over_under_return'] = [0.91 if ouc else -1 for ouc in predictions_df['over_under_correct']]

threshold = 0.6

winners_correct = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold, 'winner_correct'].sum()
winners_incorrect = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold,'winner_incorrect'].sum()
winners_tie = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold,'winner_tie'].sum()
winners_return = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold, 'winner_return'].sum()

over_unders_correct = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_correct'].sum()
over_unders_incorrect = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_incorrect'].sum()
over_unders_push = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_push'].sum()
over_unders_return = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_return'].sum()

max_date = predictions_df['game_date'].max()
latest_game = pd.Timestamp(max_date).strftime("%A, %m/%d")

record = {"winners_correct":str(winners_correct),
        "winners_incorrect":str(winners_incorrect),
        "winners_tie":("-"+str(winners_tie) if winners_tie>0 else ''),
        "winners_return":str(round(winners_return,1))+"x return",
        "over_unders_correct":str(over_unders_correct),
        "over_unders_incorrect":str(over_unders_incorrect),
        "over_unders_push":("-"+str(over_unders_push) if over_unders_push>0 else ''),
        "over_unders_return":str(round(over_unders_return,1))+"x return",
        "latest_game":latest_game}

import json
with open('Source/Data/record.json', 'w') as f:
    json.dump(record,f)

