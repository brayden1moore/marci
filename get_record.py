import datetime as dt
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

# get record and save it
predictions_df = pd.DataFrame(predictions).T
predictions_df['predicted_winner'] = [i['Winner'][0] if type(i['Winner'])==list else None for i in predictions_df[1]]
predictions_df['predicted_winner'] = predictions_df['predicted_winner'].map(team_abbreviation_to_name)
predictions_df['predicted_over_under'] = [i['Over/Under'][0] if type(i['Over/Under'])==list else None for i in predictions_df[2]]
predictions_df = predictions_df.merge(results, left_index=True, right_on='game_id').merge(gbg_and_odds_this_year[['game_id','Total Score Close']]).dropna(subset=['predicted_winner'])
predictions_df['over_under'] = ['Over' if t>tsc else 'Under' if t<tsc else 'Push' for t,tsc in predictions_df[['total','Total Score Close']].values]

predictions_df['winner_correct'] = (predictions_df['predicted_winner']==predictions_df['winner']).astype(int)
predictions_df['winner_incorrect'] = (predictions_df['predicted_winner']!=predictions_df['winner']).astype(int)
predictions_df['over_under_correct'] = (predictions_df['predicted_over_under']==predictions_df['over_under']).astype(int)
predictions_df['over_under_incorrect'] = (predictions_df['predicted_over_under']!=predictions_df['over_under']).astype(int)

winners_correct = predictions_df['winner_correct'].sum()
winners_incorrect = predictions_df['winner_incorrect'].sum()
over_unders_correct = predictions_df['over_under_correct'].sum()
over_unders_incorrect = predictions_df['over_under_incorrect'].sum()

record = {"winners_correct":str(winners_correct),
        "winners_incorrect":str(winners_incorrect),
        "over_unders_correct":str(over_unders_correct),
        "over_unders_incorrect":str(over_unders_incorrect)}

import json
with open('Source/Data/record.json', 'w') as f:
    json.dump(record,f)

