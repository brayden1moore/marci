from datetime import date, datetime
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

predictions_df['winner_return'] = [0 if tie else ao-1 if (pa and wc) else ho-1 if (ph and wc) else -1 for ao,ho,pa,ph,wc,tie in predictions_df[['Away Odds Close','Home Odds Close','picked_away','picked_home','winner_correct','winner_tie']].values]
predictions_df['over_under_return'] = [0 if push else 0.91 if ouc else -1 for ouc,push in predictions_df[['over_under_correct','over_under_push']].values]
predictions_df = predictions_df.loc[predictions_df['game_date']>datetime(year=2023,month=9,day=19)]

# Save
predictions_df.to_csv('Source/Data/predictions.csv')
bins = np.arange(0.5, 1.05, 0.05)
bin_midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

predictions_df['winner_probability_bin'] = pd.cut(predictions_df['predicted_winner_probability'], bins=bins, labels=bin_midpoints)
predictions_df['over_under_probability_bin'] = pd.cut(predictions_df['predicted_over_under_probability'], bins=bins, labels=bin_midpoints)
winner_binned = predictions_df.groupby('winner_probability_bin')['winner_correct'].mean().reset_index()
over_under_binned = predictions_df.groupby('over_under_probability_bin')['over_under_correct'].mean().reset_index()

## plot

import matplotlib.pyplot as plt
import numpy as np

def style_plot(ax, title):
    ax.set_facecolor('black')
    ax.set_title(title, color='white')
    ax.set_xlabel('MARCI Predicted Probability', color='white')
    ax.set_ylabel('Actual Probability', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    #ax.grid(True, linestyle='--', linewidth=0.5, color='grey')
    ax.set_ylim((0,1.1))

def add_identity_line(ax, max_x):
    x = np.linspace(0.5, max_x, 100)
    ax.plot(x, x, linestyle='--', color='grey')

def add_best_fit_line(ax, x_values, y_values):
    x_values = x_values.astype('float64')
    y_values = y_values.astype('float64')
    mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    coef = np.polyfit(x_values, y_values, 1)
    poly1d_fn = np.poly1d(coef)
    ax.plot(x_values, poly1d_fn(x_values), color='green')
    corr = np.corrcoef(x_values, y_values)[0,1]
    max_x = np.max(x_values)
    max_y = poly1d_fn(max_x)
    #ax.text(max_x, max_y, f'Corr: {corr:.2f}', color='green')

# Create the Winner scatter plot
x_values_winner = winner_binned['winner_probability_bin']
y_values_winner = winner_binned['winner_correct']
fig1 = plt.figure(facecolor='black')
ax1 = fig1.add_subplot(1, 1, 1)
ax1.scatter(x_values_winner,
            y_values_winner,
            color=(0/255, 128/255, 0/255), s=100, marker='o')
add_identity_line(ax1, predictions_df['predicted_winner_probability'].max())
add_best_fit_line(ax1, predictions_df['predicted_winner_probability'], predictions_df['winner_correct'])
line, = ax1.plot([], [], linestyle='--', color='grey') 
marci_line, = ax1.plot([], [], color='green')
ax1.legend([line, marci_line], ['Perfect Model', 'MARCI'], loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
style_plot(ax1, 'Winner Predictions')
plt.savefig('Static/Winner_Predictions_dark.png', facecolor='black')
plt.close(fig1)

# Create the Over/Under scatter plot
x_values_over_under = over_under_binned['over_under_probability_bin']
y_values_over_under = over_under_binned['over_under_correct'] 
fig2 = plt.figure(facecolor='black')
ax2 = fig2.add_subplot(1, 1, 1)
ax2.scatter(x_values_over_under,
            y_values_over_under,
            color=(0/255, 128/255, 0/255), s=100, marker='o')
add_identity_line(ax2, predictions_df['predicted_over_under_probability'].max())
add_best_fit_line(ax2, predictions_df['predicted_over_under_probability'], predictions_df['over_under_correct'])
line, = ax2.plot([], [], linestyle='--', color='grey') 
marci_line, = ax2.plot([], [], color='green')
ax2.legend([line, marci_line], ['Perfect Model', 'MARCI'], loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
style_plot(ax2, 'Over/Under Predictions')
plt.savefig('Static/Over_Under_Predictions_dark.png', facecolor='black')
plt.close(fig2)


## get record
threshold = 0.6

winners_correct = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold, 'winner_correct'].sum()
winners_accuracy = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold, 'winner_correct'].mean()
winners_incorrect = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold,'winner_incorrect'].sum()
winners_tie = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold,'winner_tie'].sum()
winners_return = predictions_df.loc[predictions_df['predicted_winner_probability']>threshold, 'winner_return'].sum()

over_unders_correct = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_correct'].sum()
over_unders_accuracy = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_correct'].mean()
over_unders_incorrect = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_incorrect'].sum()
over_unders_push = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_push'].sum()
over_unders_return = predictions_df.loc[predictions_df['predicted_over_under_probability']>threshold,'over_under_return'].sum()

max_date = predictions_df['game_date'].max()
latest_game = pd.Timestamp(max_date).strftime("%A, %m/%d")

## get binom prob
from scipy.stats import binom

def compare_to_coinflip(c,n):
    prob_fewer = binom.cdf(c, n, 0.5)
    prob_more = 1 - prob_fewer
    return f"{round(prob_more*100,1)}% chance of equal or better performance by flipping a coin."

record = {"winners_correct":str(winners_correct),
        "winners_incorrect":str(winners_incorrect),
        "winners_tie":("-"+str(winners_tie) if winners_tie>0 else ''),
        "winners_return": str(round(winners_accuracy*100,1))+"% accuracy, " + str(round(winners_return,1))+"x return",
        "over_unders_correct":str(over_unders_correct),
        "over_unders_incorrect":str(over_unders_incorrect),
        "over_unders_push":("-"+str(over_unders_push) if over_unders_push>0 else ''),
        "over_unders_return": str(round(over_unders_accuracy*100,1))+"% accuracy, " + str(round(over_unders_return,1))+"x return",
        "latest_game":latest_game,
        "over_unders_binom":compare_to_coinflip(over_unders_correct, (over_unders_incorrect+over_unders_correct)),
        "winners_binom":compare_to_coinflip(winners_correct, (winners_incorrect+winners_correct))}

import json
with open('Source/Data/record.json', 'w') as f:
    json.dump(record,f)

