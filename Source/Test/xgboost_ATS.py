from cgi import test
import xgboost as xgb
import pandas as pd
import pickle as pkl
import numpy as np
import os

model = 'xgboost_ATS_no_odds_57.3%'

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')
model_directory = os.path.join(parent_directory, 'Models')
pickle_directory = os.path.join(parent_directory, 'Pickles')

file_path = os.path.join(model_directory, f'{model}.json')
xgb_ml = xgb.Booster()
xgb_ml.load_model(file_path)

file_path = os.path.join(pickle_directory, 'test_games_ATS_no_odds.pkl')
with open(file_path,'rb') as f:
    test_games = pkl.load(f).tolist()

file_path = os.path.join(data_directory, 'gbg_and_odds.csv')
gbg_and_odds = pd.read_csv(file_path)
test_data = gbg_and_odds.loc[gbg_and_odds['game_id'].isin(test_games)]
test_data_matrix = xgb.DMatrix(test_data.drop(columns=['game_id','Home-Team-Win','Home-Team-Cover','Over','Season','home_team','away_team','game_date','Key','Home Score','Away Score','Home Odds Close','Away Odds Close','Home Winnings','Away Winnings','Away Odds','Home Odds']).astype(float).values)

predicted_probas = xgb_ml.predict(test_data_matrix)
predictions = np.argmax(predicted_probas, axis=1)
test_data['predicted_proba'] = [i[1] for i in predicted_probas]
test_data['prediction'] = predictions
test_data['correct'] = test_data['Home-Team-Cover']==test_data['prediction']
print(test_data['predicted_proba'])
print(test_data['correct'].mean())

bets = test_data.loc[(test_data['predicted_proba']>0.5) | (test_data['predicted_proba']<0.5)]
bets['winnings'] = [0.91 if c==1 else -1 for c in bets['correct']]

print('Actual')
print(bets.loc[bets['Home-Team-Cover']==1].shape)
print(bets.loc[bets['Home-Team-Cover']==0].shape)
print(bets.loc[bets['Home-Team-Cover']==2].shape)

print('Predicted')
print(bets.loc[bets['prediction']==1].shape)
print(bets.loc[bets['prediction']==0].shape)
print(bets.loc[bets['prediction']==2].shape)


import matplotlib.pyplot as plt
fig = plt.figure(facecolor='black')
ax = fig.add_subplot(1, 1, 1, facecolor='black')

# Plot data with line color as RGB(0, 128, 0)
ax.plot(bets['winnings'].cumsum().values*100, linewidth=3, color=(0/255, 128/255, 0/255))

# Set title and labels
ax.set_title('MARCI 3.0 - Against the Spread', color='white')
ax.set_xlabel('Games Bet On', color='white')
ax.set_ylabel('Return (%)', color='white')

# Change tick colors to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Change axis edge colors
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

plt.savefig(f'{model}_dark.png', facecolor='black')