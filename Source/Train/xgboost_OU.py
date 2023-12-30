import xgboost as xgb
import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
data_directory = os.path.join(parent_directory, 'Data')
model_directory = os.path.join(parent_directory, 'Models')
pickle_directory = os.path.join(parent_directory, 'Pickles')

file_path = os.path.join(data_directory, 'gbg_and_odds.csv')
data = pd.read_csv(file_path).dropna()

OU = data['Over']
data.drop(columns=['Home-Team-Win','Over','Season','home_team','away_team','game_date','Key','Home Score','Away Score','Home Odds Close','Away Odds Close','Home Winnings','Away Winnings','Away Odds','Home Odds'], inplace=True)

acc_results = []

for x in tqdm(range(100)):
    X_train, X_test, y_train, y_test = train_test_split(data, OU, test_size=.1)

    train_games = X_train['game_id']
    test_games = X_test['game_id']

    X_train.drop(columns=['game_id'], inplace=True)
    X_test.drop(columns=['game_id'], inplace=True)

    train = xgb.DMatrix(X_train.astype(float).values, label=y_train)
    test = xgb.DMatrix(X_test.astype(float).values, label=y_test)

    param = {
        'max_depth': 6,
        'eta': 0.05,
        'objective': 'multi:softprob',
        'num_class': 3
    }
    epochs = 300

    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(np.argmax(z))

    acc = round(accuracy_score(y_test, y)*100, 1)
    acc_results.append(acc)
    clear_output(wait=True)
    print(f"Best accuracy: {max(acc_results)}%")
    
    # only save results if they are the best so far
    if acc == max(acc_results):
        file_path = os.path.join(pickle_directory, 'train_games_OU_no_odds.pkl')
        with open(file_path,'wb') as f:
            pkl.dump(train_games,f)

        file_path = os.path.join(pickle_directory, 'test_games_OU_no_odds.pkl')
        with open(file_path,'wb') as f:
            pkl.dump(test_games,f)

        file_path = os.path.join(model_directory, f'xgboost_OU_no_odds_{acc}%.json')
        model.save_model(file_path)

print('Done')