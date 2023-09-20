from Source.Predict import predict
from flask import Flask, render_template, jsonify, request
import requests
import pickle as pkl
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

import os
import json

# input the over/under lines here for quicker predictions!
lines = [49,
         40.5,
         47,
         46,
         47,
         39.5,
         51.5,
         41,
         45.5,
         40,
         44.5,
         38,
         39,
         46,
         39.5,
         38]

# get week, season
week, season = predict.get_week()
week = 2

app = Flask(__name__, template_folder="Templates", static_folder="Static", static_url_path="/Static")
app.secret_key = 'green-flounder'

games = predict.get_games(week)[['Date','Away Team','Home Team']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_lines')
def get_lines():
    return jsonify(lines)

@app.route('/get_games')
def get_games():
    return jsonify(games.to_dict(orient='records'))

@app.route('/submit_games', methods=['POST'])
def submit_games():
    data = request.json
    data = pd.DataFrame(data).replace('', np.nan).dropna()
    print(data)
    home_teams = data['HomeTeam'].values
    away_teams = data['AwayTeam'].values
    ou_lines = data['OverUnderLine'].values
    row_indices = data['rowIndex'].values
    
    moneylines = []
    over_unders = []
    for row_index,home,away,total in zip(row_indices,home_teams,away_teams,ou_lines):
        game_id, moneyline, over_under = predict.predict(home,away,season,week,total)
        moneyline['rowIndex'] = int(row_index)
        over_under['rowIndex'] = int(row_index)
        moneylines.append(moneyline)
        over_unders.append(over_under)

    print('MoneyLines')
    print(moneylines)
    print('OverUnders')
    print(over_unders)

    return jsonify({'moneylines': moneylines,
                    'over_unders': over_unders})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='7860')