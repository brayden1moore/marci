from Source.Predict import predict
from flask import Flask, render_template, jsonify, request, session
import requests
import pickle as pkl
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

import json
with open('Source/Data/record.json','r') as f:
    record = json.load(f)
with open('Source/Data/lines.json','r') as f:
    lines = json.load(f)

app = Flask(__name__, template_folder="Templates", static_folder="Static", static_url_path="/Static")
app.secret_key = 'green-flounder'

# get week, season
current_week, season = predict.get_week()
current_games = predict.get_games(current_week)[['Date','Away Team','Home Team']]
available_weeks = list(range(current_week+1))[2:]
available_weeks.reverse()

# load current data by default
@app.route('/')
def index():
    print(current_week)
    session['selected_week'] = current_week
    session[f'games_week_{current_week}'] = current_games.to_json()
    return render_template('index.html', **record)

# send week list to front end
@app.route('/get_weeks')
def get_weeks():
    return jsonify(available_weeks)

# send lines to front end
@app.route('/get_lines')
def get_lines():
    return jsonify(lines[str(session.get('selected_week'))])

# send games of selected week to front end
@app.route('/get_games')
def get_games():
    requested_week = int(request.args.get('week'))
    session['selected_week'] = requested_week

    # If select a new week
    if requested_week and requested_week != current_week:
        
        # Check if that week's games are cached
        if session.get(f'games_week_{requested_week}'):
            print("Using cached games")
            games = session.get(f'games_week_{requested_week}')
            games = json.loads(games)
            return jsonify(games)
        else:
            games = predict.get_games(requested_week)[['Date','Away Team','Home Team']]
            session[f'games_week_{requested_week}'] = games.to_json(orient='records')
            return jsonify(games.to_dict(orient='records'))
    else:
        games = current_games
        return jsonify(games.to_dict(orient='records'))

# make predictions
@app.route('/submit_games', methods=['POST'])
def submit_games():
    data = request.json
    data = pd.DataFrame(data).replace('', np.nan).dropna()
    home_teams = data['HomeTeam'].values
    away_teams = data['AwayTeam'].values
    ou_lines = data['OverUnderLine'].values
    row_indices = data['rowIndex'].values
    
    moneylines = []
    over_unders = []
    for row_index,home,away,total in zip(row_indices,home_teams,away_teams,ou_lines):
        selected_week = session.get('selected_week')
        game_id, moneyline, over_under = predict.predict(home,away,season,selected_week,total)
        moneyline['rowIndex'] = int(row_index)
        over_under['rowIndex'] = int(row_index)
        moneylines.append(moneyline)
        over_unders.append(over_under)

    return jsonify({'moneylines': moneylines,
                    'over_unders': over_unders})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='7860', debug=True)