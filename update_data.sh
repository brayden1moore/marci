#!/bin/bash
export PATH=/opt/homebrew/bin/:$PATH 
echo "Updating MARCI!"
cd "/Users/braydenmoore/Library/Mobile Documents/com~apple~CloudDocs/Code/marci"
python3 "Source/Build/update.py"
python3 "get_record.py"
echo "Committing to repo!"
git add "Source/Data/gbg_and_odds_this_year.csv"
git add "Source/Data/gbg_this_year.csv"
git add "Source/Data/results.csv"
git add "Source/Data/record.json"
git add "Source/Data/lines.json"
git add "Source/Pickles"
git add "Static"
git add "Templates"
git commit -m "Update with up to date data"
git push all main