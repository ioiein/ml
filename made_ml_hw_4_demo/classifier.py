import joblib
import pandas as pd

class Classifier(object):
    def __init__(self):
        self.model = joblib.load("model_dump.pkl")
        self.score_stat = joblib.load("score_stats_dump.pkl")

    def predict_proba(self, team_home, team_away):
        d = {'team_home': [team_home], 'team_away': [team_away]}
        match_to_predict = pd.DataFrame(data=d)
        cols = ["team_home_mean_target", "team_away_mean_target"]
        match_to_predict["team_home_mean_target"] = match_to_predict.team_home.map(self.score_stat.groupby("team_home").target.mean())
        match_to_predict["team_away_mean_target"] = match_to_predict.team_away.map(self.score_stat.groupby("team_away").target.mean())
        submit = pd.DataFrame(self.model.predict_proba(match_to_predict[cols].fillna(0)), columns=["draw", "home_team_win", "away_team_lose"])
        result = [submit["home_team_win"][0], submit["draw"][0], submit["away_team_lose"][0]]
        return result