from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction
from typing import List, Tuple, Optional


class RandomForestPredictor(Predictor):
    def __init__(self, model: RandomForestClassifier, team_encoding: OneHotEncoder) -> None:
        self.model = model
        self.team_encoding = team_encoding

    def predict(self, fixture: Fixture) -> Prediction:
        encoded_home_name = self.__encode_team(fixture.home_team)
        #encoding the input home team
        encoded_away_name = self.__encode_team(fixture.away_team)
        #encoding the input away team

        if encoded_home_name is None:
            # no home team available the away team wins
            return Prediction(outcome=Outcome.AWAY)
        if encoded_away_name is None:
            # no away team available the home team wins
            return Prediction(outcome=Outcome.HOME)

        # create array from encoded team names

        x: NDArray[float64] = np.concatenate([encoded_home_name, encoded_away_name], 1)

        # prediction

        pred = self.model.predict(x)

        if pred > 0:
            return Prediction(outcome=Outcome.HOME)
        elif pred < 0:
            return Prediction(outcome=Outcome.AWAY)
        else:
            return Prediction(outcome=Outcome.DRAW)

    def __encode_team(self, team: Team) -> Optional[NDArray[float64]]:
        # check team available from fixture
        try:
            # necessary to convert categorical features into numerical ones
            # feature encoding to convert team names into numerical values
            result: NDArray[float64] = self.team_encoding.transform(np.array(team.name).reshape(-1, 1))
            return result
        except ValueError:
            return None


def build_random_forest_model(results: List[Result]) -> Tuple[RandomForestClassifier, OneHotEncoder]:
    home_names = np.array([r.fixture.home_team.name for r in results])
    away_names = np.array([r.fixture.away_team.name for r in results])
    home_goals = np.array([r.home_goals for r in results])
    away_goals = np.array([r.away_goals for r in results])

    team_names = np.array(list(home_names) + list(away_names)).reshape(-1, 1)
    team_encoding = OneHotEncoder(sparse=False).fit(team_names)

    encoded_home_names = team_encoding.transform(home_names.reshape(-1, 1))
    encoded_away_names = team_encoding.transform(away_names.reshape(-1, 1))

    x: NDArray[float64] = np.concatenate([encoded_home_names, encoded_away_names], 1)
    y = np.sign(home_goals - away_goals)
    #adjusting hyperparameters
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, min_samples_leaf=4)
    model.fit(x, y)

    return model, team_encoding


def train_random_forest_predictor(results: List[Result]) -> Predictor:
    model, team_encoding = build_random_forest_model(results)
    return RandomForestPredictor(model, team_encoding)
