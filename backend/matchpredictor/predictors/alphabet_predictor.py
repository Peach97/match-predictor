from matchpredictor.matchresults.result import Fixture, Outcome
from matchpredictor.predictors.predictor import Prediction, Predictor


class AlphabetPredictor(Predictor):
    def predict(self, fixture: Fixture) -> Prediction:
        home_letter = fixture.home_team.name[0]
        away_letter = fixture.away_team.name[0]
        if home_letter > away_letter:
            return Prediction(outcome=Outcome.HOME)
        else:
            return Prediction(outcome=Outcome.AWAY)
