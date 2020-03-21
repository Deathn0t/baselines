from sklearn.ensemble import GradientBoostingRegressor

from deathbase.supervised.regression.base import BaseRegressor

class GradientBoosting(BaseRegressor):
    def __init__(self, *args, **kwargs):
        regressor = GradientBoostingRegressor(verbose=1)
        super().__init__(regressor, *args, **kwargs)