from sklearn.ensemble import RandomForestRegressor
from deathbase.supervised.regression.base import BaseRegressor

class RandomForest(BaseRegressor):
    def __init__(self, *args, **kwargs):
        regressor = RandomForestRegressor(n_jobs=-1, verbose=1)
        super().__init__(regressor, *args, **kwargs)