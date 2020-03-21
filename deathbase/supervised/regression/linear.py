from sklearn.linear_model import LinearRegression
from deathbase.supervised.regression.base import BaseRegressor

class Linear(BaseRegressor):
    def __init__(self, *args, **kwargs):
        regressor = LinearRegression()
        super().__init__(regressor, *args, **kwargs)