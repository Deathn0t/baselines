from sklearn.metrics import r2_score

class BaseRegressor:
    """BaseRegressor

    Args:
        load_data (func): returns (trainX, trainy), (validX, validy).
    """
    def __init__(self, regressor, load_data_fn, score_fn, n_jobs=-1, verbose=1,):
        self.name = type(self).__name__
        self.load_data_fn = load_data_fn
        self.score_fn = score_fn
        self.regressor = regressor

    def run(self):
        (tX, ty), (vX, vy) = self.load_data_fn()

        regr = self.regressor
        regr.fit(tX, ty)

        ty_pred = regr.predict(tX)
        vy_pred = regr.predict(vX)

        score_train = self.score_fn(ty, ty_pred)
        score_valid = self.score_fn(vy, vy_pred)

        print(f"Method: {self.name}")
        print(f"    {self.score_fn.__name__} on tX: {score_train}")
        print(f"    {self.score_fn.__name__} on vX: {score_valid}")