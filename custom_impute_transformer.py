from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class CustomImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_type=None):
        self.impute_vals = None
        self.impute_type = impute_type

    def fit(self, X, y=None):
        if self.impute_type == "median":
            self.impute_vals = X.median()
        else:
            self.impute_vals = X.mean()
        return self

    def transform(self, X):
        return X.fillna(self.impute_vals)