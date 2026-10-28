"""Preprocessing helpers for the CHD risk model."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SmokerAwareCigsImputer(BaseEstimator, TransformerMixin):
    """Fill missing ``cigsPerDay`` using smoking status.

    Non-smokers smoke 0 cigarettes a day by definition. Smokers with a missing
    value get the median ``cigsPerDay`` of smokers in the training data. A plain
    overall median would be 0 (about half the cohort does not smoke) and would
    wrongly record smokers as smoking nothing.
    """

    def __init__(self, smoker_col="currentSmoker", cigs_col="cigsPerDay"):
        self.smoker_col = smoker_col
        self.cigs_col = cigs_col

    def fit(self, X, y=None):
        smokers = X[self.smoker_col] == 1
        self.smoker_median_ = float(X.loc[smokers, self.cigs_col].median())
        return self

    def transform(self, X):
        X = X.copy()
        missing = X[self.cigs_col].isna()
        X.loc[missing & (X[self.smoker_col] == 0), self.cigs_col] = 0.0
        X.loc[missing & (X[self.smoker_col] == 1), self.cigs_col] = self.smoker_median_
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
