import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextLowercaser(BaseEstimator, TransformerMixin):
    """
    A custom Scikit-learn transformer for lowercasing text data.
    """

    def fit(self, X, y=None):
        return self  # Nothing to learn in this step

    def transform(self, X):
        if not isinstance(X, list):
            X = X.tolist() if isinstance(X, pd.Series) else list(X)
        return [text.lower() for text in X]
