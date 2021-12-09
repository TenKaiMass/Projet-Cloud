import pandas as pd


class FeatureRecipe:
    """
    Feature processing class
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.continuous = None
        self.categorical = None
        self.discrete = None
        self.datetime = None


class FeatureExtractor:
    """
    Feature Extractor class
    """

    def __init__(self, data: pd.DataFrame, flist: list):
        """
        Input: pandas.DataFrame, feature list to drop
        Output: X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """
