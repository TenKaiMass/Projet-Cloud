import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class DataHandler:
    """
        Get data from sources
        construction des fichier avec 2 fichier en arg qu'on va grouper
    """

    def __init__(self, csvfile1, csvfile2):
        self.csvfile1 = self.getCsvfile(csvfile1)
        self.csvfile2 = self.getCsvfile(csvfile2)
        self.gouped_data = pd.concat([self.csvfile1, self.csvfile2])

    def getCsvfile(self, filename: str):
        return pd.read_csv(filename)


class FeatureRecipe:
    """
    Feature processing class
    cree le dataframe avec le DataHandler(csv1,csv2)
    """

    def __init__(self, data: pd.DataFrame, continus: bool, type_data: bool):
        self.data = data
        self.continuous = continus
        self.categorical = type_data
        self.discrete = not continus
        #self.datetime = None


class FeatureExtractor:
    """
    Feature Extractor class
    avec FeatureRecipe(), flist
    """

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.y = data[target]
        self.X = self.getFlist(data, target)
        self.X_train, self.X_test, self.y_train, self.y_test = self.getSplitData(
            self.X, self.y)
        """
        Input: pandas.DataFrame, feature list to drop
        Output: X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """

    def getSplitData(self, X, y):
        return train_test_split(X, y, test_size=0.3)

    def getFlist(self, data, target):
        return data.drop(target, axis=1)


class ModelBuilder:
    """
    Class for train and print results of ml model
    """

    def __init__(self, model_path: str, save: bool, data: FeatureExtractor):
        self.model = make_pipeline(
            StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        self.data = data

        pass

    def __repr__(self):
        pass

    def train(self, X, Y):
        self.model.fit(X, Y)

        pass

    def predict_test(self, X) -> np.ndarray:
        self.model.predict(X)

        pass

    def predict_from_dump(self, X) -> np.ndarray:
        pass

    def save_model(self, path: str):
        # with the format : ‘model_{}_{}’.format(date)
        pass

    def print_accuracy(self):
        pass

    def load_model(self):
        try:
            # load model
            pass
        except:
            pass
