
from datetime import datetime
from os import mkdir

import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from joblib import dump, load


class DataHandler:
    """
        Get data from sources
        construction des fichier avec 2 fichier en arg qu'on va grouper
    """

    def __init__(self, csvfile1, csvfile2):
        self.csvfile1 = self.getCsvfile(csvfile1)
        self.csvfile2 = self.getCsvfile(csvfile2)
        self.grouped_data = pd.concat([self.csvfile1, self.csvfile2])

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
        # self.datetime = None


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
        self.save = save
        self.model_path = model_path
        self.model = self.loadModel()
        self.data = data

    def __repr__(self):
        pass

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predictTest(self, X):  # -> np.ndarray:
        self.model.predict(X)

    def predictFromDump(self, X) -> np.ndarray:
        pass

    def saveModel(self, model_name: str):
        date = datetime.now()
        path = '../Model_Save/'
        extension = ".joblib"
        d = "_{}_{}_{}".format(date.day, date.month, date.year)
        path = "{}{}_{}{}".format(path, model_name, d, extension)
        try:
            dump(self.model, path)
        except FileNotFoundError:
            mkdir("../Model_Save")
            dump(self.model, path)
            print(f"le model {model_name} à bien été sauvegarder")
        else:
            print(f"le model {model_name} à bien été sauvegarder")

        # with the format : ‘model_{}_{}’.format(date)

    def printAccuracy(self):
        predict = self.predictTest(self.data.X_train)
        print(accuracy_score(predict, self.data.y_test)*100)

    def loadModel(self):
        model_default = make_pipeline(
            StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        if self.save == False:
            print("save = False, on charge donc le Model par defaut")
            return model_default
        try:
            # load model
            return load(self.model_path)

        except FileNotFoundError:
            print(
                f"Erreur, Il n'existe aucun model du nom de {self.model_path}")
            print("Chargement du default model")
            return model_default
