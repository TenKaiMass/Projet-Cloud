from ml import FeatureExtractor, FeatureRecipe
from __init__ import DataHandler


def DataManager(d: DataHandler = None, fr: FeatureRecipe = None, fe: FeatureExtractor = None):
    """
    Fonction qui lie les 3 premières classes de la pipeline et qui return FeatureExtractor.split(0.1)
    """
    pass


# on appelera la fonction DataManager() de la facon suivante :
X_train, X_test, y_train, y_test = DataManager()
