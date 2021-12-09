import numpy as np


class ModelBuilder:
    """
    Class for train and print results of ml model
    """

    def __init__(self, model_path: str = None, save: bool = None):
        pass

    def __repr__(self):
        pass

    def train(self, X, Y):
        pass

    def predict_test(self, X) -> np.ndarray:
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
