import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


class LicenseplateDetector():
    def __init__(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.clf = self.train_model()
        self.iris_type = {
            0: 'setosa',
            1: 'versicolor',
            2: 'virginica'
        }

    def detect(self, features: dict):
        X = [features['sepal_l'], features['sepal_w'], features['petal_l'], features['petal_w']]
        prediction = self.clf.predict_proba([X])
        return {'class': self.iris_type[np.argmax(prediction)],
                'probability': round(max(prediction[0]), 2)}
