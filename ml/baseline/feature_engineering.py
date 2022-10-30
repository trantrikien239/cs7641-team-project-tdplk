from sklearn.base import TransformerMixin

class FeatureEngineeringPipieline(TransformerMixin):

    def fit(X, y=None):
        # will contain all features
        pass

    def transform(X):
        # will contain all features
        pass

    def fit_transform(X):
        self.fit(X,y=None)
        return self.transform(X)

    def feature_1(text):
        return new_feature

    def feature2(text):
        return new_feature