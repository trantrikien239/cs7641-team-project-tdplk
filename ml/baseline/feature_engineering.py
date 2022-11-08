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

    # Number of unique words(Vocabulary)
    def unique_word(text):
        return new_feature
    # Number of spelling error(Conventions)
    def spelling_error(text):
        return new_feature
    # Number of capitalization/punctuation error(Conventions)
    def capital_punc_error(text):
        return new_feature
    # #of grammar/tense error
    def grammar_tense_error(text):
        return new_feature