import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Create cross validation set
from sklearn.base import TransformerMixin, BaseEstimator

class SimpleSentenceEmbeddingGrader(BaseEstimator):
    def __init__(self, encoder, decoder_cls, decoder_kwargs):
        self.encoder = encoder
        self.decoder_cls = decoder_cls
        self.decoder_kwargs = decoder_kwargs
    def fit(self, X, y):
        """
        Fit the model to the data.
        Input:
            X: Essay text (list of strings). Shape: (n_samples,)
            y: Scores (array of floats). Shape: (n_samples,n_tasks)
        """
        self.n_tasks = y.shape[1]
        # Encode the essays
        X_embeddings = self.encoder.encode(X, show_progress_bar=True)
        # Fit the decoder
        self.decoders = []
        for i in range(self.n_tasks):
            decoder = self.decoder_cls(**self.decoder_kwargs)
            decoder.fit(X_embeddings, y[:,i])
            self.decoders.append(decoder)
        return self

    def predict(self, X):
        """
        Predict the scores for the essays.
        Input:
            X: Essay text (list of strings). Shape: (n_samples,)
        Output:
            y_pred: Predicted scores (array of floats). Shape: (n_samples,n_tasks)
        """
        # Encode the essays
        X_embeddings = self.encoder.encode(X, show_progress_bar=True)
        # Predict the scores
        y_pred = np.zeros((len(X), self.n_tasks))
        for i in range(self.n_tasks):
            y_pred[:,i] = self.decoders[i].predict(X_embeddings)
        return y_pred

class SimpleGrader(BaseEstimator):
    def __init__(self, decoder_cls, decoder_kwargs):
        self.decoder_cls = decoder_cls
        self.decoder_kwargs = decoder_kwargs
    def fit(self, X, y):
        """
        Fit the model to the data.
        Input:
            X: Essay text (list of strings). Shape: (n_samples,)
            y: Scores (array of floats). Shape: (n_samples,n_tasks)
        """
        self.n_tasks = y.shape[1]
        # Fit the decoder
        self.decoders = []
        for i in range(self.n_tasks):
            decoder = self.decoder_cls(**self.decoder_kwargs)
            decoder.fit(X, y[:,i])
            self.decoders.append(decoder)
        return self

    def predict(self, X):
        """
        Predict the scores for the essays.
        Input:
            X: Essay text (list of strings). Shape: (n_samples,)
        Output:
            y_pred: Predicted scores (array of floats). Shape: (n_samples,n_tasks)
        """
        # Predict the scores
        y_pred = np.zeros((len(X), self.n_tasks))
        for i in range(self.n_tasks):
            y_pred[:,i] = self.decoders[i].predict(X)
        return y_pred