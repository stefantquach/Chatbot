import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models
from tensorflow.keras import preprocessing , utils
import os

# WIP

class Seq2Seq_Model:
    def __init__(vocab_size, context_size, layers=1, embedding_matrix=None):
        if embedding_matrix is not None:
            self.embedding = layers.Embedding(vocab_size, context_size, weights=embedding_matrix, trainable=False)
        else:
            self.embedding = layers.Embedding(vocab_size, context_size)
