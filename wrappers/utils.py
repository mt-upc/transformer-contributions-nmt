import numpy as np
import pandas as pd

def spearmanr(x, y):
    """Compute Spearman rank's correlation bertween two attribution vectors.
        https://github.com/samiraabnar/attention_flow/blob/master/compute_corel_distilbert_sst.py"""

    x = pd.Series(x)
    y = pd.Series(y)
    assert x.shape == y.shape
    rx = x.rank(method='dense')
    ry = y.rank(method='dense')
    d = rx - ry
    dsq = np.sum(np.square(d))
    n = x.shape[0]
    coef = 1. - (6. * dsq) / (n * (n**2 - 1.))
    return [coef]

def get_normalized_rank(x):
    """Compute normalized [0,1] ranks. The higher the value, the higher the rank."""
    
    length_tok_sentence = x.shape
    x = pd.Series(x)
    rank = x.rank(method='dense')
    rank_normalized = rank/length_tok_sentence
    return rank_normalized