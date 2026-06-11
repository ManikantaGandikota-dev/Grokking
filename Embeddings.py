#In order to train the model, It is necessary to convert text to embeddings
import numpy as np

def EEmbeddings(p, d_embed):
    E = np.random.randn(p,d_embed)/np.sqrt(d_embed)
    return E