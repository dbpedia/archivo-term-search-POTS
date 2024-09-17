import os
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class Index:
    def __init__(self, path_index, endpoint, normalizer, embedding_function, embedding_method="Term") -> None:
        self.path_index = path_index
        #print("path index is", path_index)
        self.endpoint = endpoint
        self.normalizer = normalizer
        self.type = "WEAVIATE"
        self.embedding_function = embedding_function
        self.embedding_method = embedding_method
        self.create()

    def load_metadata(self):
        pass

    def save_metadata(self):
        pass

    def create(self):
        pass
    def loadTerms(self,endpoint):
        # ignore this one
        pass

    def prepare_dataset(self,terms):
        pass
        
    
    def exists(self):
        pass

    def extract_vectors(self):
        pass
    
    def visualize_index(self, method='pca', n_components=2, output_file='faiss_index.png', label_size=5, dot_size=10):
        pass
class TBoxIndex(Index):
    def __init__(self, endpoint, normalizer, embedding_function, embedding_method="Term", search_type="classes") -> None:
        if endpoint.ontology:
            path_index = f"index/temp/{endpoint.ontology}_{search_type}/t_box_index/faiss"
        else:
            path_index = f"index/temp/{search_type}/t_box_index/faiss"
        super().__init__(path_index, endpoint, normalizer, embedding_function, embedding_method)
    
    def loadTerms(self,endpoint):
        self.terms = endpoint.listTerms()
