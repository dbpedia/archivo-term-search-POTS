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
        self.type = "FAISS"
        self.embedding_function = embedding_function
        self.embedding_method = embedding_method
        self.metadata = []  # Initialize metadata attribute
        if not os.path.isdir(path_index):
            #print("Creating new index...")
            self.loadTerms(endpoint)
            self.index, self.metadata = self.create()  # Store metadata from creation
            #print("New index created!")
        else: 
            #print("Loading existing index...")
            self.index = FAISS.load_local(self.path_index, self.embedding_function, allow_dangerous_deserialization=True)
            self.metadata = self.load_metadata()  # Load metadata from a file or other source
            #print("Existing index Loaded!")

    def load_metadata(self):
        # Implement loading of metadata (e.g., from a file)
        metadata_file = self.path_index + '_metadata.npy'
        if os.path.exists(metadata_file):
            return np.load(metadata_file, allow_pickle=True).tolist()
        else:
            return []

    def save_metadata(self):
        # Implement saving of metadata (e.g., to a file)
        metadata_file = self.path_index + '_metadata.npy'
        np.save(metadata_file, self.metadata, allow_pickle=True)

    def create(self):
        keys, metadata = self.prepare_dataset(self.terms)
        #print("EMBEDDING:", keys, "using", self.embedding_function)
        #print(keys)
        #print(metadata)
        faiss = FAISS.from_texts(keys, self.embedding_function, metadata)
        faiss.save_local(self.path_index)
        self.metadata = metadata  # Store metadata
        self.save_metadata()  # Save metadata
        return faiss, metadata
    def loadTerms(self,endpoint):
        pass

    def prepare_dataset(self,terms):
        keys = []
        metadata = []
        # print(terms)
        if self.embedding_method == "Term":
            for term in terms:
                keys_r = self.endpoint.get_labels(term['?term'])
                for key in keys_r:
                    keys.append(key[0])
                    metadata.append(term)
                if not keys_r:
                    keys.append(f"{term['?term']}: None")
                    metadata.append(term)
        elif self.embedding_method == "Term+Description":
            for term in terms:
                description = self.endpoint.get_description(term['?term'])
                if description:
                    text_to_embed = f"{term['?term']}: {description}"
                else:
                    text_to_embed = f"{term['?term']}: None"
                keys.append(text_to_embed)
                metadata.append(term)
        elif self.embedding_method == "Term+Subclasses":
            for term in terms:
                subclasses = self.endpoint.get_subclasses(term['?term'])
                if subclasses:
                    text_to_embed = f"{term['?term']}: {subclasses}"
                else:
                    text_to_embed = f"{term['?term']}: None"
                keys.append(text_to_embed)
                metadata.append(term)
            
        elif self.embedding_method == "Term+Superclasses":
            for term in terms:
                superclasses = self.endpoint.get_superclasses(term['?term'])
                if superclasses:
                    text_to_embed = f"{term['?term']}: {superclasses}"
                else:
                    text_to_embed = f"{term['?term']}: None"
                keys.append(text_to_embed)
                metadata.append(term)
            
        elif self.embedding_method == "Term+SeeAlso":
            for term in terms:
                seealso = self.endpoint.get_seealso(term['?term'])
                if seealso:
                    text_to_embed = f"{term['?term']}: {seealso}"
                else:
                    text_to_embed = f"{term['?term']}: None"
                keys.append(text_to_embed)
                metadata.append(term)
        return keys,metadata

    def search(self,term,hits=50):
        #print("Searching for: "+term)
        results = []
        results = self.index.similarity_search_with_score(term,hits)

        r = []
        n = 0
        for i in results:
            #print("Result:", str(n+1), i[0].metadata["?label"], float(i[1]))
            metadata = i[0].metadata
            r.append({'label':metadata['?label'],'content':metadata,'score':float(i[1])})
            n += 1
        results = r
        return results
    
    def exists(self):
        try:
            if os.path.exists(self.path_index) and FAISS.load_local(self.path_index, self.embedding_function) != None:
                return True
        except:
            return False

    def extract_vectors(self):
        return self.index.index.reconstruct_n(0, self.index.index.ntotal)
    
    def visualize_index(self, method='pca', n_components=2, output_file='faiss_index.png', label_size=5, dot_size=10):
        
        vectors = self.extract_vectors()

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError("Unsupported dimensionality reduction method: choose 'pca' or 'tsne'.")

        reduced_vectors = reducer.fit_transform(vectors)

        # Ensure that metadata is loaded or exists
        if not self.metadata:
            raise ValueError("Metadata is not available for visualization.")
        #print(output_file)
        plt.figure(figsize=(15, 15))
        if n_components == 2:
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], s=dot_size, alpha=0.5)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.title(f"2D Visualization of FAISS Index using {method.upper()}")

            # Annotate each point with the label
            for i, label in enumerate(self.metadata):
                label = label["?label"]
                #print(label)
                plt.annotate(
                    label,
                    (reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                    fontsize=label_size,
                    ha='center',  # Horizontal alignment
                    va='top',  # Vertical alignment
                    xytext=(0, -10),  # Offset label 10 points below the dot
                    textcoords='offset points'
                )
        elif n_components == 3:
            ax = plt.axes(projection='3d')
            sc = ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], s=dot_size, alpha=0.5)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            plt.title(f"3D Visualization of FAISS Index using {method.upper()}")

            # Annotate each point with the label (3D annotations can be cluttered)
            for i, label in enumerate(self.metadata):
                ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], label, fontsize=label_size)

        plt.savefig(output_file)
        plt.show()
        plt.close()
class TBoxIndex(Index):
    def __init__(self, endpoint, normalizer, embedding_function, embedding_method="Term", search_type="classes") -> None:
        if endpoint.ontology:
            path_index = f"index/temp/{endpoint.ontology}_{search_type}/t_box_index/faiss"
        else:
            path_index = f"index/temp/{search_type}/t_box_index/faiss"
        super().__init__(path_index, endpoint, normalizer, embedding_function, embedding_method)
    
    def loadTerms(self,endpoint):
        self.terms = endpoint.listTerms()

class ABoxIndex(Index):
    def __init__(self, endpoint, normalizer) -> None:
        if endpoint.ontology:
            path_index = f"index/temp/{endpoint.ontology}/a_box_index/faiss"
        else:
            path_index = "index/temp/a_box_index/faiss"
        super().__init__(path_index, endpoint, normalizer)
    
    def loadTerms(self,endpoint):
        self.terms = endpoint.listResources()
