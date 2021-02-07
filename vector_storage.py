from .pyw_hnswlib import Hnswlib
import json
from .typings import Vector, VectorList, StringList, NearestNeighborList
from .embedding.model import EmbeddingModel
from typing import Callable, Optional, TypeVar

# Generic Type
T = TypeVar('T')

NUM_ELEMENTS = 20000
DIMS = 768
EF_CONSTRUCTION = 200
M = 100

class VectorStorage():

    def __init__(
        self, 
        path: str = None,
        dim=DIMS,
        num_elements=NUM_ELEMENTS
    ):
        self.dim = dim
        self.num_elements = num_elements
        self.embedder = EmbeddingModel()
        self.storage = Hnswlib(space='cosine', dim = dim)

        if path is not None:
            self.storage.load_index(path, max_elements=num_elements)
        else:
            self.storage.init_index(max_elements=num_elements, ef_construction = EF_CONSTRUCTION, M = M)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        self.storage.set_ef(150) # ef should always be > k

    def add_items(self, emb_batch, id_batch):
        self.storage.add_items(emb_batch, id_batch)

    def save(self, path: str):
        self.storage.save_index(path)


    def get_k_nearest(self, embeddings: VectorList, k: int) -> NearestNeighborList:
        '''
        embeddings (shape:N*dim). Returns a numpy array of (shape: N*K)
        '''
        labels, distances = self.storage.knn_query(embeddings, k)
        nearest: NearestNeighborList = []
        for row_i, row in enumerate(labels):
            nn = []
            for col_i, col in enumerate(row):
                dic = {}
                dic[col] = distances[row_i][col_i]
                nn.append(dic)
            nearest.append(nn)
        return nearest