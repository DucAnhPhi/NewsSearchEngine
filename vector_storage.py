from .pyw_hnswlib import Hnswlib
import json
from .typings import Vector, VectorList, StringList, NearestNeighborList
from .embedding.model import EmbeddingModel
from typing import Callable, TypeVar

# Generic Type
T = TypeVar('T')

NUM_ELEMENTS = 20000
DIMS = 768
EF_CONSTRUCTION = 200
M = 16

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
        self.storage.set_ef(20) # ef should always be > k


    def add_items_from_file(self, path: str, embedding_func: Callable[[T], Vector]):
        with open(path, 'r') as data_file:
            emb_batch: VectorList = []
            id_batch: StringList = []

            for line in data_file:
                article = json.loads(line)
                emb = embedding_func(article)
                emb_batch.append(emb)
                id_batch.append(article["id"])

                if len(emb_batch) is 1000:
                    self.storage.add_items(emb_batch, id_batch)
                    emb_batch = []
                    id_batch = []

            if len(emb_batch) is not 0:
                self.storage.add_items(emb_batch, id_batch)

        self.storage.save_index('data/storage.bin')


    def get_k_nearest(self, data: StringList, k: int) -> NearestNeighborList:
        embeddings = [self.embedder.encode(s) for s in data]
        labels, distances = self.storage.knn_query(embeddings, k)
        nearest: NearestNeighborList = []
        for row_i, row in enumerate(labels):
            nn = []
            for col_i, col in enumerate(row):
                dic = {}
                dic[col] = distances[row_i][col_i]
                nn.append(dic)
            nearest = nearest + nn
        return nearest