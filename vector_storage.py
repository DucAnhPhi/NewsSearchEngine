import json
import os
from .pyw_hnswlib import Hnswlib
from .typings import Vector, VectorList, StringList, NearestNeighborList
from .embedding.model import EmbeddingModel

class VectorStorage():

    def __init__(
        self,
        storage_location,
        max_elements,
        dim = 768,
        ef_construction = 200,
        m = 100,
        ef = 150
    ):
        self.storage = Hnswlib(space='cosine', dim = dim)
        self.storage_location = storage_location

        if os.path.isfile(storage_location):
            self.storage.load_index(storage_location, max_elements=max_elements)
        else:
            self.storage.init_index(max_elements=max_elements, ef_construction = ef_construction, M = m)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        self.storage.set_ef(ef) # ef should always be > k

    def get_k_nearest(self, embedding: Vector, k: int) -> NearestNeighborList:
        '''
        embeddings (shape:N*dim). Returns a numpy array of (shape: N*K)
        '''
        labels, distances = self.storage.knn_query([embedding], k)
        nearest: NearestNeighborList = []
        for row_i, row in enumerate(labels):
            nn = []
            for col_i, col in enumerate(row):
                dic = {}
                dic[col] = distances[row_i][col_i]
                nn.append(dic)
            nearest.append(nn)
        return nearest

    def add_items_from_file(self, file_path, emb_func):
        with open(file_path, 'r', encoding="utf-8") as data_file:
            emb_batch: VectorList = []
            id_batch: StringList = []

            for line in data_file:
                raw = json.loads(line)
                article_id = raw["id"]
                emb = emb_func(raw)
                if emb == None:
                    continue
                emb_batch.append(emb)
                id_batch.append(article_id)

                if len(emb_batch) == 1000:
                    self.storage.add_items(emb_batch, id_batch)
                    emb_batch = []
                    id_batch = []

            if len(emb_batch) != 0:
                self.storage.add_items(emb_batch, id_batch)
        self.storage.save_index(self.storage_location)