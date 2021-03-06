import json
import os
from tqdm import tqdm
from .pyw_hnswlib import Hnswlib
from .typings import Vector, VectorList, StringList, NearestNeighborList
from .embedding.model import EmbeddingModel

class VectorStorage():

    def __init__(
        self,
        storage_location,
        max_elements = None,
        dim = 768,
        ef_construction = 200,
        m = 100,
        ef = 150,
        persist = True
    ):
        self.storage = Hnswlib(space='cosine', dim = dim)
        self.storage_location = storage_location
        self.max_elements = max_elements

        if os.path.isfile(storage_location):
            if max_elements:
                self.storage.load_index(storage_location, max_elements=max_elements)
            else:
                self.storage.load_index(storage_location)
        else:
            self.storage.init_index(max_elements=max_elements, ef_construction = ef_construction, M = m)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        self.storage.set_ef(ef) # ef should always be > k
        self.persist = persist

    def get_max_elements(self):
        return self.storage.get_max_elements()

    def get_current_count(self):
        return self.storage.get_current_count()

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

    def add_items_from_file(self, file_path, emb_func, get_id_func):
        total = 0
        exception_count = 0
        with open(file_path, 'r', encoding="utf-8") as data_file:
            emb_batch: VectorList = []
            id_batch: StringList = []

            for line in tqdm(data_file, total=self.max_elements):
                raw = json.loads(line)
                article_id = get_id_func(raw)
                emb = emb_func(raw)
                if emb is None or article_id is None:
                    exception_count += 1
                    continue
                emb_batch.append(emb)
                id_batch.append(article_id)
                total += 1

                if len(emb_batch) == 1000:
                    self.storage.add_items(emb_batch, id_batch)
                    emb_batch = []
                    id_batch = []

            if len(emb_batch) != 0:
                self.storage.add_items(emb_batch, id_batch)
        if self.persist:
            self.storage.save_index(self.storage_location)
            print(f"Done. Exception Count: {exception_count}. Total: {total}")

    def add_items_from_ids_file(self, file_path, emb_func):
        total = 0
        exception_count = 0
        with open(file_path, 'r', encoding="utf-8") as data_file:
            emb_batch: VectorList = []
            id_batch: StringList = []

            for line in data_file:
                article_id = line.strip()
                emb = emb_func(article_id)
                if emb is None:
                    exception_count += 1
                    continue
                emb_batch.append(emb)
                id_batch.append(article_id)
                total += 1

                if len(emb_batch) == 1000:
                    self.storage.add_items(emb_batch, id_batch)
                    emb_batch = []
                    id_batch = []

            if len(emb_batch) != 0:
                self.storage.add_items(emb_batch, id_batch)
        if self.persist:
            self.storage.save_index(self.storage_location)
            print(f"Done. Exception Count: {exception_count}. Total: {total}")