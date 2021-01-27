import sys
sys.path.append("..")

from sentence_transformers import models, util, SentenceTransformer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os
import nvidia_smi
from ..typings import Vector

class EmbeddingModel():
    def __init__(self, lang = "de", device="cpu"):
        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        free_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).free

        # disable GPU usage if VRAM < 2GB
        if free_memory > 2e9:
            if torch.cuda.is_available():
                device = "cuda"

        # Initialize model based on selected language

        if lang == "de":
            # load BERT model from Hugging Face
            word_embedding_model = models.Transformer('bert-base-german-cased')

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),    # dimensions for the word embeddings
                pooling_mode_cls_token=False,                           # not use the first token (CLS token) as text representations
                pooling_mode_mean_tokens=True,                          # perform mean-pooling
                pooling_mode_max_tokens=False                           # not use max in each dimension over all tokens
            )

            # join BERT model and pooling to get the sentence transformer
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = device)
            self.model.max_seq_length = 512

        if lang == "en":
            # available pre-trained models: https://www.sbert.net/docs/pretrained-models/msmarco-v2.html
            self.model = SentenceTransformer('msmarco-distilbert-base-v2', device = device)
            self.model.max_seq_length = 512

    def encode(self, text:str) -> Vector:
        return (self.model.encode(text, convert_to_tensor=False, batch_size=8)).tolist()