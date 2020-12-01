from sentence_transformers import models, util, SentenceTransformer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os
import sys
import nvidia_smi

class EmbeddingModel():
    def __init__(self):
        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        free_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).free

        # If you run into GPU memory problems, uncomment the following line:

        # os.environ["CUDA_VISIBLE_DEVICES"]=""

        # disable GPU usage if VRAM < 2GB
        if free_memory < 1500000000:
            os.environ["CUDA_VISIBLE_DEVICES"]=""

        # load BERT model from Hugging Face
        word_embedding_model = models.Transformer(
        'bert-base-german-cased')

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

        # join BERT model and pooling to get the sentence transformer
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])