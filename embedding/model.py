from sentence_transformers import models, SentenceTransformer
from ..typings import Vector

class EmbeddingModel():
    # make sure you have > 2GB of free VRAM to enable CUDA
    def __init__(self, lang, device="cpu"):
        supported_languages = ["de", "en"]
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
            # available pre-trained models: https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
            self.model = SentenceTransformer('stsb-distilbert-base', device = device)
            self.model.max_seq_length = 512

        if lang not in supported_languages:
            raise ValueError(f"language: {lang} not supported.")

    def encode(self, text:str) -> Vector:
        return (self.model.encode(text, convert_to_tensor=False, batch_size=8)).tolist()