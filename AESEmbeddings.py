import AESData
import os
import json
import torch
from transformers import BertTokenizer, BertModel

class AESEmbeddings:
    """Abstraction for embedding generation and caching"""

    def __init__(self, data: AESData, model_name: str = "bert-base-uncased", tokenizer_name: str = "", max_length: int = -1):
        """Constructor. Requires model_name and tokenizer. 
        If tokenizer is not specified, it is assumed that tokenizer_name is a model_name.
        If max_length is not specified, it is assumed that maximum input length is the maximum input length of a model."""
        self.data: AESData = data
        self.model_name: str = model_name
        self.tokenizer_name: str = model_name
        if tokenizer_name:
            self.tokenizer_name: str = tokenizer_name
        self.model = None
        self.tokenizer = None
        self.model_loaded: bool = False
        self.max_length = max_length
        self.cache_path = "cached_embeddings"
        self.cache_filename = "{}-{}.pt"
    
    def load_model(self):
        """Loads model into memory. Called automatically when needed"""
        print("Loading {} model...".format(self.model_name))
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model_loaded = True
        if self.max_length == -1:
            self.max_length = self.tokenizer.model_max_length
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded

    def get_model_name(self) -> str:
        """Get name of the currently loaded model"""
        return self.model_name

    def get_tokenizer_name(self) -> str:
        """Get name of the currently loaded tokenizer"""
        return self.tokenizer_name

    def get_model_max_length(self) -> int:
        """Get maximum input length of currently loaded model"""
        if not self.model_loaded:
            self.load_model()
        return self.tokenizer.model_max_length

    def get_max_length(self) -> int:
        """Get maximum input length for current run"""
        return self.max_length

    def get_embeddings(self, id: int, rewrite=False):
        """Returns embeddings for essay with specified id and caches the result. 
        If rewrite option is set to True the embeddings will be regenerated again instead of reading the cache."""
        embeddings = None
        # Create directory for cache
        save_path = os.path.join(self.cache_path, self.model_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # Define path for saving tensors
        save_path = os.path.join(save_path, self.cache_filename.format(self.data.get_dataset_id(), id))
        # Read from cache if cache exists and not marked for rewrite
        if os.path.exists(save_path) and not rewrite:
            embeddings = torch.load(save_path)
        # Generate embeddings
        else:
            # Load model if not loaded
            if not self.model_loaded:
                self.load_model()
            # Tokenize
            text = self.data.get_essay(id)
            encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
            # Get embeddings
            embeddings = self.model(**encoded_input)
            # Save emdeddings
            torch.save(embeddings, save_path)
        return embeddings

    def cache_all_data(self, rewrite = False):
        """Generates embeddings and caches them for all essays in dataset.
        Rewrites existing cache if rewrite is True."""
        for i in range(self.data.count_essays()):
            self.get_embeddings(i, rewrite=rewrite)
            print("{}/{}".format(i,self.data.count_essays()))