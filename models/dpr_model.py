import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple

class DualEncoder(nn.Module):
    def __init__(self, model_name: str, projection_dim: int = 768):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.passage_encoder = AutoModel.from_pretrained(model_name)
        
        # Projection layers
        self.query_projection = nn.Linear(self.query_encoder.config.hidden_size, projection_dim)
        self.passage_projection = nn.Linear(self.passage_encoder.config.hidden_size, projection_dim)
        
        # Initialize projections
        self._init_weights(self.query_projection)
        self._init_weights(self.passage_projection)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, query_input_ids, query_attention_mask, 
                passage_input_ids, passage_attention_mask):
        # Encode queries
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        query_embeddings = self._mean_pooling(query_outputs, query_attention_mask)
        query_embeddings = self.query_projection(query_embeddings)
        query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        # Encode passages
        passage_outputs = self.passage_encoder(
            input_ids=passage_input_ids,
            attention_mask=passage_attention_mask
        )
        passage_embeddings = self._mean_pooling(passage_outputs, passage_attention_mask)
        passage_embeddings = self.passage_projection(passage_embeddings)
        passage_embeddings = nn.functional.normalize(passage_embeddings, p=2, dim=1)
        
        return query_embeddings, passage_embeddings
    
    def encode_queries(self, input_ids, attention_mask):
        query_outputs = self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        query_embeddings = self._mean_pooling(query_outputs, attention_mask)
        query_embeddings = self.query_projection(query_embeddings)
        return nn.functional.normalize(query_embeddings, p=2, dim=1)
    
    def encode_passages(self, input_ids, attention_mask):
        passage_outputs = self.passage_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        passage_embeddings = self._mean_pooling(passage_outputs, attention_mask)
        passage_embeddings = self.passage_projection(passage_embeddings)
        return nn.functional.normalize(passage_embeddings, p=2, dim=1)
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class MultilingualDPR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DualEncoder(config.model_name, config.projection_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def forward(self, batch):
        return self.model(
            query_input_ids=batch['query_input_ids'],
            query_attention_mask=batch['query_attention_mask'],
            passage_input_ids=batch['positive_passage_input_ids'],
            passage_attention_mask=batch['positive_passage_attention_mask']
        )