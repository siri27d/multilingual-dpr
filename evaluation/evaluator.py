import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import faiss

class RetrievalEvaluator:
    """Evaluates retrieval performance using various metrics"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.tokenizer = model.tokenizer
    
    def evaluate(self, queries: List[Dict], passages: List[Dict]) -> Dict:
        """Evaluate retrieval performance"""
        print("🧪 Running evaluation...")
        
        # Encode all passages
        passage_embeddings = []
        passage_ids = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(passages), self.config.batch_size):
                batch_passages = passages[i:i + self.config.batch_size]
                batch_texts = [p['text'] for p in batch_passages]
                
                encodings = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=256, return_tensors='pt'
                ).to(self.config.device)
                
                embeddings = self.model.encode_passages(
                    encodings['input_ids'], 
                    encodings['attention_mask']
                )
                passage_embeddings.append(embeddings.cpu().numpy())
                passage_ids.extend([p['id'] for p in batch_passages])
        
        passage_embeddings = np.vstack(passage_embeddings)
        
        # Build FAISS index
        index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        faiss.normalize_L2(passage_embeddings)
        index.add(passage_embeddings.astype('float32'))
        
        # Evaluate each query
        recall_at_1 = 0
        recall_at_5 = 0
        recall_at_10 = 0
        reciprocal_ranks = []
        
        with torch.no_grad():
            for query in tqdm(queries, desc="Evaluating queries"):
                query_text = query['text']
                positive_id = query['positive_passage_ids'][0]
                
                # Encode query
                encoding = self.tokenizer(
                    [query_text], padding=True, truncation=True,
                    max_length=256, return_tensors='pt'
                ).to(self.config.device)
                
                query_embedding = self.model.encode_queries(
                    encoding['input_ids'], 
                    encoding['attention_mask']
                ).cpu().numpy()
                
                # Normalize for cosine similarity
                faiss.normalize_L2(query_embedding)
                
                # Search
                distances, indices = index.search(query_embedding.astype('float32'), 10)
                
                # Find rank of positive passage
                try:
                    positive_idx = passage_ids.index(positive_id)
                    rank = np.where(indices[0] == positive_idx)[0]
                    if len(rank) > 0:
                        rank = rank[0] + 1
                    else:
                        rank = len(passage_ids) + 1  # Not found in top k
                    
                    # Update metrics
                    if rank == 1:
                        recall_at_1 += 1
                    if rank <= 5:
                        recall_at_5 += 1
                    if rank <= 10:
                        recall_at_10 += 1
                    
                    reciprocal_ranks.append(1.0 / rank)
                except ValueError:
                    reciprocal_ranks.append(0.0)
        
        # Compute final metrics
        recall_at_1 /= len(queries)
        recall_at_5 /= len(queries)
        recall_at_10 /= len(queries)
        mrr = np.mean(reciprocal_ranks)
        
        results = {
            'recall@1': float(recall_at_1),
            'recall@5': float(recall_at_5),
            'recall@10': float(recall_at_10),
            'mrr': float(mrr),
            'num_queries': len(queries),
            'num_passages': len(passages)
        }
        
        return results