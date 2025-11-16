import numpy as np
import torch
import faiss
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import os

class SemanticClusterManager:
    """Implements our Semantic-Adaptive Clustering innovation"""
    
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.cluster_history = []
        
    def perform_semantic_clustering(self, passages: List[Dict], current_epoch: int) -> Tuple[Any, Dict]:
        """Perform adaptive semantic clustering across all languages"""
        print(f"🔄 Performing semantic clustering for epoch {current_epoch}...")
        
        # Encode all passages in shared multilingual space
        passage_texts = [p['text'] for p in passages]
        passage_embeddings = self._encode_passages_batch(passage_texts)
        
        # Calculate adaptive cluster size based on training progress
        n_clusters = self._calculate_adaptive_cluster_size(current_epoch, passage_embeddings.shape[0])
        
        print(f"   Using {n_clusters} clusters for {len(passages)} passages")
        
        # Perform k-means clustering
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(self.config.cluster_batch_size, len(passage_embeddings)),
            n_init=3
        )
        
        cluster_labels = kmeans.fit_predict(passage_embeddings.astype('float32'))
        
        # Build FAISS index for efficient similarity search
        index = self._build_faiss_index(passage_embeddings)
        
        # Prepare metadata
        metadata = {
            'passage_ids': [p['id'] for p in passages],
            'languages': [p.get('language', 'en') for p in passages],
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'epoch': current_epoch,
            'n_clusters': n_clusters,
            'silhouette_score': self._calculate_silhouette_score(passage_embeddings, cluster_labels)
        }
        
        # Store cluster history for visualization
        self.cluster_history.append({
            'epoch': current_epoch,
            'n_clusters': n_clusters,
            'silhouette_score': metadata['silhouette_score']
        })
        
        print(f"✅ Clustering completed with {n_clusters} clusters, silhouette: {metadata['silhouette_score']:.4f}")
        
        return index, metadata
    
    def _calculate_adaptive_cluster_size(self, current_epoch: int, num_passages: int) -> int:
        """Our Innovation: Adaptive cluster sizing based on training progress"""
        base_clusters = self.config.initial_clusters
        
        if current_epoch <= 3:
            # Early training: use fewer, broader clusters
            clusters = base_clusters
        elif current_epoch <= 8:
            # Middle training: gradually increase clusters
            growth = (current_epoch - 3) * self.config.cluster_growth_rate
            clusters = min(base_clusters + growth, self.config.max_clusters)
        else:
            # Late training: use maximum clusters for fine-grained negatives
            clusters = self.config.max_clusters
        
        # Ensure we don't have more clusters than passages
        clusters = min(clusters, num_passages // 10)
        
        return max(clusters, 50)  # Minimum 50 clusters
    
    def _encode_passages_batch(self, texts: List[str]) -> np.ndarray:
        """Encode passages in batches to avoid memory issues"""
        embeddings = []
        batch_size = 32
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                ).to(self.config.device)
                
                # Encode
                batch_embeddings = self.model.encode_passages(
                    encodings['input_ids'],
                    encodings['attention_mask']
                )
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for efficient similarity search"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        return index
    
    def _calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate clustering quality score"""
        if len(np.unique(labels)) > 1:
            # Use sample for large datasets
            sample_size = min(1000, len(embeddings))
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            return silhouette_score(embeddings[indices], labels[indices])
        return -1.0
    
    # In training/trainer.py - during training
  def _get_hard_negatives_for_batch(self, batch, cluster_metadata):
    hard_negatives_batch = []  # Will store hard negatives for each query in batch
    
    for i, query_data in enumerate(batch):
        # Get the POSITIVE passage for this query
        positive_passage_id = query_data['positive_passage_id']  # e.g., 'passage_156'
        
        # Find which cluster the positive passage belongs to
        positive_passage_idx = cluster_metadata['passage_ids'].index(positive_passage_id)
        positive_cluster_id = cluster_metadata['cluster_labels'][positive_passage_idx]
        # positive_cluster_id might be 45 (meaning passage_156 is in cluster 45)
        
        # Find ALL other passages in the SAME cluster
        same_cluster_passage_ids = []
        for idx, cluster_id in enumerate(cluster_metadata['cluster_labels']):
            passage_id = cluster_metadata['passage_ids'][idx]
            
            # Check: Same cluster AND not the positive passage itself
            if cluster_id == positive_cluster_id and passage_id != positive_passage_id:
                same_cluster_passage_ids.append(passage_id)
        
        # Randomly select 3 passages from same cluster as hard negatives
        num_negatives = min(3, len(same_cluster_passage_ids))
        selected_hard_negatives = random.sample(same_cluster_passage_ids, num_negatives)
        
        hard_negatives_batch.append(selected_hard_negatives)
    
    return hard_negatives_batch

class AdaptiveRefreshManager:
    """Implements our Adaptive Cluster Refreshing innovation"""
    
    def __init__(self, config):
        self.config = config
        self.loss_history = []
        self.last_refresh_epoch = 0
        
    def should_refresh_clusters(self, current_epoch: int, current_loss: float, force_refresh: bool = False) -> bool:
        """Determine if clusters should be refreshed based on learning progress"""
        
        # Always refresh on first epoch
        if current_epoch == 1:
            self.loss_history.append(current_loss)
            self.last_refresh_epoch = current_epoch
            return True
        
        self.loss_history.append(current_loss)
        
        # Force refresh every max_refresh_interval epochs (conservative fallback)
        if force_refresh and (current_epoch - self.last_refresh_epoch) >= self.config.max_refresh_interval:
            self.last_refresh_epoch = current_epoch
            return True
        
        # Minimum refresh interval
        if (current_epoch - self.last_refresh_epoch) < self.config.min_refresh_interval:
            return False
        
        # Calculate learning progress (Our Innovation)
        if len(self.loss_history) >= 4:
            recent_losses = self.loss_history[-4:]
            improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
            
            # Refresh if significant improvement
            if improvement > self.config.loss_improvement_threshold:
                self.last_refresh_epoch = current_epoch
                return True
        

        return False
