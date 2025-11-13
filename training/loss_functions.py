import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """Contrastive loss with in-batch negatives"""
    
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query_embeddings, passage_embeddings, hard_negative_embeddings=None):
        """
        Compute contrastive loss
        
        Args:
            query_embeddings: [batch_size, dim]
            passage_embeddings: [batch_size, dim] 
            hard_negative_embeddings: [batch_size, num_negatives, dim] (optional)
        """
        batch_size = query_embeddings.size(0)
        
        # Compute similarity matrix (batch_size x batch_size)
        similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.T) / self.temperature
        
        # Labels are the diagonal elements (positive pairs)
        labels = torch.arange(batch_size).to(query_embeddings.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class HardNegativeLoss(nn.Module):
    """Contrastive loss with hard negatives"""
    
    def __init__(self, temperature=0.05, hard_negative_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
    
    def forward(self, query_embeddings, positive_embeddings, hard_negative_embeddings):
        """
        Compute loss with hard negatives
        """
        batch_size = query_embeddings.size(0)
        
        # Positive similarities
        positive_similarities = torch.sum(query_embeddings * positive_embeddings, dim=1) / self.temperature
        
        # Hard negative similarities
        hard_negative_similarities = torch.bmm(
            query_embeddings.unsqueeze(1),
            hard_negative_embeddings.transpose(1, 2)
        ).squeeze(1) / self.temperature
        
        # Combine all similarities
        all_similarities = torch.cat([
            positive_similarities.unsqueeze(1),
            hard_negative_similarities
        ], dim=1)
        
        # Labels are 0 (positive is at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long).to(query_embeddings.device)
        
        loss = F.cross_entropy(all_similarities, labels)
        
        return loss