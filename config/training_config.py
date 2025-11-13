import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TrainingConfig:
    # Model parameters
    model_name: str = "xlm-roberta-base"
    projection_dim: int = 768
    temperature: float = 0.05
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs_pretrain: int = 3
    num_epochs_finetune: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Clustering parameters (Our Innovations)
    initial_clusters: int = 500
    max_clusters: int = 2000
    cluster_growth_rate: int = 100
    min_refresh_interval: int = 3
    max_refresh_interval: int = 8
    loss_improvement_threshold: float = 0.05
    cluster_batch_size: int = 10000
    
    # Hard negative mining
    hard_negatives_per_query: int = 5
    in_batch_negatives: bool = True
    cross_lingual_sampling: bool = True
    
    # Evaluation
    eval_steps: int = 500
    top_k_retrieval: List[int] = (1, 5, 10, 100)
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    
    # Experimental features
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and k != 'device'
        }