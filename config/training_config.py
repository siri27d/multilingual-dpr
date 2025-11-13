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
    num_epochs_pretrain: int = 2
    num_epochs_finetune: int = 8
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Clustering parameters (Our Innovations)
    initial_clusters: int = 200
    max_clusters: int = 500
    cluster_growth_rate: int = 50
    min_refresh_interval: int = 2
    max_refresh_interval: int = 5
    loss_improvement_threshold: float = 0.05
    cluster_batch_size: int = 1000
    
    # Hard negative mining
    hard_negatives_per_query: int = 3
    in_batch_negatives: bool = True
    cross_lingual_sampling: bool = True
    
    # Evaluation
    eval_steps: int = 100
    top_k_retrieval: List[int] = (1, 5, 10)
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k != 'device'}