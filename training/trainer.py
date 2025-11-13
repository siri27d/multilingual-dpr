import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import time
from tqdm import tqdm
import json

from training.cluster_manager import SemanticClusterManager, AdaptiveRefreshManager
from training.loss_functions import ContrastiveLoss

class ComprehensiveTrainer:
    """Main trainer class implementing our complete training pipeline"""
    
    def __init__(self, config, model, train_loader, eval_loader=None, use_wandb=False):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Initialize components
        self.cluster_manager = SemanticClusterManager(config, model, model.tokenizer)
        self.refresh_manager = AdaptiveRefreshManager(config)
        self.loss_function = ContrastiveLoss(config.temperature)
        
        # Training state
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs_finetune)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.global_step = 0
        self.best_metric = 0.0
        
        self.training_history = {
            'losses': [],
            'learning_rates': [],
            'cluster_operations': []
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def train(self, passages):
        """Complete training pipeline with our innovations"""
        print("🚀 Starting comprehensive training with real data...")
        
        current_cluster_index = None
        current_cluster_info = None
        
        for epoch in range(1, self.config.num_epochs_finetune + 1):
            print(f"\n⭐ Epoch {epoch}/{self.config.num_epochs_finetune}")
            
            # Adaptive Cluster Refresh (Our Innovation)
            current_loss = self.training_history['losses'][-1]['loss'] if self.training_history['losses'] else 1.0
            should_refresh = self.refresh_manager.should_refresh_clusters(
                current_epoch=epoch,
                current_loss=current_loss,
                force_refresh=(epoch % self.config.max_refresh_interval == 0)
            )
            
            if should_refresh or current_cluster_index is None:
                current_cluster_index, current_cluster_info = self.cluster_manager.perform_semantic_clustering(
                    passages, epoch
                )
                self.training_history['cluster_operations'].append({
                    'epoch': epoch,
                    'n_clusters': current_cluster_info['n_clusters'],
                    'silhouette_score': current_cluster_info['silhouette_score']
                })
            
            # Training epoch
            epoch_loss = self._train_epoch(epoch)
            self.training_history['losses'].append({
                'epoch': epoch, 
                'stage': 'finetune', 
                'loss': epoch_loss
            })
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Update scheduler
            self.scheduler.step()
            
            print(f'✅ Epoch {epoch} completed. Average Loss: {epoch_loss:.4f}')
        
        # Final save
        self._save_final_model()
        self._save_training_history()
        
        print("🎉 Training completed!")
    
    def _train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        
        for batch in progress_bar:
            # Move to device
            query_inputs = {k: v.to(self.config.device) for k, v in batch['query_inputs'].items()}
            passage_inputs = {k: v.to(self.config.device) for k, v in batch['passage_inputs'].items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                query_embeddings, passage_embeddings = self.model(batch)
                
                # Compute contrastive loss
                similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.T) / self.config.temperature
                labels = torch.arange(len(query_embeddings)).to(self.config.device)
                loss = nn.functional.cross_entropy(similarity_matrix, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def _save_final_model(self):
        """Save the final model and training history"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history,
        }, f'{self.config.output_dir}/final_model.pth')
        
        print(f"💾 Model saved to {self.config.output_dir}")
    
    def _save_training_history(self):
        """Save training history to JSON"""
        with open(f'{self.config.output_dir}/training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)