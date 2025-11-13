import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import time
from tqdm import tqdm
import json
import wandb

from training.cluster_manager import SemanticClusterManager, AdaptiveRefreshManager
from training.loss_functions import ContrastiveLoss
from visualization.visualizer import TrainingVisualizer

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
        self.visualizer = TrainingVisualizer(config)
        
        # Training state
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs_finetune)
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        self.global_step = 0
        self.best_metric = 0.0
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="multilingual-dpr", config=config.to_dict())
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
    def train(self, passages):
        """Complete training pipeline with our innovations"""
        print("🚀 Starting multilingual DPR training with innovations...")
        
        training_history = {
            'losses': [],
            'learning_rates': [],
            'clustering_operations': [],
            'evaluation_metrics': []
        }
        
        # Stage 1: Pre-training (optional)
        if self.config.num_epochs_pretrain > 0:
            print("\n=== Stage 1: Pre-training ===")
            self._pretrain_stage(passages, training_history)
        
        # Stage 2: Fine-tuning with our innovations
        print("\n=== Stage 2: Fine-tuning with SAC-ICT & ACR-ICT ===")
        self._finetune_stage(passages, training_history)
        
        # Save final model and training history
        self._save_checkpoint(self.config.num_epochs_finetune, is_best=False)
        self._save_training_history(training_history)
        
        if self.use_wandb:
            wandb.finish()
        
        print("✅ Training completed!")
        
    def _pretrain_stage(self, passages, training_history):
        """Pre-training stage without hard negatives"""
        self.model.train()
        
        for epoch in range(1, self.config.num_epochs_pretrain + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f'Pre-train Epoch {epoch}')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with optional mixed precision
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        query_embeddings, passage_embeddings = self.model(batch)
                        loss = self.loss_function(query_embeddings, passage_embeddings, None)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    query_embeddings, passage_embeddings = self.model(batch)
                    loss = self.loss_function(query_embeddings, passage_embeddings, None)
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                self.global_step += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                if self.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'pretrain_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'step': self.global_step
                    })
            
            avg_loss = epoch_loss / len(self.train_loader)
            training_history['losses'].append({
                'epoch': epoch, 'stage': 'pretrain', 'loss': avg_loss
            })
            
            print(f'Pre-train Epoch {epoch}, Average Loss: {avg_loss:.4f}')
    
    def _finetune_stage(self, passages, training_history):
        """Fine-tuning stage with our SAC-ICT and ACR-ICT innovations"""
        current_cluster_index = None
        current_cluster_metadata = None
        
        for epoch in range(1, self.config.num_epochs_finetune + 1):
            # Adaptive Cluster Refresh (Our Innovation)
            should_refresh = self.refresh_manager.should_refresh_clusters(
                current_epoch=epoch,
                current_loss=training_history['losses'][-1]['loss'] if training_history['losses'] else 1.0,
                force_refresh=(epoch % self.config.max_refresh_interval == 0)
            )
            
            if should_refresh or current_cluster_index is None:
                print(f"🔄 Refreshing clusters for epoch {epoch}...")
                current_cluster_index, current_cluster_metadata = self.cluster_manager.perform_semantic_clustering(
                    passages, epoch, 
                    training_history['losses'][-1]['loss'] if training_history['losses'] else None
                )
                current_cluster_metadata['index'] = current_cluster_index
                training_history['clustering_operations'].append({
                    'epoch': epoch,
                    'n_clusters': current_cluster_metadata['n_clusters'],
                    'silhouette_score': self.cluster_manager.cluster_history[-1]['silhouette_score']
                })
            
            # Training epoch with hard negatives
            epoch_loss = self._train_epoch_with_hard_negatives(
                epoch, current_cluster_metadata, training_history
            )
            
            # Update scheduler
            self.scheduler.step()
            
            # Evaluation
            if self.eval_loader and epoch % 2 == 0:
                eval_metrics = self.evaluate()
                training_history['evaluation_metrics'].append({
                    'epoch': epoch,
                    'metrics': eval_metrics
                })
                
                # Save best model
                if eval_metrics.get('mrr', 0) > self.best_metric:
                    self.best_metric = eval_metrics['mrr']
                    self._save_checkpoint(epoch, is_best=True)
            
            # Visualization
            if epoch % 3 == 0:
                self.visualizer.create_training_visualizations(
                    training_history, self.cluster_manager.cluster_history, epoch
                )
            
            # Save checkpoint
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, is_best=False)
    
    def _train_epoch_with_hard_negatives(self, epoch, cluster_metadata, training_history):
        """Train one epoch with hard negative mining"""
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Fine-tune Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Mine hard negatives using our semantic clustering
            hard_negative_passage_ids = []
            for i, query_id in enumerate(batch['query_ids']):
                # In a real implementation, you'd use the cluster metadata to find hard negatives
                # This is simplified for the example
                hard_negatives = self.cluster_manager.find_hard_negatives(
                    batch, i, cluster_metadata, self.config.hard_negatives_per_query
                )
                hard_negative_passage_ids.append(hard_negatives)
            
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    query_embeddings, positive_embeddings = self.model(batch)
                    # Note: Hard negative encoding would be implemented here
                    loss = self.loss_function(query_embeddings, positive_embeddings, None)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                query_embeddings, positive_embeddings = self.model(batch)
                loss = self.loss_function(query_embeddings, positive_embeddings, None)
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'finetune_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step': self.global_step,
                    'epoch': epoch
                })
        
        avg_loss = epoch_loss / len(self.train_loader)
        training_history['losses'].append({
            'epoch': epoch, 'stage': 'finetune', 'loss': avg_loss
        })
        
        print(f'Fine-tune Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        return avg_loss
    
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        # Implementation would go here
        return {'mrr': 0.0, 'recall@1': 0.0, 'recall@5': 0.0, 'recall@10': 0.0}
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config.to_dict()
        }
        
        if is_best:
            filename = "best_model.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, os.path.join(self.config.output_dir, filename))
        
    def _save_training_history(self, training_history):
        """Save training history to JSON"""
        with open(os.path.join(self.config.output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)