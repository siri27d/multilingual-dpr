import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os

class TrainingVisualizer:
    """Creates visualizations for training analysis"""
    
    def __init__(self, config):
        self.config = config
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_training_curves(self, training_history: Dict, save_path: str = None):
        """Create training loss and metric curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        losses = [x['loss'] for x in training_history['losses']]
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rates' in training_history:
            lrs = training_history['learning_rates']
            ax2.plot(epochs[:len(lrs)], lrs, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # Cluster operations
        if 'cluster_operations' in training_history:
            cluster_ops = training_history['cluster_operations']
            cluster_epochs = [op['epoch'] for op in cluster_ops]
            cluster_sizes = [op['n_clusters'] for op in cluster_ops]
            ax3.plot(cluster_epochs, cluster_sizes, 'ro-', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Number of Clusters')
            ax3.set_title('Adaptive Cluster Sizing (SAC-ICT)')
            ax3.grid(True, alpha=0.3)
            
            # Silhouette scores
            silhouette_scores = [op['silhouette_score'] for op in cluster_ops]
            ax4.plot(cluster_epochs, silhouette_scores, 'purple', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Silhouette Score')
            ax4.set_title('Clustering Quality Over Time')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Training curves saved to {save_path}")
        
        plt.show()
    
    def create_cluster_visualization(self, cluster_history: List[Dict], save_path: str = None):
        """Visualize cluster evolution over training"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = [entry['epoch'] for entry in cluster_history]
        n_clusters = [entry['n_clusters'] for entry in cluster_history]
        silhouette_scores = [entry['silhouette_score'] for entry in cluster_history]
        
        # Cluster growth
        ax1.plot(epochs, n_clusters, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Dynamic Cluster Granularity (SAC-ICT)')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(epochs, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Training Epoch')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Clustering Quality Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_evaluation_comparison(self, baseline_results: Dict, our_method_results: Dict, save_path: str = None):
        """Create comparison bar chart between baseline and our method"""
        metrics = ['recall@1', 'recall@5', 'recall@10', 'mrr']
        baseline_scores = [baseline_results[metric] for metric in metrics]
        our_scores = [our_method_results[metric] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline ICT', alpha=0.7)
        bars2 = ax.bar(x + width/2, our_scores, width, label='Our Method (SAC-ICT + ACR-ICT)', alpha=0.7)
        
        ax.set_xlabel('Evaluation Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison: Baseline vs Our Method')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison chart saved to {save_path}")
        
        plt.show()

    def create_multilingual_performance(self, results_by_language: Dict, save_path: str = None):
        """Create visualization of performance across different languages"""
        languages = list(results_by_language.keys())
        recall_scores = [results_by_language[lang]['recall@5'] for lang in languages]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(languages, recall_scores, color=sns.color_palette("husl", len(languages)))
        
        ax.set_xlabel('Language')
        ax.set_ylabel('Recall@5')
        ax.set_title('Multilingual Retrieval Performance Across Languages')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()