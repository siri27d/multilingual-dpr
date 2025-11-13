#!/usr/bin/env python3
"""
Main execution script for Multilingual DPR with Adaptive Clustering
"""

import os
import json
import torch
from transformers import AutoTokenizer

from config.training_config import TrainingConfig
from data.data_loader import DataProcessor, create_data_loaders
from data.data_generation import SyntheticDataGenerator
from models.dpr_model import MultilingualDPR
from training.trainer import ComprehensiveTrainer
from evaluation.evaluator import RetrievalEvaluator
from visualization.visualizer import TrainingVisualizer

def setup_environment():
    """Setup training environment"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("🚀 Setting up Multilingual DPR Training Environment")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

def main():
    """Main training pipeline"""
    # Setup
    setup_environment()
    config = TrainingConfig()
    
    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Save config
    with open(f'{config.output_dir}/config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print("📊 Configuration:")
    for key, value in config.to_dict().items():
        print(f"   {key}: {value}")
    
    # Load or generate data
    print("\n📚 Loading data...")
    try:
        # Try to load real dataset first
        hf_datasets = DataProcessor.load_mr_tydi()
        queries, passages = DataProcessor.process_hf_datasets(hf_datasets)
        print(f"✅ Loaded {len(queries)} queries and {len(passages)} passages from Mr. TyDi")
    except Exception as e:
        print(f"❌ Failed to load Mr. TyDi: {e}")
        print("🔄 Generating synthetic data...")
        queries, passages = SyntheticDataGenerator.create_synthetic_data()
        print(f"✅ Generated {len(queries)} synthetic queries and {len(passages)} passages")
    
    # Initialize model and tokenizer
    print("\n🤖 Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = MultilingualDPR(config)
    
    # Create data loaders
    print("\n📦 Creating data loaders...")
    train_loader = create_data_loaders(
        queries[:int(0.8 * len(queries))],  # 80% for training
        passages, 
        tokenizer, 
        batch_size=config.batch_size
    )
    
    eval_queries = queries[int(0.8 * len(queries)):]  # 20% for evaluation
    eval_loader = create_data_loaders(
        eval_queries,
        passages,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Evaluation batches: {len(eval_loader)}")
    
    # Initialize and run training
    print("\n🎯 Starting training with our innovations...")
    trainer = ComprehensiveTrainer(config, model, train_loader, eval_loader)
    trainer.train(passages)
    
    # Evaluation
    print("\n📊 Running final evaluation...")
    evaluator = RetrievalEvaluator(config, model)
    results = evaluator.evaluate(eval_queries, passages)
    
    print("\n🎯 Final Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    # Save results
    with open(f'{config.output_dir}/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    visualizer = TrainingVisualizer(config)
    
    # Training curves
    visualizer.create_training_curves(
        trainer.training_history,
        save_path=f'{config.output_dir}/training_curves.png'
    )
    
    # Cluster evolution
    if trainer.training_history.get('cluster_operations'):
        visualizer.create_cluster_visualization(
            trainer.training_history['cluster_operations'],
            save_path=f'{config.output_dir}/cluster_evolution.png'
        )
    
    print(f"\n🎉 All done! Check outputs in: {config.output_dir}")

if __name__ == "__main__":
    main()