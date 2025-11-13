#!/usr/bin/env python3
"""
Training script for Multilingual Dense Passage Retrieval
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config.training_config import TrainingConfig
from models.dpr_model import MultilingualDPR
from data.data_loader import DataProcessor, create_data_loaders
from training.trainer import ComprehensiveTrainer
from utils.helpers import set_seed, save_config

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(config.output_dir, 'training_config.json'))
    
    print("🔧 Initializing Multilingual DPR with Innovations...")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"⚙️  Configuration: {config.__dict__}")
    
    # Load or generate data
    print("\n📊 Loading data...")
    processor = DataProcessor()
    
    # Try to load real data, fall back to synthetic
    try:
        datasets = processor.load_mr_tydi(['english', 'finnish', 'japanese'], split='train[:1000]')
        # Process real data into queries and passages format
        queries, passages = [], []
        for lang, dataset in datasets.items():
            for example in dataset:
                queries.append({
                    'id': example['query_id'],
                    'text': example['query'],
                    'language': lang,
                    'positive_passage_ids': [example['positive_passages'][0]['docid']]
                })
                passages.append({
                    'id': example['positive_passages'][0]['docid'],
                    'text': example['positive_passages'][0]['text'],
                    'language': lang
                })
        print(f"✅ Loaded real data: {len(queries)} queries, {len(passages)} passages")
    except Exception as e:
        print(f"❌ Could not load real data: {e}")
        print("🔄 Generating synthetic data...")
        queries, passages = processor.create_synthetic_data(num_queries=500, num_passages=2000)
        print(f"✅ Generated synthetic data: {len(queries)} queries, {len(passages)} passages")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = MultilingualDPR(config)
    
    # Create data loaders
    train_loader = create_data_loaders(queries, passages, model.tokenizer, config.batch_size)
    
    print(f"📈 Training data: {len(queries)} queries, {len(passages)} passages")
    print(f"🤖 Model: {config.model_name}")
    print(f"🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = ComprehensiveTrainer(config, model, train_loader, use_wandb=False)
    
    # Start training
    trainer.train(passages)
    
    print("🎉 Training completed successfully!")
    print(f"📊 Results saved to: {config.output_dir}")

if __name__ == "__main__":
    main()