#!/usr/bin/env python3
"""
Evaluation script for Multilingual Dense Passage Retrieval
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from config.training_config import TrainingConfig
from models.dpr_model import MultilingualDPR
from data.data_loader import DataProcessor
from evaluation.evaluator import RetrievalEvaluator

def main():
    # Load configuration
    config = TrainingConfig()
    
    # Load model
    model = MultilingualDPR(config)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.output_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    else:
        print("❌ No checkpoint found")
        return
    
    model.eval()
    
    # Load or generate test data
    print("📊 Loading test data...")
    processor = DataProcessor()
    
    try:
        # Try to load real test data
        test_datasets = processor.load_mr_tydi(['english'], split='validation[:100]')
        test_queries, test_passages = [], []
        for lang, dataset in test_datasets.items():
            for example in dataset:
                test_queries.append({
                    'id': example['query_id'],
                    'text': example['query'],
                    'positive_passage_ids': [example['positive_passages'][0]['docid']]
                })
                test_passages.append({
                    'id': example['positive_passages'][0]['docid'],
                    'text': example['positive_passages'][0]['text']
                })
    except:
        # Fall back to synthetic data
        print("🔄 Using synthetic test data")
        test_queries, test_passages = processor.create_synthetic_data(num_queries=50, num_passages=200)
    
    print(f"🧪 Evaluating on {len(test_queries)} queries and {len(test_passages)} passages...")
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(config, model)
    
    # Run evaluation
    results = evaluator.evaluate(test_queries, test_passages)
    
    print("\n📊 EVALUATION RESULTS:")
    print(f"Recall@1:  {results['recall@1']:.4f}")
    print(f"Recall@5:  {results['recall@5']:.4f}")
    print(f"Recall@10: {results['recall@10']:.4f}")
    print(f"MRR:       {results['mrr']:.4f}")
    
    # Save results
    results_path = os.path.join(config.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")

if __name__ == "__main__":
    main()