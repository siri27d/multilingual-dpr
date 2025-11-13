import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random
from datasets import load_dataset

class MultilingualRetrievalDataset(Dataset):
    def __init__(self, queries: List[Dict], passages: List[Dict], tokenizer, max_length=256):
        self.queries = queries
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create passage ID to index mapping
        self.passage_id_to_idx = {p['id']: idx for idx, p in enumerate(passages)}
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        positive_passage = self.passages[self.passage_id_to_idx[query['positive_passage_ids'][0]]]
        
        return {
            'query_id': query['id'],
            'query_text': query['text'],
            'query_language': query['language'],
            'positive_passage_id': positive_passage['id'],
            'positive_passage_text': positive_passage['text'],
            'positive_passage_language': positive_passage['language']
        }
    
    def collate_fn(self, batch):
        queries = [item['query_text'] for item in batch]
        positive_passages = [item['positive_passage_text'] for item in batch]
        
        # Tokenize queries and passages
        query_encodings = self.tokenizer(
            queries, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        passage_encodings = self.tokenizer(
            positive_passages, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'query_ids': [item['query_id'] for item in batch],
            'query_input_ids': query_encodings['input_ids'],
            'query_attention_mask': query_encodings['attention_mask'],
            'positive_passage_ids': [item['positive_passage_id'] for item in batch],
            'positive_passage_input_ids': passage_encodings['input_ids'],
            'positive_passage_attention_mask': passage_encodings['attention_mask'],
            'languages': [item['query_language'] for item in batch]
        }

class DataProcessor:
    """Handles data loading and processing for multilingual datasets"""
    
    @staticmethod
    def load_mr_tydi(languages: List[str] = None, split: str = 'train'):
        """Load Mr. TyDi dataset"""
        if languages is None:
            languages = ['arabic', 'bengali', 'english', 'finnish', 'indonesian']
        
        datasets = {}
        for lang in languages:
            try:
                dataset = load_dataset("castorini/mr-tydi", lang, split=split)
                datasets[lang] = dataset
                print(f"✅ Loaded {lang} with {len(dataset)} examples")
            except Exception as e:
                print(f"❌ Failed to load {lang}: {e}")
        
        return datasets
    
    @staticmethod
    def process_hf_datasets(datasets):
        """Process HuggingFace datasets into our format"""
        queries = []
        passages = []
        
        for lang, dataset in datasets.items():
            for example in dataset:
                query_id = example.get('query_id', f"query_{len(queries)}")
                queries.append({
                    'id': query_id,
                    'text': example['query'],
                    'language': lang,
                    'positive_passage_ids': [example['positive_passages'][0]['docid']]
                })
                
                # Add positive passage
                positive_passage = example['positive_passages'][0]
                passages.append({
                    'id': positive_passage['docid'],
                    'text': positive_passage['text'],
                    'language': lang
                })
        
        return queries, passages
    
    @staticmethod
    def create_synthetic_data(num_queries=500, num_passages=2000):
        """Create synthetic multilingual data for testing"""
        languages = ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh', 'ar']
        topics = [
            'artificial intelligence', 'climate change', 'healthcare', 
            'renewable energy', 'space exploration', 'quantum computing',
            'biodiversity', 'financial technology'
        ]
        
        # Generate passages
        passages = []
        for i in range(num_passages):
            lang = random.choice(languages)
            topic = random.choice(topics)
            passages.append({
                'id': f'passage_{i}',
                'text': f"This is a detailed passage in {lang} about {topic}. " * 3,
                'language': lang,
                'topic': topic
            })
        
        # Generate queries with positive passages
        queries = []
        for i in range(num_queries):
            lang = random.choice(languages)
            topic = random.choice(topics)
            # Find relevant passages for this query
            relevant_passages = [p for p in passages if p['topic'] == topic and p['language'] == lang]
            positive_passage = random.choice(relevant_passages) if relevant_passages else random.choice(passages)
            
            queries.append({
                'id': f'query_{i}',
                'text': f"{lang} query about {topic}",
                'language': lang,
                'positive_passage_ids': [positive_passage['id']],
                'topic': topic
            })
        
        return queries, passages

def create_data_loaders(queries, passages, tokenizer, batch_size=32, shuffle=True):
    """Create data loaders for training and evaluation"""
    dataset = MultilingualRetrievalDataset(queries, passages, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=dataset.collate_fn
    )
    return dataloader