import json
import random
from typing import List, Dict

class SyntheticDataGenerator:
    """Generate synthetic multilingual data for testing"""
    
    @staticmethod
    def create_synthetic_data(num_queries=500, num_passages=2000):
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

    @staticmethod
    def save_sample_data():
        """Create and save sample data for GitHub"""
        queries, passages = SyntheticDataGenerator.create_synthetic_data(50, 200)
        
        sample_data = {
            'queries': queries[:10],
            'passages': passages[:50]
        }
        
        with open('data/sample_synthetic/sample_data.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print("✅ Sample data saved to data/sample_synthetic/sample_data.json")

if __name__ == "__main__":
    SyntheticDataGenerator.save_sample_data()