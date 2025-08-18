# src/models/bias_classifier.py

import torch
import torch.nn as nn
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    DistilBertConfig,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import pickle
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

class BiasClassifier:
    """
    DistilBERT-based political bias classifier
    Classifies news articles as left-leaning, centrist, or right-leaning
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 num_labels: int = 3, load_pretrained: bool = False):
        """
        Initialize the bias classifier
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of bias categories (3: left, center, right)
            load_pretrained: Whether to load a pre-trained model
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        if load_pretrained:
            self.model = self._load_model()
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        
        self.model.to(self.device)
        
        # Label mapping
        self.label_map = settings.BIAS_LABELS
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def _load_model(self) -> DistilBertForSequenceClassification:
        """Load a pre-trained model from disk"""
        model_path = settings.MODEL_DIR / "bias_classifier"
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            return DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            logger.warning("No pre-trained model found, using base model")
            return DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
    
    def save_model(self, path: Optional[Path] = None):
        """Save the trained model"""
        if path is None:
            path = settings.MODEL_DIR / "bias_classifier"
        
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def preprocess_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode texts for model input
        
        Args:
            texts: List of article texts
            
        Returns:
            Tokenized inputs as tensors
        """
        # Clean and truncate texts
        cleaned_texts = []
        for text in texts:
            # Remove extra whitespace and limit length
            cleaned = ' '.join(text.split())
            if len(cleaned) > 3000:  # Rough character limit
                cleaned = cleaned[:3000]
            cleaned_texts.append(cleaned)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=settings.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict political bias for articles
        
        Args:
            texts: List of article texts
            
        Returns:
            List of prediction dictionaries with probabilities
        """
        self.model.eval()
        
        # Process in batches to avoid memory issues
        batch_size = settings.BATCH_SIZE
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.preprocess_text(batch_texts)
                
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Convert to predictions
                for probs in probabilities:
                    pred_dict = {}
                    for label_idx, prob in enumerate(probs.cpu().numpy()):
                        label_name = self.label_map[label_idx]
                        pred_dict[label_name] = float(prob)
                    
                    # Add predicted class
                    predicted_class = torch.argmax(probs).item()
                    pred_dict['predicted_class'] = self.label_map[predicted_class]
                    pred_dict['confidence'] = float(torch.max(probs))
                    
                    all_predictions.append(pred_dict)
        
        return all_predictions
    
    def predict_single(self, text: str) -> Dict[str, float]:
        """Predict bias for a single article"""
        predictions = self.predict([text])
        return predictions[0]
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None,
              epochs: int = 3, learning_rate: float = 2e-5) -> Dict:
        """
        Train the bias classifier
        
        Args:
            train_texts: Training article texts
            train_labels: Training labels (0, 1, 2)
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Training bias classifier on {len(train_texts)} articles")
        
        # Prepare datasets
        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = self._create_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(settings.MODEL_DIR / "bias_classifier_training"),
            num_train_epochs=epochs,
            per_device_train_batch_size=settings.BATCH_SIZE,
            per_device_eval_batch_size=settings.BATCH_SIZE,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(settings.MODEL_DIR / "logs"),
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_accuracy" if val_dataset else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        training_result = trainer.train()
        
        # Save model
        self.save_model()
        
        return {
            'train_loss': training_result.training_loss,
            'train_steps': training_result.global_step,
            'epochs': epochs
        }
    
    def _create_dataset(self, texts: List[str], labels: List[int]):
        """Create dataset for training"""
        encodings = self.preprocess_text(texts)
        
        class BiasDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return BiasDataset(encodings, labels)
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
        }
    
    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """
        Evaluate the model on test data
        
        Returns:
            Evaluation metrics including accuracy and classification report
        """
        predictions = self.predict(test_texts)
        pred_labels = [self.reverse_label_map[p['predicted_class']] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, pred_labels)
        report = classification_report(
            test_labels, 
            pred_labels,
            target_names=list(self.label_map.values()),
            output_dict=True
        )
        cm = confusion_matrix(test_labels, pred_labels)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'detailed_predictions': predictions
        }

# src/models/similarity_detector.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class SimilarityDetector:
    """
    Semantic similarity detector using sentence transformers
    Finds articles covering the same story across different sources
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize similarity detector
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.threshold = settings.SIMILARITY_THRESHOLD
        
        logger.info(f"Initialized similarity detector with {model_name}")
    
    def encode_articles(self, texts: List[str]) -> np.ndarray:
        """
        Encode articles into vector representations
        
        Args:
            texts: List of article texts (titles + content)
            
        Returns:
            Numpy array of embeddings
        """
        # Clean texts
        cleaned_texts = []
        for text in texts:
            # Combine title and content, limit length
            if len(text) > 2000:
                text = text[:2000]
            cleaned_texts.append(text.strip())
        
        # Generate embeddings with progress bar only for large batches
        show_progress = len(texts) > 50
        
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=32,
            show_progress_bar=show_progress,  # Only show for large batches
            convert_to_numpy=True
        )
        
        return embeddings
    
    def find_similar_articles(self, query_article: str, 
                            candidate_articles: List[str],
                            top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find articles similar to a query article
        
        Args:
            query_article: The article to find matches for
            candidate_articles: Pool of candidate articles
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        # Encode all articles
        all_texts = [query_article] + candidate_articles
        embeddings = self.encode_articles(all_texts)
        
        # Calculate similarities
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top-k similar articles above threshold
        similar_indices = []
        for idx, sim_score in enumerate(similarities):
            if sim_score >= self.threshold:
                similar_indices.append((idx, float(sim_score)))
        
        # Sort by similarity score and return top-k
        similar_indices.sort(key=lambda x: x[1], reverse=True)
        return similar_indices[:top_k]
    
    def cluster_similar_articles(self, articles: List[str], 
                               article_metadata: List[Dict] = None) -> List[List[int]]:
        """
        Cluster articles by semantic similarity
        
        Args:
            articles: List of article texts
            article_metadata: Optional metadata for each article
            
        Returns:
            List of clusters, each containing article indices
        """
        if len(articles) < 2:
            return [[i] for i in range(len(articles))]
        
        # Encode articles
        embeddings = self.encode_articles(articles)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Simple clustering based on similarity threshold
        clusters = []
        assigned = set()
        
        for i, article in enumerate(articles):
            if i in assigned:
                continue
                
            # Start new cluster
            cluster = [i]
            assigned.add(i)
            
            # Find similar articles
            for j in range(i + 1, len(articles)):
                if j in assigned:
                    continue
                    
                if similarities[i][j] >= self.threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def find_cross_perspective_matches(self, articles_by_bias: Dict[str, List]) -> Dict:
        """
        Find articles covering the same story across different political perspectives
        
        Args:
            articles_by_bias: Dict with bias categories as keys, article lists as values
            
        Returns:
            Dict of matched article groups with different perspectives
        """
        matches = []
        
        # Get all bias categories
        bias_categories = list(articles_by_bias.keys())
        
        if len(bias_categories) < 2:
            return {'matches': matches}
        
        # Compare articles across bias categories
        for i, bias1 in enumerate(bias_categories):
            articles1 = articles_by_bias[bias1]
            
            for j, bias2 in enumerate(bias_categories[i+1:], i+1):
                articles2 = articles_by_bias[bias2]
                
                # Find similarities between the two groups
                for idx1, article1 in enumerate(articles1):
                    article1_text = f"{article1.get('title', '')} {article1.get('content', '')}"
                    
                    candidate_texts = []
                    for article2 in articles2:
                        candidate_texts.append(f"{article2.get('title', '')} {article2.get('content', '')}")
                    
                    # Find similar articles
                    similar = self.find_similar_articles(
                        article1_text, 
                        candidate_texts, 
                        top_k=3
                    )
                    
                    # Record matches
                    for idx2, similarity in similar:
                        match = {
                            'articles': {
                                bias1: {
                                    'index': idx1,
                                    'article': articles1[idx1],
                                    'bias': bias1
                                },
                                bias2: {
                                    'index': idx2, 
                                    'article': articles2[idx2],
                                    'bias': bias2
                                }
                            },
                            'similarity_score': similarity,
                            'topic_detected': True
                        }
                        matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'matches': matches,
            'total_matches': len(matches),
            'average_similarity': np.mean([m['similarity_score'] for m in matches]) if matches else 0
        }

# Example usage and testing
if __name__ == "__main__":
    # Test bias classifier
    print("Testing Bias Classifier...")
    bias_classifier = BiasClassifier()
    
    # Test articles
    test_articles = [
        "The progressive policy aims to expand healthcare access for all Americans.",
        "This fiscally responsible approach will reduce government spending.",
        "The bipartisan bill received support from both parties in Congress."
    ]
    
    predictions = bias_classifier.predict(test_articles)
    for i, pred in enumerate(predictions):
        print(f"Article {i+1}: {pred['predicted_class']} (confidence: {pred['confidence']:.3f})")
    
    print("\nTesting Similarity Detector...")
    similarity_detector = SimilarityDetector()
    
    # Test similarity
    query = "President announces new economic policy"
    candidates = [
        "White House reveals economic stimulus plan",
        "Sports team wins championship game", 
        "New economic measures announced by administration"
    ]
    
    similar = similarity_detector.find_similar_articles(query, candidates)
    print(f"Similar articles: {similar}")