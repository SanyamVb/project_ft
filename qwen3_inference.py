"""
Inference script for Qwen3 sequence classification off-topic detection model.
Uses AutoModelForSequenceClassification which automatically loads Qwen3ForSequenceClassification.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class Qwen3OffTopicClassifier:
    """
    Wrapper for Qwen3 off-topic classification model with optimized threshold.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Load the trained Qwen3 classification model.
        
        Args:
            model_path: Path to saved model directory
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        
        # Load threshold configuration
        threshold_config_path = f"{model_path}/threshold_config.json"
        try:
            with open(threshold_config_path, "r") as f:
                config = json.load(f)
                self.threshold = config.get("best_threshold", 0.5)
                print(f"Loaded optimized threshold: {self.threshold:.3f}")
                print(f"Model: {config.get('model_name', 'Unknown')}")
                print(f"Kappa at best threshold: {config.get('kappa_at_best', 'N/A')}")
        except FileNotFoundError:
            self.threshold = 0.5
            print(f"No threshold config found, using default: {self.threshold}")
    
    def predict(
        self, 
        script: str, 
        category: str, 
        transcription: str,
        return_prob: bool = False
    ):
        """
        Predict if a transcription is off-topic.
        
        Args:
            script: The prompt script
            category: The topic category
            transcription: The test-taker's transcription
            return_prob: If True, return probability instead of binary label
            
        Returns:
            If return_prob=False: 0 (on-topic) or 1 (off-topic)
            If return_prob=True: Probability of being off-topic (0.0 to 1.0)
        """
        # Build input as in training
        text_a = f"{script}\nTopic: {category}"
        text_b = transcription
        
        # Tokenize
        inputs = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            off_topic_prob = probs[0][1].item()
        
        if return_prob:
            return off_topic_prob
        else:
            return int(off_topic_prob >= self.threshold)
    
    def predict_batch(
        self,
        scripts: list[str],
        categories: list[str],
        transcriptions: list[str],
        batch_size: int = 16,
        return_probs: bool = False
    ):
        """
        Predict for multiple examples in batches.
        
        Args:
            scripts: List of prompt scripts
            categories: List of topic categories
            transcriptions: List of test-taker transcriptions
            batch_size: Batch size for processing
            return_probs: If True, return probabilities instead of binary labels
            
        Returns:
            List of predictions (0/1 if return_probs=False, floats if True)
        """
        all_predictions = []
        
        for i in range(0, len(scripts), batch_size):
            batch_scripts = scripts[i:i + batch_size]
            batch_categories = categories[i:i + batch_size]
            batch_transcriptions = transcriptions[i:i + batch_size]
            
            # Build inputs
            text_a_list = [
                f"{script}\nTopic: {category}"
                for script, category in zip(batch_scripts, batch_categories)
            ]
            
            # Tokenize batch
            inputs = self.tokenizer(
                text_a_list,
                batch_transcriptions,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                off_topic_probs = probs[:, 1].cpu().numpy()
            
            if return_probs:
                all_predictions.extend(off_topic_probs.tolist())
            else:
                predictions = (off_topic_probs >= self.threshold).astype(int)
                all_predictions.extend(predictions.tolist())
        
        return all_predictions


if __name__ == "__main__":
    # Example usage
    print("=== Qwen3 Off-Topic Classifier - Inference Example ===\n")
    
    # Initialize classifier
    model_path = "models/qwen3_cross_encoder_offtopic"
    classifier = Qwen3OffTopicClassifier(model_path)
    
    # Example 1: Single prediction
    print("\n--- Example 1: Single Prediction ---")
    script = "Describe your favorite vacation destination"
    category = "Travel"
    transcription = "I really enjoy traveling to beaches and mountains"
    
    prediction = classifier.predict(script, category, transcription)
    prob = classifier.predict(script, category, transcription, return_prob=True)
    
    print(f"Script: {script}")
    print(f"Category: {category}")
    print(f"Transcription: {transcription}")
    print(f"Prediction: {'Off-Topic' if prediction == 1 else 'On-Topic'}")
    print(f"Off-topic probability: {prob:.3f}")
    
    # Example 2: Off-topic example
    print("\n--- Example 2: Off-topic Example ---")
    script = "Describe your favorite vacation destination"
    category = "Travel"
    transcription = "I like to eat pizza and pasta for dinner"
    
    prediction = classifier.predict(script, category, transcription)
    prob = classifier.predict(script, category, transcription, return_prob=True)
    
    print(f"Script: {script}")
    print(f"Category: {category}")
    print(f"Transcription: {transcription}")
    print(f"Prediction: {'Off-Topic' if prediction == 1 else 'On-Topic'}")
    print(f"Off-topic probability: {prob:.3f}")
    
    # Example 3: Batch prediction
    print("\n--- Example 3: Batch Prediction ---")
    scripts = [
        "What is your opinion on climate change?",
        "Describe your daily routine",
        "Talk about a book you recently read",
    ]
    categories = ["Environment", "Lifestyle", "Literature"]
    transcriptions = [
        "Climate change is a serious issue that affects everyone",
        "I usually wake up at 7am and go for a jog",
        "My favorite color is blue and I like sunny days",  # Off-topic
    ]
    
    predictions = classifier.predict_batch(
        scripts, categories, transcriptions, return_probs=False
    )
    probs = classifier.predict_batch(
        scripts, categories, transcriptions, return_probs=True
    )
    
    for i, (s, c, t, pred, p) in enumerate(zip(scripts, categories, transcriptions, predictions, probs)):
        print(f"\n{i+1}. Category: {c}")
        print(f"   Script: {s[:50]}...")
        print(f"   Transcription: {t[:50]}...")
        print(f"   Prediction: {'Off-Topic' if pred == 1 else 'On-Topic'} (p={p:.3f})")
