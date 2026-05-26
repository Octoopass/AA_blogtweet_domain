import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from utils.paths import ensure_parent_dir, project_path, results_path


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """
    Configuration for DistilBERT training
    """
    
    # Input data files - Choose one mode:
    # MODE 1: Single CSV with automatic split
    BLOG_CSV = str(project_path('training_data', 'selected_blogs.csv'))
    TWEET_CSV = str(project_path('training_data', 'selected_tweets.csv'))
    
    # MODE 2: Pre-existing train/test split
    USE_PRESPLIT_DATA = False  # Set to True to use pre-split files below
    TRAIN_CSV = r''  # Training data CSV
    TEST_CSV = r''    # Test data CSV
    
    # Column names in CSV
    TEXT_COLUMN = 'Text'
    AUTHOR_COLUMN = 'Author Name'
    
    # Output files
    MODEL_SAVE_DIR = str(results_path('distilbert', 'distilbert_authorship_model'))
    LABEL_ENCODER_FILE = str(results_path('distilbert', 'distilbert_label_encoder.pkl'))
    RESULTS_FILE = str(results_path('distilbert', 'distilbert_results.txt'))
    CONFUSION_MATRIX_IMG = str(results_path('distilbert', 'distilbert_confusion_matrix.png'))
    
    # Model parameters
    MODEL_NAME = 'distilbert-base-uncased'  # HuggingFace model name
    MAX_LENGTH = 256  # Maximum sequence length
    BATCH_SIZE = 8  # Batch size for training/evaluation
    LEARNING_RATE = 2e-5  # Learning rate
    NUM_EPOCHS = 5  # Number of training epochs
    WEIGHT_DECAY = 0.01  # Weight decay for regularization
    WARMUP_STEPS = 500  # Warmup steps for learning rate scheduler
    
    # Data parameters
    USE_BLOG_DATA = True  # Include blog data in training
    USE_TWEET_DATA = True  # Include tweet data in training
    TWEET_GROUP_SIZE = 10  # Number of tweets to combine per instance
    TEST_SIZE = 0.2  # Proportion of data for testing
    RANDOM_STATE = 42  # Random seed
    
    # Training parameters
    USE_GPU = True  # Use GPU if available
    LOGGING_STEPS = 50  # Log every N steps
    EVAL_STRATEGY = 'epoch'  # Evaluation strategy ('epoch', 'steps', or 'no')
    SAVE_STRATEGY = 'epoch'  # Save strategy
    SAVE_TOTAL_LIMIT = 2  # Maximum number of checkpoints to keep
    LOAD_BEST_MODEL_AT_END = True  # Load best model at the end of training

# ============================================================================


class TextDataset(Dataset):
    """
    Custom Dataset for text classification
    """
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if pd.notna(self.texts[idx]) else ""
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DistilBERTAuthorshipModel:
    """
    DistilBERT model for authorship attribution
    """
    
    def __init__(self, model_name=None, max_length=None, device=None):
        """
        Initialize the model
        
        Parameters:
        -----------
        model_name : str
            HuggingFace model name
        max_length : int
            Maximum sequence length
        device : str or torch.device
            Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name or Config.MODEL_NAME
        self.max_length = max_length or Config.MAX_LENGTH
        
        if device is None:
            self.device = torch.device('cuda' if (Config.USE_GPU and torch.cuda.is_available()) else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.trainer = None
        self.num_labels = None
        
        print(f"Using device: {self.device}")
    
    def load_data(self, csv_path, text_column, author_column, group_size=None):
        """
        Load data from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file
        text_column : str
            Name of text column
        author_column : str
            Name of author column
        group_size : int, optional
            If provided, group texts (for tweets)
            
        Returns:
        --------
        texts : list
            List of text strings
        authors : list
            List of author labels
        """
        print(f"\nLoading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
        if author_column not in df.columns:
            raise ValueError(f"Column '{author_column}' not found in CSV. Available: {list(df.columns)}")
        
        # Filter out empty texts
        df = df[df[text_column].notna() & (df[text_column] != '')]
        
        if group_size is not None and group_size > 1:
            print(f"Grouping texts (group_size={group_size})...")
            texts, authors = self._group_texts(df, text_column, author_column, group_size)
        else:
            texts = df[text_column].tolist()
            authors = df[author_column].tolist()
        
        print(f"Loaded {len(texts)} instances from {len(set(authors))} authors")
        
        return texts, authors
    
    def _group_texts(self, df, text_column, author_column, group_size):
        """
        Group texts by author (useful for tweets)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with text data
        text_column : str
            Name of text column
        author_column : str
            Name of author column
        group_size : int
            Number of texts to group together
            
        Returns:
        --------
        grouped_texts : list
            List of combined text strings
        grouped_authors : list
            List of author labels for each group
        """
        grouped_texts = []
        grouped_authors = []
        
        for author in df[author_column].unique():
            author_texts = df[df[author_column] == author][text_column].tolist()
            
            # Filter out empty texts
            author_texts = [t for t in author_texts if pd.notna(t) and str(t).strip() != '']
            
            if len(author_texts) < group_size * 0.5:
                continue
            
            # Group texts
            for i in range(0, len(author_texts), group_size):
                text_group = author_texts[i:i+group_size]
                if len(text_group) >= group_size * 0.5:  # At least 50% of group_size
                    combined_text = ' '.join([str(t) for t in text_group])
                    grouped_texts.append(combined_text)
                    grouped_authors.append(author)
        
        return grouped_texts, grouped_authors
    
    def prepare_data(self, texts, authors, test_size=None, random_state=None):
        """
        Prepare and split data for training
        
        Parameters:
        -----------
        texts : list
            List of text strings
        authors : list
            List of author labels
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed
            
        Returns:
        --------
        train_texts, test_texts, train_labels_encoded, test_labels_encoded : tuple
        """
        test_size = test_size or Config.TEST_SIZE
        random_state = random_state or Config.RANDOM_STATE
        
        # Encode labels
        print("\nEncoding author labels...")
        labels_encoded = self.label_encoder.fit_transform(authors)
        self.num_labels = len(self.label_encoder.classes_)
        
        print(f"Number of authors: {self.num_labels}")
        print(f"Authors: {list(self.label_encoder.classes_)}")
        
        # Split data
        print(f"\nSplitting data (test_size={test_size})...")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels_encoded
        )
        
        print(f"Training instances: {len(train_texts)}")
        print(f"Test instances: {len(test_texts)}")
        
        return train_texts, test_texts, train_labels, test_labels
    
    def create_datasets(self, train_texts, train_labels, test_texts=None, test_labels=None):
        """
        Create PyTorch datasets
        
        Parameters:
        -----------
        train_texts : list
            Training texts
        train_labels : array-like
            Training labels (encoded)
        test_texts : list, optional
            Test texts
        test_labels : array-like, optional
            Test labels (encoded)
            
        Returns:
        --------
        train_dataset, test_dataset : tuple
        """
        print("\nInitializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print("Creating training dataset...")
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        
        test_dataset = None
        if test_texts is not None and test_labels is not None:
            print("Creating test dataset...")
            test_dataset = TextDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        return train_dataset, test_dataset
    
    def initialize_model(self):
        """
        Initialize the DistilBERT model
        """
        if self.num_labels is None:
            raise ValueError("Number of labels not set. Call prepare_data() first.")
        
        print(f"\nInitializing model: {self.model_name}")
        print(f"Number of labels: {self.num_labels}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        print(f"Model initialized and moved to {self.device}")
    
    def train(self, train_dataset, eval_dataset=None, output_dir=None):
        """
        Train the model
        
        Parameters:
        -----------
        train_dataset : TextDataset
            Training dataset
        eval_dataset : TextDataset, optional
            Evaluation dataset
        output_dir : str, optional
            Directory to save model checkpoints
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        output_dir = output_dir or Config.MODEL_SAVE_DIR
        
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        # Training arguments - handle version compatibility
        import inspect
        
        training_args_dict = {
            'output_dir': output_dir,
            'num_train_epochs': Config.NUM_EPOCHS,
            'per_device_train_batch_size': Config.BATCH_SIZE,
            'per_device_eval_batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'weight_decay': Config.WEIGHT_DECAY,
            'warmup_steps': Config.WARMUP_STEPS,
            'logging_dir': f'{output_dir}/logs',
            'logging_steps': Config.LOGGING_STEPS,
            'save_strategy': Config.SAVE_STRATEGY,
            'save_total_limit': Config.SAVE_TOTAL_LIMIT,
        }
        
        # Check which parameter name is supported
        valid_params = inspect.signature(TrainingArguments.__init__).parameters
        
        # Add evaluation-related args only if we have eval dataset
        if eval_dataset is not None:
            # Handle both old and new parameter names
            if 'eval_strategy' in valid_params:
                training_args_dict['eval_strategy'] = Config.EVAL_STRATEGY
            elif 'evaluation_strategy' in valid_params:
                training_args_dict['evaluation_strategy'] = Config.EVAL_STRATEGY
            
            training_args_dict.update({
                'load_best_model_at_end': Config.LOAD_BEST_MODEL_AT_END,
                'metric_for_best_model': 'accuracy',
                'greater_is_better': True,
            })
        
        # Add fp16 only if CUDA is available and supported
        if torch.cuda.is_available() and 'fp16' in valid_params:
            training_args_dict['fp16'] = True
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predictions)}
        
        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset is not None else None
        )
        
        # Train
        print("\nStarting training...")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Learning rate: {Config.LEARNING_RATE}")
        
        self.trainer.train()
        
        print("\nTraining complete!")
    
    def evaluate(self, test_dataset, test_labels_original=None):
        """
        Evaluate the model
        
        Parameters:
        -----------
        test_dataset : TextDataset
            Test dataset
        test_labels_original : array-like, optional
            Original test labels (for detailed metrics)
            
        Returns:
        --------
        results : dict
            Evaluation results
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        # Evaluate
        print("\nEvaluating on test set...")
        eval_results = self.trainer.evaluate(test_dataset)
        
        # Get predictions
        print("Generating predictions...")
        predictions = self.trainer.predict(test_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Convert to original labels if provided
        if test_labels_original is not None:
            predicted_authors = self.label_encoder.inverse_transform(predicted_labels)
            true_authors = self.label_encoder.inverse_transform(true_labels)
        else:
            predicted_authors = predicted_labels
            true_authors = true_labels
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Calculate precision, recall, and F1-score (macro-averaged)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            true_labels, 
            predicted_labels,
            average='macro',
            zero_division=0
        )
        
        # Calculate weighted averages as well
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, 
            predicted_labels,
            average='weighted',
            zero_division=0
        )
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print(f"Macro-averaged Recall: {recall*100:.2f}%")
        print(f"Macro-averaged F1-Score: {f1_score*100:.2f}%")
        print(f"Macro-averaged Precision: {precision*100:.2f}%")
        print(f"\nWeighted-averaged Recall: {recall_weighted*100:.2f}%")
        print(f"Weighted-averaged F1-Score: {f1_weighted*100:.2f}%")
        print(f"Weighted-averaged Precision: {precision_weighted*100:.2f}%")
        print(f"\nNumber of authors: {self.num_labels}")
        print(f"Random baseline: {100/self.num_labels:.2f}%")
        print(f"Improvement: {accuracy*100 - 100/self.num_labels:.2f}%")
        
        # Classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        report = classification_report(
            true_labels, 
            predicted_labels,
            target_names=[str(c) for c in self.label_encoder.classes_],
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot confusion matrix if not too large
        if self.num_labels <= 20:
            self._plot_confusion_matrix(cm)
        
        results = {
            'accuracy': accuracy,
            'recall_macro': recall,
            'f1_score_macro': f1_score,
            'precision_macro': precision,
            'recall_weighted': recall_weighted,
            'f1_score_weighted': f1_weighted,
            'precision_weighted': precision_weighted,
            'eval_results': eval_results,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predicted_authors,
            'true_labels': true_authors
        }
        
        return results
    
    def _plot_confusion_matrix(self, cm):
        """
        Plot and save confusion matrix
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix - DistilBERT Authorship Attribution')
        plt.ylabel('True Author')
        plt.xlabel('Predicted Author')
        plt.tight_layout()
        
        ensure_parent_dir(Config.CONFUSION_MATRIX_IMG)
        plt.savefig(Config.CONFUSION_MATRIX_IMG, dpi=300, bbox_inches='tight')
        print(f"\n[SUCCESS] Confusion matrix saved to: {Config.CONFUSION_MATRIX_IMG}")
        plt.close()
    
    def save_model(self, save_dir=None):
        """
        Save the trained model and label encoder
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save the model
        """
        save_dir = save_dir or Config.MODEL_SAVE_DIR
        
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        print(f"\nSaving model to: {save_dir}")
        ensure_parent_dir(save_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save label encoder
        ensure_parent_dir(Config.LABEL_ENCODER_FILE)
        with open(Config.LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to: {Config.LABEL_ENCODER_FILE}")
        
        print("[SUCCESS] Model saved successfully!")
    
    def load_model(self, model_dir=None):
        """
        Load a trained model
        
        Parameters:
        -----------
        model_dir : str, optional
            Directory containing the saved model
        """
        model_dir = model_dir or Config.MODEL_SAVE_DIR
        
        print(f"\nLoading model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        
        # Load label encoder
        with open(Config.LABEL_ENCODER_FILE, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.num_labels = len(self.label_encoder.classes_)
        print(f"[SUCCESS] Model loaded successfully!")
        print(f"Number of authors: {self.num_labels}")
    
    def predict(self, texts):
        """
        Predict authors for new texts
        
        Parameters:
        -----------
        texts : list or str
            Text(s) to predict
            
        Returns:
        --------
        predictions : array or str
            Predicted author(s)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        # Prepare texts
        texts = [str(t) if pd.notna(t) else "" for t in texts]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        
        # Decode labels
        predicted_authors = self.label_encoder.inverse_transform(predictions)
        
        if return_single:
            return predicted_authors[0]
        return predicted_authors


def main():
    """
    Main function to train DistilBERT model
    """
    print("="*70)
    print("DISTILBERT AUTHORSHIP ATTRIBUTION TRAINING")
    print("="*70)
    
    # Initialize model
    model = DistilBERTAuthorshipModel()
    
    # Check if using pre-split data
    if Config.USE_PRESPLIT_DATA:
        print("\nUsing pre-existing train/test split...")
        
        # Load training data
        print(f"\nLoading training data from: {Config.TRAIN_CSV}")
        train_texts, train_authors = model.load_data(
            Config.TRAIN_CSV,
            Config.TEXT_COLUMN,
            Config.AUTHOR_COLUMN,
            group_size=Config.TWEET_GROUP_SIZE if 'tweet' in Config.TRAIN_CSV.lower() else None
        )
        
        # Load test data
        print(f"\nLoading test data from: {Config.TEST_CSV}")
        test_texts, test_authors = model.load_data(
            Config.TEST_CSV,
            Config.TEXT_COLUMN,
            Config.AUTHOR_COLUMN,
            group_size=Config.TWEET_GROUP_SIZE if 'tweet' in Config.TEST_CSV.lower() else None
        )
        
        # Encode labels using all unique authors from both train and test
        print("\nEncoding author labels...")
        all_authors = train_authors + test_authors
        model.label_encoder.fit(all_authors)
        model.num_labels = len(model.label_encoder.classes_)
        
        train_labels = model.label_encoder.transform(train_authors)
        test_labels = model.label_encoder.transform(test_authors)
        
        print(f"\nNumber of authors: {model.num_labels}")
        print(f"Authors: {list(model.label_encoder.classes_)}")
        print(f"Training instances: {len(train_texts)}")
        print(f"Test instances: {len(test_texts)}")
        
    else:
        print("\nUsing automatic train/test split...")
        
        # Load data
        all_texts = []
        all_authors = []
        
        if Config.USE_BLOG_DATA:
            try:
                blog_texts, blog_authors = model.load_data(
                    Config.BLOG_CSV,
                    Config.TEXT_COLUMN,
                    Config.AUTHOR_COLUMN,
                    group_size=None  # Don't group blogs
                )
                all_texts.extend(blog_texts)
                all_authors.extend(blog_authors)
            except Exception as e:
                print(f"[WARNING] Could not load blog data: {e}")
        
        if Config.USE_TWEET_DATA:
            try:
                tweet_texts, tweet_authors = model.load_data(
                    Config.TWEET_CSV,
                    Config.TEXT_COLUMN,
                    Config.AUTHOR_COLUMN,
                    group_size=Config.TWEET_GROUP_SIZE
                )
                all_texts.extend(tweet_texts)
                all_authors.extend(tweet_authors)
            except Exception as e:
                print(f"[WARNING] Could not load tweet data: {e}")
        
        if len(all_texts) == 0:
            print("[ERROR] No data loaded. Check your configuration.")
            return
        
        print(f"\nTotal instances: {len(all_texts)}")
        print(f"Total authors: {len(set(all_authors))}")
        
        # Prepare data with automatic split
        train_texts, test_texts, train_labels, test_labels = model.prepare_data(
            all_texts, all_authors
        )
    
    # Create datasets
    train_dataset, test_dataset = model.create_datasets(
        train_texts, train_labels,
        test_texts, test_labels
    )
    
    # Initialize model
    model.initialize_model()
    
    # Train
    model.train(train_dataset, eval_dataset=test_dataset)
    
    # Evaluate
    results = model.evaluate(test_dataset, test_labels)
    
    # Save model
    model.save_model()
    
    # Save results to file
    ensure_parent_dir(Config.RESULTS_FILE)
    with open(Config.RESULTS_FILE, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DISTILBERT AUTHORSHIP ATTRIBUTION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Max Length: {Config.MAX_LENGTH}\n")
        f.write(f"Epochs: {Config.NUM_EPOCHS}\n")
        f.write(f"Batch Size: {Config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {Config.LEARNING_RATE}\n")
        f.write(f"Data Mode: {'Pre-split' if Config.USE_PRESPLIT_DATA else 'Auto-split'}\n\n")
        f.write(f"Number of Authors: {model.num_labels}\n")
        f.write(f"Training Instances: {len(train_texts)}\n")
        f.write(f"Test Instances: {len(test_texts)}\n\n")
        f.write("="*70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"Accuracy: {results['accuracy']*100:.2f}%\n")
        f.write(f"Random Baseline: {100/model.num_labels:.2f}%\n")
        f.write(f"Improvement: {results['accuracy']*100 - 100/model.num_labels:.2f}%\n\n")
        f.write("Macro-averaged Metrics:\n")
        f.write(f"  Precision: {results['precision_macro']*100:.2f}%\n")
        f.write(f"  Recall: {results['recall_macro']*100:.2f}%\n")
        f.write(f"  F1-Score: {results['f1_score_macro']*100:.2f}%\n\n")
        f.write("Weighted-averaged Metrics:\n")
        f.write(f"  Precision: {results['precision_weighted']*100:.2f}%\n")
        f.write(f"  Recall: {results['recall_weighted']*100:.2f}%\n")
        f.write(f"  F1-Score: {results['f1_score_weighted']*100:.2f}%\n\n")
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(results['classification_report'])
    
    print(f"\n[SUCCESS] Results saved to: {Config.RESULTS_FILE}")
    
    # Example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    if len(test_texts) > 0:
        example_indices = np.random.choice(len(test_texts), min(3, len(test_texts)), replace=False)
        for idx in example_indices:
            text = test_texts[idx]
            true_author = model.label_encoder.inverse_transform([test_labels[idx]])[0]
            predicted_author = model.predict(text)
            
            print(f"\nText: {text[:200]}...")
            print(f"True Author: {true_author}")
            print(f"Predicted Author: {predicted_author}")
            print(f"Correct: {'v' if true_author == predicted_author else 'x'}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
