"""
Tweet-Only Model Training Script
Simplified version that only trains tweet models (no blog models)
For faster training when comparing approaches.
"""

import pandas as pd
import numpy as np
import re
import string
import emoji
from collections import Counter
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from utils.feature_selection import select_features_mi as select_features_mi_shared
from utils.paths import ensure_parent_dir, project_path, results_path

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words, stopwords
from nltk import pos_tag

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available. Install with: pip install xgboost")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for data sources and model parameters"""
    
    # Input data files
    TWEET_CSV = str(project_path('training_data', 'selected_tweets.csv'))
    
    # Column names
    TEXT_COLUMN = 'Text'
    AUTHOR_COLUMN = 'Author Name'
    
    # Data parameters
    TWEET_GROUP_SIZE = 10
    
    # Train/test split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Feature extraction parameters
    TWEET_USE_FEATURE_SELECTION = False
    TWEET_FEATURE_SELECTION_RATIO = 0.05
    
    # Model parameters
    TWEET_MODEL_TYPE = 'all'  # Options: 'svm', 'logistic_regression', 'xgboost', 'bagging', 'voting', 'all'
    
    # SVM parameters
    SVM_C = 1.0
    SVM_KERNEL = 'linear'
    
    # Logistic Regression parameters
    LR_C = 1.0
    LR_MAX_ITER = 1000
    
    # XGBoost parameters
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1
    XGB_N_ESTIMATORS = 100
    
    # Bagging parameters
    BAGGING_N_ESTIMATORS = 10
    BAGGING_MAX_SAMPLES = 1.0
    BAGGING_BASE_ESTIMATOR = 'decision_tree'
    
    # Voting parameters
    VOTING_METHOD = 'soft'
    VOTING_ESTIMATORS = ['logistic_regression', 'svm', 'xgboost']
    
    # Output files
    TWEET_MODEL_FILE = str(results_path('tweet_only', 'tweet_model_only.pkl'))
    RESULTS_FILE = str(results_path('tweet_only', 'evaluation_results_tweet_only.txt'))
    CONFUSION_MATRIX_IMG = str(results_path('tweet_only', 'confusion_matrix_tweet_only.png'))


# ============================================================================
# TWEET FEATURE EXTRACTOR
# ============================================================================

class TweetFeatureExtractor:
    """Extract tweet-specific features"""
    
    def __init__(self):
        self.english_words = set(words.words())
        self.stop_words = set(stopwords.words('english'))
    
    def extract_emojis(self, text):
        return [c for c in text if c in emoji.EMOJI_DATA]
    
    def count_word_extensions(self, text):
        pattern = r'\b\w*(\w)\1{2,}\w*\b'
        return len(re.findall(pattern, text, re.IGNORECASE))
    
    def count_alphanumeric_words(self, text):
        words_list = text.split()
        alphanumeric = [w for w in words_list if re.search(r'[a-zA-Z]', w) and re.search(r'\d', w)]
        return len(alphanumeric)
    
    def count_all_caps_words(self, text):
        words_list = text.split()
        all_caps = [w for w in words_list if w.isupper() and len(w) > 1 and w.isalpha()]
        return len(all_caps)
    
    def count_hashtags(self, text):
        return len(re.findall(r'#\w+', text))
    
    def count_user_mentions(self, text):
        return len(re.findall(r'@\w+', text))
    
    def count_urls(self, text):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return len(re.findall(url_pattern, text))
    
    def is_retweet(self, text):
        return 1 if text.strip().startswith('RT @') or text.strip().startswith('RT@') else 0
    
    def has_quotation(self, text):
        return 1 if '"' in text or "'" in text or '"' in text or '"' in text or "'" in text or "'" in text else 0
    
    def count_dictionary_words(self, text):
        try:
            words_list = word_tokenize(text.lower())
        except:
            words_list = text.lower().split()
        dict_words = [w for w in words_list if w.isalpha() and w in self.english_words]
        return len(dict_words)
    
    def calculate_lexical_diversity(self, text):
        try:
            words_list = word_tokenize(text.lower())
        except:
            words_list = text.lower().split()
        words_list = [w for w in words_list if w.isalpha()]
        if len(words_list) == 0:
            return 0
        return len(set(words_list)) / len(words_list)
    
    def count_special_chars(self, text):
        special_chars = r'[*#%^&@$!~`]'
        return len(re.findall(special_chars, text))
    
    def extract_features_from_tweets(self, tweets_list):
        """Extract features from a group of tweets"""
        if not tweets_list or len(tweets_list) == 0:
            return None
        
        # Aggregate features from all tweets in group
        all_features = []
        for tweet in tweets_list:
            single_features = self.extract_single_tweet_features(tweet)
            if single_features:
                all_features.append(single_features)
        
        if not all_features:
            return None
        
        # Average features across all tweets in group
        features_df = pd.DataFrame(all_features)
        avg_features = features_df.mean().to_dict()
        
        # Add mimicry features
        mimicry = self.calculate_mimicry_features(tweets_list)
        avg_features.update(mimicry)
        
        return avg_features
    
    def extract_single_tweet_features(self, text):
        if pd.isna(text) or text.strip() == '':
            return None
        
        original_text = text
        features = {}
        
        # Twitter-specific features (from original text)
        features['is_retweet'] = self.is_retweet(original_text)
        num_hashtags = self.count_hashtags(original_text)
        num_mentions = self.count_user_mentions(original_text)
        num_urls = self.count_urls(original_text)
        
        # Clean text for word counting (remove URLs, mentions, hashtags)
        clean_text = re.sub(r'http[s]?://\S+', '', text)
        clean_text = re.sub(r'@\w+', '', clean_text)
        clean_text = re.sub(r'#\w+', '', clean_text)
        
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = [text]
        
        try:
            clean_words = word_tokenize(clean_text)
            clean_words = [w for w in clean_words if w.isalpha()]
        except:
            clean_words = clean_text.split()
            clean_words = [w for w in clean_words if w.isalpha()]
        
        num_words = len(clean_words)
        num_chars = len(text)
        
        # Twitter-specific frequency features
        features['hashtags_freq'] = num_hashtags / num_words if num_words > 0 else 0
        features['user_mentions_freq'] = num_mentions / num_words if num_words > 0 else 0
        features['urls_freq'] = num_urls / num_words if num_words > 0 else 0
        
        # Lexical features
        features['total_words'] = num_words
        features['total_sentences'] = len(sentences)
        features['words_per_sentence'] = num_words / len(sentences) if len(sentences) > 0 else 0
        features['avg_word_length'] = np.mean([len(w) for w in clean_words]) if clean_words else 0
        features['dictionary_words_freq'] = self.count_dictionary_words(text) / num_words if num_words > 0 else 0
        features['word_extensions_freq'] = self.count_word_extensions(text) / num_words if num_words > 0 else 0
        features['lexical_diversity'] = self.calculate_lexical_diversity(text)
        
        # Syntactical features
        features['bos_capitalized'] = sum(1 for s in sentences if s and s[0].isupper())
        features['punctuations_per_sentence'] = len(re.findall(r'[,.!?;:]', text)) / len(sentences) if len(sentences) > 0 else 0
        features['all_caps_words_freq'] = self.count_all_caps_words(text) / num_words if num_words > 0 else 0
        features['alphanumeric_words_freq'] = self.count_alphanumeric_words(text) / num_words if num_words > 0 else 0
        features['special_chars_count'] = self.count_special_chars(text)
        features['digits_count'] = sum(c.isdigit() for c in text)
        features['exclamation_marks'] = text.count('!')
        features['question_marks'] = text.count('?')
        features['uppercase_letters'] = sum(1 for c in text if c.isupper())
        features['has_quotation'] = self.has_quotation(text)
        
        # Emoji features
        emojis = self.extract_emojis(original_text)
        features['emoji_count'] = len(emojis)
        features['emoji_per_word'] = len(emojis) / num_words if num_words > 0 else 0
        features['emoji_per_char'] = len(emojis) / num_chars if num_chars > 0 else 0
        
        return features
    
    def calculate_mimicry_features(self, tweets_list):
        if len(tweets_list) < 2:
            return {
                'mimicry_length_avg': 0,
                'mimicry_words_avg': 0
            }
        
        mimicry_length = []
        mimicry_words = []
        
        for i in range(len(tweets_list) - 1):
            tweet1 = tweets_list[i]
            tweet2 = tweets_list[i + 1]
            
            len1 = len(tweet1.split())
            len2 = len(tweet2.split())
            if len2 > 0:
                mimicry_length.append(len1 / len2)
            
            try:
                words1 = set(word_tokenize(tweet1.lower()))
                words2 = set(word_tokenize(tweet2.lower()))
            except:
                words1 = set(tweet1.lower().split())
                words2 = set(tweet2.lower().split())
            
            common = len(words1.intersection(words2))
            mimicry_words.append(common)
        
        return {
            'mimicry_length_avg': np.mean(mimicry_length) if mimicry_length else 0,
            'mimicry_words_avg': np.mean(mimicry_words) if mimicry_words else 0
        }
    
    def group_tweets(self, df, group_size=10):
        grouped_data = []
        
        for author in df[Config.AUTHOR_COLUMN].unique():
            author_tweets = df[df[Config.AUTHOR_COLUMN] == author][Config.TEXT_COLUMN].tolist()
            author_tweets = [t for t in author_tweets if pd.notna(t) and str(t).strip() != '']
            
            if len(author_tweets) < group_size * 0.5:
                continue
            
            for i in range(0, len(author_tweets), group_size):
                group = author_tweets[i:i + group_size]
                if len(group) >= group_size * 0.5:
                    grouped_data.append({
                        'author': author,
                        'tweets': group
                    })
        
        return grouped_data
    
    def process_tweets(self, tweet_groups, authors):
        """Process groups of tweets and extract features"""
        features_list = []
        valid_authors = []
        
        for tweets, author in zip(tweet_groups, authors):
            features = self.extract_features_from_tweets(tweets)
            if features:
                features_list.append(features)
                valid_authors.append(author)
        
        if not features_list:
            raise ValueError("No valid features extracted!")
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        return features_df, pd.Series(valid_authors)


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_mi(X, y, ratio=0.05, random_state=42):
    """Select top features using mutual information"""
    selected_features, _ = select_features_mi_shared(
        X,
        y,
        ratio=ratio,
        random_state=random_state,
        top_n_preview=5,
    )
    return selected_features


# ============================================================================
# AUTHORSHIP MODEL
# ============================================================================

class AuthorshipModel:
    """Tweet authorship attribution model"""
    
    def __init__(self, model_type='svm', random_state=42,
                 svm_c=Config.SVM_C, svm_kernel=Config.SVM_KERNEL,
                 lr_c=Config.LR_C, lr_max_iter=Config.LR_MAX_ITER,
                 xgb_max_depth=Config.XGB_MAX_DEPTH, xgb_learning_rate=Config.XGB_LEARNING_RATE,
                 xgb_n_estimators=Config.XGB_N_ESTIMATORS,
                 bagging_n_estimators=Config.BAGGING_N_ESTIMATORS,
                 bagging_max_samples=Config.BAGGING_MAX_SAMPLES,
                 bagging_base_estimator=Config.BAGGING_BASE_ESTIMATOR,
                 voting_method=Config.VOTING_METHOD,
                 voting_estimators=Config.VOTING_ESTIMATORS):
        
        self.model_type = model_type
        self.random_state = random_state
        self.svm_c = svm_c
        self.svm_kernel = svm_kernel
        self.lr_c = lr_c
        self.lr_max_iter = lr_max_iter
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_n_estimators = xgb_n_estimators
        self.bagging_n_estimators = bagging_n_estimators
        self.bagging_max_samples = bagging_max_samples
        self.bagging_base_estimator = bagging_base_estimator
        self.voting_method = voting_method
        self.voting_estimators = voting_estimators
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.training_history = {}
    
    def _create_classifier(self):
        if self.model_type == 'svm':
            return SVC(
                kernel=self.svm_kernel,
                C=self.svm_c,
                random_state=self.random_state,
                probability=True
            )
        elif self.model_type == 'logistic_regression':
            lr = LogisticRegression(
                C=self.lr_c,
                max_iter=self.lr_max_iter,
                random_state=self.random_state,
                solver='lbfgs'
            )
            return OneVsRestClassifier(lr)
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost not available. Install with: pip install xgboost")
            return XGBClassifier(
                max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate,
                n_estimators=self.xgb_n_estimators,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'bagging':
            if self.bagging_base_estimator == 'svm':
                base_estimator = SVC(
                    kernel=self.svm_kernel,
                    C=self.svm_c,
                    random_state=self.random_state,
                    probability=True
                )
            elif self.bagging_base_estimator == 'decision_tree':
                from sklearn.tree import DecisionTreeClassifier
                base_estimator = DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10
                )
            
            try:
                return BaggingClassifier(
                    estimator=base_estimator,
                    n_estimators=self.bagging_n_estimators,
                    max_samples=self.bagging_max_samples,
                    random_state=self.random_state,
                    verbose=1,
                    n_jobs=-1
                )
            except TypeError:
                return BaggingClassifier(
                    base_estimator=base_estimator,
                    n_estimators=self.bagging_n_estimators,
                    max_samples=self.bagging_max_samples,
                    random_state=self.random_state,
                    verbose=1,
                    n_jobs=-1
                )
        elif self.model_type == 'voting':
            estimators = []
            for est_name in self.voting_estimators:
                if est_name == 'logistic_regression':
                    lr = LogisticRegression(
                        C=self.lr_c,
                        max_iter=self.lr_max_iter,
                        random_state=self.random_state,
                        solver='lbfgs'
                    )
                    estimators.append(('lr', OneVsRestClassifier(lr)))
                elif est_name == 'svm':
                    estimators.append(('svm', SVC(
                        kernel=self.svm_kernel,
                        C=self.svm_c,
                        random_state=self.random_state,
                        probability=True
                    )))
                elif est_name == 'xgboost':
                    if not XGBOOST_AVAILABLE:
                        continue
                    estimators.append(('xgb', XGBClassifier(
                        max_depth=self.xgb_max_depth,
                        learning_rate=self.xgb_learning_rate,
                        n_estimators=self.xgb_n_estimators,
                        random_state=self.random_state,
                        eval_metric='mlogloss'
                    )))
            
            if len(estimators) == 0:
                raise ValueError("No valid estimators for voting!")
            
            return VotingClassifier(
                estimators=estimators,
                voting=self.voting_method,
                n_jobs=-1
            )
    
    def train(self, X_train, y_train):
        print(f"\nTraining {self.model_type.upper().replace('_', ' ')} model...")
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_authors = len(self.label_encoder.classes_)
        
        X_train = X_train.fillna(0)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.classifier = self._create_classifier()
        self.classifier.fit(X_train_scaled, y_train_encoded)
        
        y_train_pred = self.classifier.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        
        self.training_history.update({
            'n_authors': n_authors,
            'n_features': X_train.shape[1],
            'train_accuracy': train_accuracy
        })
        
        print(f"Training complete - Accuracy: {train_accuracy*100:.2f}%")
        return self
    
    def predict(self, X_test):
        if self.classifier is None:
            raise ValueError("Model not trained!")
        
        X_test = X_test.fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_encoded = self.classifier.predict(X_test_scaled)
        predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return predictions
    
    def predict_proba(self, X_test):
        """Predict class probabilities"""
        if self.classifier is None:
            raise ValueError("Model not trained!")
        
        X_test = X_test.fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        return self.classifier.predict_proba(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        
        results = {
            'accuracy': accuracy,
            'recall_macro': recall_macro,
            'f1_score_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_weighted': recall_weighted,
            'f1_score_weighted': f1_weighted,
            'precision_weighted': precision_weighted,
            'n_authors': self.training_history['n_authors'],
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return results


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_and_compare_all_models(X_train, y_train, X_test, y_test, model_prefix="TWEET"):
    """Train and compare all model types"""
    print(f"\n{model_prefix} - Training and comparing all models...")
    
    model_types = ['svm', 'logistic_regression', 'bagging', 'voting']
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    
    results_all = {}
    models_all = {}
    
    for i, model_type in enumerate(model_types, 1):
        print(f"\n[{i}/{len(model_types)}] Training {model_type.upper().replace('_', ' ')}...")
        
        try:
            model = AuthorshipModel(model_type=model_type)
            model.train(X_train, y_train)
            result = model.evaluate(X_test, y_test)
            
            results_all[model_type] = result
            models_all[model_type] = model
            
            print(f"  Test Accuracy: {result['accuracy']*100:.2f}%")
            print(f"  Macro F1: {result['f1_score_macro']*100:.2f}%")
            
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            results_all[model_type] = None
            models_all[model_type] = None
    
    # Find best model
    best_model_name = max(results_all.items(), key=lambda x: x[1]['accuracy'] if x[1] else 0)[0]
    best_model = models_all[best_model_name]
    
    print(f"\n{model_prefix} - Best Model: {best_model_name.upper().replace('_', ' ')}")
    print(f"  Accuracy: {results_all[best_model_name]['accuracy']*100:.2f}%")
    
    return results_all, models_all, best_model_name, best_model


def save_results(tweet_results_all, tweet_best_name, n_tweet_train, n_tweet_test):
    """Save comprehensive results"""
    ensure_parent_dir(Config.RESULTS_FILE)
    with open(Config.RESULTS_FILE, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TWEET-ONLY MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Training instances: {n_tweet_train}\n")
        f.write(f"Test instances: {n_tweet_test}\n\n")
        
        # Tweet results
        f.write("TWEET MODEL RESULTS (All Models):\n")
        f.write("-"*70 + "\n")
        for model_name, results in tweet_results_all.items():
            if results:
                f.write(f"{model_name.upper().replace('_', ' ')}:\n")
                f.write(f"  Accuracy: {results['accuracy']*100:.2f}%\n")
                f.write(f"  Macro F1: {results['f1_score_macro']*100:.2f}%\n")
                f.write(f"  Macro Recall: {results['recall_macro']*100:.2f}%\n\n")
        
        f.write(f"Best Tweet Model: {tweet_best_name.upper().replace('_', ' ')}\n\n")
        
        # Detailed results for best tweet model
        tweet_results = tweet_results_all[tweet_best_name]
        f.write("="*70 + "\n")
        f.write(f"BEST TWEET MODEL DETAILED RESULTS\n")
        f.write(f"Model: {tweet_best_name.upper().replace('_', ' ')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy: {tweet_results['accuracy']*100:.2f}%\n")
        f.write(f"Macro Precision: {tweet_results['precision_macro']*100:.2f}%\n")
        f.write(f"Macro Recall: {tweet_results['recall_macro']*100:.2f}%\n")
        f.write(f"Macro F1-Score: {tweet_results['f1_score_macro']*100:.2f}%\n\n")
        f.write(f"Weighted Precision: {tweet_results['precision_weighted']*100:.2f}%\n")
        f.write(f"Weighted Recall: {tweet_results['recall_weighted']*100:.2f}%\n")
        f.write(f"Weighted F1-Score: {tweet_results['f1_score_weighted']*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(tweet_results['classification_report'])
    
    print(f"[SUCCESS] Results saved to: {Config.RESULTS_FILE}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("="*70)
    print("TWEET-ONLY MODEL TRAINING")
    print("="*70)
    
    # [1] Load and process TWEET data
    print("\n[1/4] Loading tweet data...")
    tweet_df = pd.read_csv(Config.TWEET_CSV)
    print(f"  Loaded {tweet_df.shape[0]} tweets")
    
    tweet_extractor = TweetFeatureExtractor()
    grouped_tweets = tweet_extractor.group_tweets(tweet_df, Config.TWEET_GROUP_SIZE)
    print(f"  Created {len(grouped_tweets)} tweet groups")
    
    # [2] Split tweet data
    print(f"\n[2/4] Splitting tweet data ({(1-Config.TEST_SIZE)*100:.0f}%/{Config.TEST_SIZE*100:.0f}%)...")
    tweet_groups_train, tweet_groups_test = train_test_split(
        grouped_tweets,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=[g['author'] for g in grouped_tweets]
    )
    print(f"  Tweet train groups: {len(tweet_groups_train)}")
    print(f"  Tweet test groups: {len(tweet_groups_test)}")
    
    # [3] Extract tweet features
    print("\n[3/4] Extracting tweet features...")
    tweet_train_features, tweet_train_labels = tweet_extractor.process_tweets(
        [g['tweets'] for g in tweet_groups_train],
        [g['author'] for g in tweet_groups_train]
    )
    tweet_test_features, tweet_test_labels = tweet_extractor.process_tweets(
        [g['tweets'] for g in tweet_groups_test],
        [g['author'] for g in tweet_groups_test]
    )
    
    if Config.TWEET_USE_FEATURE_SELECTION:
        selected_tweet_features = select_features_mi(
            tweet_train_features, tweet_train_labels,
            ratio=Config.TWEET_FEATURE_SELECTION_RATIO
        )
        tweet_train_features = tweet_train_features[selected_tweet_features]
        tweet_test_features = tweet_test_features[selected_tweet_features]
        
        assert list(tweet_train_features.columns) == list(tweet_test_features.columns), \
            "Tweet train and test features are not aligned!"
    
    print(f"  Final tweet features: {tweet_train_features.shape[1]}")
    print(f"  Tweet train shape: {tweet_train_features.shape}")
    print(f"  Tweet test shape: {tweet_test_features.shape}")
    
    # [4] Train tweet models
    print("\n[4/4] Training tweet models...")
    print(f"Tweet train: {len(tweet_train_labels)} instances, {len(set(tweet_train_labels))} unique authors")
    print(f"Tweet test: {len(tweet_test_labels)} instances, {len(set(tweet_test_labels))} unique authors")
    
    # Check for label mismatch
    train_authors_set = set(tweet_train_labels)
    test_authors_set = set(tweet_test_labels)
    missing_in_train = test_authors_set - train_authors_set
    if missing_in_train:
        print(f"\n[WARNING] {len(missing_in_train)} authors in test but NOT in train: {list(missing_in_train)[:5]}")
        print("These instances will likely be misclassified!")
    
    if Config.TWEET_MODEL_TYPE == 'all':
        tweet_results_all, tweet_models_all, tweet_best_name, tweet_best_model = train_and_compare_all_models(
            tweet_train_features, tweet_train_labels,
            tweet_test_features, tweet_test_labels,
            model_prefix="TWEET"
        )
    else:
        tweet_best_model = AuthorshipModel(model_type=Config.TWEET_MODEL_TYPE)
        tweet_best_model.train(tweet_train_features, tweet_train_labels)
        tweet_results_all = {Config.TWEET_MODEL_TYPE: tweet_best_model.evaluate(tweet_test_features, tweet_test_labels)}
        tweet_models_all = {Config.TWEET_MODEL_TYPE: tweet_best_model}
        tweet_best_name = Config.TWEET_MODEL_TYPE
    
    # Print results
    print("\n" + "="*70)
    print("SUMMARY - TWEET MODEL RESULTS")
    print("="*70)
    
    print(f"\nBest Model: {tweet_best_name.upper().replace('_', ' ')}")
    print(f"  Test Accuracy: {tweet_results_all[tweet_best_name]['accuracy']*100:.2f}%")
    print(f"  Macro F1: {tweet_results_all[tweet_best_name]['f1_score_macro']*100:.2f}%")
    print(f"  Macro Recall: {tweet_results_all[tweet_best_name]['recall_macro']*100:.2f}%")
    
    # Save results
    save_results(
        tweet_results_all, tweet_best_name,
        tweet_train_features.shape[0],
        tweet_test_features.shape[0]
    )
    
    # Save model
    ensure_parent_dir(Config.TWEET_MODEL_FILE)
    with open(Config.TWEET_MODEL_FILE, 'wb') as f:
        pickle.dump(tweet_best_model, f)
    
    print(f"\n[SUCCESS] Model saved to: {Config.TWEET_MODEL_FILE}")
    
    return tweet_best_model, tweet_results_all[tweet_best_name], tweet_results_all


if __name__ == "__main__":
    tweet_model, best_results, all_results = main()
