import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from utils.feature_selection import select_features_mi
from utils.paths import ensure_parent_dir, project_path, results_path
from utils.stylometric_features import UnifiedFeatureExtractor


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for data sources and model parameters"""
    
    # Input data files
    BLOG_CSV = str(project_path('training_data', 'selected_blogs.csv'))
    TWEET_CSV = str(project_path('training_data', 'selected_tweets.csv'))
    
    # Column names
    TEXT_COLUMN = 'Text'
    AUTHOR_COLUMN = 'Author Name'
    BLOG_TITLE_COLUMN = 'Title'
    
    # Data parameters
    USE_BLOG_TITLE = True
    TWEET_GROUP_SIZE = 10  # Number of tweets to combine per instance
    USE_SEMANTIC_CLUSTERING = False
    SEMANTIC_CLUSTER_K = 10
    SEMANTIC_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    # Train/test split (ONLY for tweet data, blog uses all data)
    TEST_SIZE = 0.2  # 20% of tweets for testing, 80% for training
    RANDOM_STATE = 42
    USE_STRATIFIED_KFOLD = True
    CV_N_SPLITS = 5
    
    # Domain weighting (tweet vs blog)
    USE_DOMAIN_WEIGHTING = False
    TWEET_WEIGHT = 3.0
    BLOG_WEIGHT = 1.0
    
    # Feature extraction parameters
    USE_4GRAMS = True
    TOP_4GRAMS = 1000
    USE_FEATURE_SELECTION = True
    FEATURE_SELECTION_RATIO = 0.05
    
    # Model parameters
    MODEL_TYPE = 'all'  # Options: 'svm', 'logistic_regression', 'xgboost', 'bagging', 'voting', 'all'
    
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
    MODEL_FILE = str(results_path('unified', 'authorship_model_blog_tweet.pkl'))
    SCALER_FILE = str(results_path('unified', 'feature_scaler.pkl'))
    LABEL_ENCODER_FILE = str(results_path('unified', 'label_encoder.pkl'))
    FEATURE_NAMES_FILE = str(results_path('unified', 'feature_names.pkl'))
    RESULTS_FILE = str(results_path('unified', 'evaluation_results_blog_tweet.txt'))
    CONFUSION_MATRIX_IMG = str(results_path('unified', 'confusion_matrix_blog_tweet.png'))


def make_feature_extractor():
    return UnifiedFeatureExtractor(
        top_4grams=Config.TOP_4GRAMS,
        use_4grams=Config.USE_4GRAMS,
        text_column=Config.TEXT_COLUMN,
        author_column=Config.AUTHOR_COLUMN,
        use_semantic_clustering=Config.USE_SEMANTIC_CLUSTERING,
        semantic_cluster_k=Config.SEMANTIC_CLUSTER_K,
        semantic_model_name=Config.SEMANTIC_MODEL_NAME,
        random_state=Config.RANDOM_STATE,
        tweet_group_size=Config.TWEET_GROUP_SIZE,
    )


# ============================================================================
# MODEL CLASS
# ============================================================================

class IntegratedAuthorshipModel:
    """Integrated authorship attribution model with on-the-fly feature extraction"""
    
    def __init__(self, model_type=None, random_state=None, **kwargs):
        self.model_type = model_type or Config.MODEL_TYPE
        self.random_state = random_state or Config.RANDOM_STATE
        
        # Validate model type
        valid_models = ['svm', 'logistic_regression', 'xgboost', 'bagging', 'voting', 'all']
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_models}")
        
        # Model parameters
        self.svm_c = kwargs.get('svm_c', Config.SVM_C)
        self.svm_kernel = kwargs.get('svm_kernel', Config.SVM_KERNEL)
        self.lr_c = kwargs.get('lr_c', Config.LR_C)
        self.lr_max_iter = kwargs.get('lr_max_iter', Config.LR_MAX_ITER)
        self.xgb_max_depth = kwargs.get('xgb_max_depth', Config.XGB_MAX_DEPTH)
        self.xgb_learning_rate = kwargs.get('xgb_learning_rate', Config.XGB_LEARNING_RATE)
        self.xgb_n_estimators = kwargs.get('xgb_n_estimators', Config.XGB_N_ESTIMATORS)
        self.bagging_n_estimators = kwargs.get('bagging_n_estimators', Config.BAGGING_N_ESTIMATORS)
        self.bagging_max_samples = kwargs.get('bagging_max_samples', Config.BAGGING_MAX_SAMPLES)
        self.bagging_base_estimator = kwargs.get('bagging_base_estimator', Config.BAGGING_BASE_ESTIMATOR)
        self.voting_method = kwargs.get('voting_method', Config.VOTING_METHOD)
        self.voting_estimators = kwargs.get('voting_estimators', Config.VOTING_ESTIMATORS)
        
        # Components
        self.feature_extractor = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.classifier = None
        self.selected_features = None
        
        # Training history
        self.training_history = {
            'n_authors': None,
            'n_features': None,
            'train_accuracy': None,
            'model_type': self.model_type,
            'n_blog_instances': None,
            'n_tweet_train_instances': None,
            'n_tweet_test_instances': None
        }
    
    def _create_classifier(self):
        """Create classifier based on model_type"""
        if self.model_type == 'svm':
            return SVC(
                kernel=self.svm_kernel,
                C=self.svm_c,
                random_state=self.random_state,
                verbose=True
            )
        elif self.model_type == 'logistic_regression':
            lr = LogisticRegression(
                C=self.lr_c,
                max_iter=self.lr_max_iter,
                random_state=self.random_state,
                solver='lbfgs',
                verbose=1
            )
            return OneVsRestClassifier(lr)
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            return XGBClassifier(
                max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate,
                n_estimators=self.xgb_n_estimators,
                random_state=self.random_state,
                verbosity=1,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'bagging':
            if self.bagging_base_estimator == 'logistic_regression':
                lr = LogisticRegression(
                    C=self.lr_c,
                    max_iter=self.lr_max_iter,
                    random_state=self.random_state,
                    solver='lbfgs'
                )
                base_estimator = OneVsRestClassifier(lr)
            elif self.bagging_base_estimator == 'svm':
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
            else:
                raise ValueError(f"Invalid bagging_base_estimator: {self.bagging_base_estimator}")
            
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
                        probability=True if self.voting_method == 'soft' else False
                    )))
                elif est_name == 'xgboost':
                    if not XGBOOST_AVAILABLE:
                        print(f"[WARNING] XGBoost not available, skipping in voting ensemble")
                        continue
                    estimators.append(('xgb', XGBClassifier(
                        max_depth=self.xgb_max_depth,
                        learning_rate=self.xgb_learning_rate,
                        n_estimators=self.xgb_n_estimators,
                        random_state=self.random_state,
                        eval_metric='mlogloss'
                    )))
                elif est_name == 'decision_tree':
                    from sklearn.tree import DecisionTreeClassifier
                    estimators.append(('dt', DecisionTreeClassifier(
                        random_state=self.random_state,
                        max_depth=10
                    )))
            
            if len(estimators) == 0:
                raise ValueError("No valid estimators for voting classifier!")
            
            return VotingClassifier(
                estimators=estimators,
                voting=self.voting_method,
                n_jobs=-1
            )
    
    def _build_domain_sample_weights(self, domain_labels, domain_weights):
        """Build normalized sample weights based on data domain labels."""
        if domain_labels is None:
            return None
        
        weights = np.array([
            float(domain_weights.get(str(domain).lower(), 1.0))
            for domain in domain_labels
        ], dtype=float)
        
        mean_weight = np.mean(weights)
        if mean_weight > 0:
            weights = weights / mean_weight
        
        return weights
    
    def _resample_with_weights(self, X, y, sample_weights):
        """Resample training data according to sample weights as fallback when model doesn't support sample_weight."""
        if sample_weights is None or len(sample_weights) != len(y):
            return X, y
        
        probs = sample_weights / sample_weights.sum()
        rng = np.random.RandomState(self.random_state)
        sampled_indices = rng.choice(
            np.arange(len(y)),
            size=len(y),
            replace=True,
            p=probs
        )
        return X[sampled_indices], y[sampled_indices]
    
    def train(self, X_train, y_train, n_blog=None, n_tweet_train=None, domain_labels=None, domain_weights=None):
        """Train the model"""
        print("\n" + "="*70)
        print("CROSS-DOMAIN AUTHORSHIP ATTRIBUTION MODEL TRAINING")
        print(f"Model: {self.model_type.upper().replace('_', ' ')}")
        print("Training on: ALL Blog Data + Tweet Train Data")
        print("="*70)
        
        # Encode labels
        print("\nEncoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_authors = len(self.label_encoder.classes_)
        
        print(f"Number of authors: {n_authors}")
        print(f"Number of features: {X_train.shape[1]}")
        if n_blog and n_tweet_train:
            print(f"Training instances:")
            print(f"  - Blog (all): {n_blog}")
            print(f"  - Tweet (train): {n_tweet_train}")
            print(f"  - Total: {X_train.shape[0]}")
        
        # Fill NaN values
        X_train = X_train.fillna(0)
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build domain-based sample weights
        sample_weights = None
        if domain_labels is not None and Config.USE_DOMAIN_WEIGHTING:
            if domain_weights is None:
                domain_weights = {
                    'tweet': Config.TWEET_WEIGHT,
                    'blog': Config.BLOG_WEIGHT
                }
            sample_weights = self._build_domain_sample_weights(domain_labels, domain_weights)
            print("\nDomain weighting enabled:")
            print(f"  - tweet weight: {domain_weights.get('tweet', 1.0)}")
            print(f"  - blog weight: {domain_weights.get('blog', 1.0)}")
        
        # Create and train classifier
        print(f"\nTraining {self.model_type} classifier...")
        self.classifier = self._create_classifier()
        
        if sample_weights is not None:
            try:
                self.classifier.fit(X_train_scaled, y_train_encoded, sample_weight=sample_weights)
            except TypeError:
                print("[WARNING] Classifier does not support sample_weight directly. Using weighted resampling fallback.")
                X_resampled, y_resampled = self._resample_with_weights(
                    X_train_scaled,
                    y_train_encoded,
                    sample_weights
                )
                self.classifier.fit(X_resampled, y_resampled)
        else:
            self.classifier.fit(X_train_scaled, y_train_encoded)
        
        # Calculate training accuracy
        y_train_pred = self.classifier.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        
        # Store training history
        self.training_history.update({
            'n_authors': n_authors,
            'n_features': X_train.shape[1],
            'train_accuracy': train_accuracy,
            'n_blog_instances': n_blog,
            'n_tweet_train_instances': n_tweet_train,
            'domain_weighting_enabled': bool(sample_weights is not None),
            'domain_weights': domain_weights if sample_weights is not None else None,
        })
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Features used: {X_train.shape[1]}")
        
        return self
    
    def predict(self, X_test):
        """Predict authors for test data"""
        if self.classifier is None:
            raise ValueError("Model not trained! Call train() first.")
        
        X_test = X_test.fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_encoded = self.classifier.predict(X_test_scaled)
        predictions = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return predictions
    
    def evaluate(self, X_test, y_test, save_results=True):
        """Evaluate model on test data"""
        print("\n" + "="*70)
        print("MODEL EVALUATION ON TEST DATA (TWEETS)")
        print("="*70)
        
        # Predict
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate precision, recall, and F1-score (macro-averaged)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, 
            y_pred,
            average='macro',
            zero_division=0
        )
        
        # Calculate weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, 
            y_pred,
            average='weighted',
            zero_division=0
        )
        
        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        print(f"Macro-averaged Recall: {recall_macro*100:.2f}%")
        print(f"Macro-averaged F1-Score: {f1_macro*100:.2f}%")
        print(f"Macro-averaged Precision: {precision_macro*100:.2f}%")
        print(f"\nWeighted-averaged Recall: {recall_weighted*100:.2f}%")
        print(f"Weighted-averaged F1-Score: {f1_weighted*100:.2f}%")
        print(f"Weighted-averaged Precision: {precision_weighted*100:.2f}%")
        print(f"\nNumber of authors: {self.training_history['n_authors']}")
        print(f"Random baseline: {100/self.training_history['n_authors']:.2f}%")
        print(f"Improvement: {accuracy*100 - 100/self.training_history['n_authors']:.2f}%")
        
        # Classification report
        print(f"\n{'='*70}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*70}")
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.label_encoder.classes_)
        
        # Plot confusion matrix if not too large
        if len(self.label_encoder.classes_) <= 20:
            self._plot_confusion_matrix(cm, save_results)
        
        results = {
            'accuracy': accuracy,
            'recall_macro': recall_macro,
            'f1_score_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_weighted': recall_weighted,
            'f1_score_weighted': f1_weighted,
            'precision_weighted': precision_weighted,
            'n_authors': self.training_history['n_authors'],
            'random_baseline': 100/self.training_history['n_authors'],
            'improvement': accuracy * 100 - (100/self.training_history['n_authors']),
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        if save_results:
            self._save_results(results)
        
        return results
    
    def _plot_confusion_matrix(self, cm, save_fig=True):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix - Cross-Domain Authorship Attribution')
        plt.ylabel('True Author')
        plt.xlabel('Predicted Author')
        plt.tight_layout()
        
        if save_fig:
            ensure_parent_dir(Config.CONFUSION_MATRIX_IMG)
            plt.savefig(Config.CONFUSION_MATRIX_IMG, dpi=300, bbox_inches='tight')
            print(f"\n[SUCCESS] Confusion matrix saved to: {Config.CONFUSION_MATRIX_IMG}")
        
        plt.close()
    
    def _save_results(self, results):
        """Save evaluation results"""
        ensure_parent_dir(Config.RESULTS_FILE)
        with open(Config.RESULTS_FILE, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CROSS-DOMAIN AUTHORSHIP ATTRIBUTION RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Training Setup:\n")
            f.write(f"  - Blog data: ALL instances ({self.training_history['n_blog_instances']})\n")
            f.write(f"  - Tweet data: {(1-Config.TEST_SIZE)*100:.0f}% train / {Config.TEST_SIZE*100:.0f}% test\n")
            f.write(f"  - Tweet train instances: {self.training_history['n_tweet_train_instances']}\n")
            f.write(f"  - Tweet test instances: {self.training_history['n_tweet_test_instances']}\n\n")
            f.write("="*70 + "\n")
            f.write("PERFORMANCE METRICS (on Tweet Test Set)\n")
            f.write("="*70 + "\n")
            f.write(f"Accuracy: {results['accuracy']*100:.2f}%\n")
            f.write(f"Number of authors: {results['n_authors']}\n")
            f.write(f"Random baseline: {results['random_baseline']:.2f}%\n")
            f.write(f"Improvement over random: {results['improvement']:.2f}%\n\n")
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
        
        print(f"[SUCCESS] Results saved to: {Config.RESULTS_FILE}")
    
    def save_model(self, filepath=None):
        """Save trained model"""
        if filepath is None:
            filepath = Config.MODEL_FILE
        ensure_parent_dir(filepath)
        
        model_data = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'feature_extractor': self.feature_extractor,
            'selected_features': self.selected_features,
            'training_history': self.training_history,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[SUCCESS] Model saved to: {filepath}")


# ============================================================================
# ALL MODELS COMPARISON
# ============================================================================

def train_and_compare_all_models(X_train, y_train, X_test, y_test, 
                                  n_blog=None, n_tweet_train=None, 
                                  extractor=None, selected_features=None,
                                  domain_labels_train=None, domain_weights=None,
                                  save_artifacts=True):
    """
    Train and evaluate all available models, then compare results
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training labels
    X_test : DataFrame
        Test features
    y_test : Series
        Test labels
    n_blog : int
        Number of blog instances
    n_tweet_train : int
        Number of tweet train instances
    extractor : UnifiedFeatureExtractor
        Feature extractor instance
    selected_features : list
        List of selected feature names
        
    Returns:
    --------
    all_results : dict
        Dictionary containing results for each model
    comparison_df : DataFrame
        Comparison table of all models
    """
    print("\n" + "="*70)
    print("TRAINING AND COMPARING ALL MODELS")
    print("="*70)
    
    # Define models to test
    models_to_test = ['svm', 'logistic_regression', 'bagging', 'voting']
    if XGBOOST_AVAILABLE:
        models_to_test.insert(2, 'xgboost')  # Insert after logistic_regression
    
    all_results = {}
    all_models = {}
    
    # Train each model
    for i, model_type in enumerate(models_to_test, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(models_to_test)}] Training {model_type.upper().replace('_', ' ')} Model")
        print(f"{'='*70}")
        
        try:
            # Initialize model
            model = IntegratedAuthorshipModel(model_type=model_type)
            model.feature_extractor = extractor
            model.selected_features = selected_features
            
            # Train
            model.train(
                X_train,
                y_train,
                n_blog=n_blog,
                n_tweet_train=n_tweet_train,
                domain_labels=domain_labels_train,
                domain_weights=domain_weights
            )
            
            # Store test set size
            model.training_history['n_tweet_test_instances'] = X_test.shape[0]
            
            # Evaluate (don't save results yet)
            results = model.evaluate(X_test, y_test, save_results=False)
            
            # Store results
            all_results[model_type] = results
            all_models[model_type] = model
            
            print(f"\n {model_type.upper().replace('_', ' ')} complete!")
            print(f"  Accuracy: {results['accuracy']*100:.2f}%")
            print(f"  F1-Score (Macro): {results['f1_score_macro']*100:.2f}%")
            print(f"  Recall (Macro): {results['recall_macro']*100:.2f}%")
            
        except Exception as e:
            print(f"\n {model_type.upper().replace('_', ' ')} failed: {e}")
            all_results[model_type] = None
    
    # Create comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    
    comparison_data = []
    for model_type in models_to_test:
        if all_results.get(model_type):
            results = all_results[model_type]
            comparison_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'Accuracy (%)': f"{results['accuracy']*100:.2f}",
                'Precision (Macro) (%)': f"{results['precision_macro']*100:.2f}",
                'Recall (Macro) (%)': f"{results['recall_macro']*100:.2f}",
                'F1-Score (Macro) (%)': f"{results['f1_score_macro']*100:.2f}",
                'Precision (Weighted) (%)': f"{results['precision_weighted']*100:.2f}",
                'Recall (Weighted) (%)': f"{results['recall_weighted']*100:.2f}",
                'F1-Score (Weighted) (%)': f"{results['f1_score_weighted']*100:.2f}",
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best model by accuracy
    best_model_name = max(all_results.items(), 
                          key=lambda x: x[1]['accuracy'] if x[1] else 0)[0]
    best_results = all_results[best_model_name]
    best_model = all_models[best_model_name]
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    print(f"Accuracy: {best_results['accuracy']*100:.2f}%")
    print(f"Macro F1-Score: {best_results['f1_score_macro']*100:.2f}%")
    print(f"Macro Recall: {best_results['recall_macro']*100:.2f}%")
    
    if save_artifacts:
        # Save comprehensive results
        _save_all_models_results(all_results, comparison_df, best_model_name)
        
        # Save best model
        print(f"\nSaving best model ({best_model_name})...")
        best_model.save_model(Config.MODEL_FILE)
        
        # Save comparison plot
        _plot_model_comparison(comparison_df)
    
    return all_results, comparison_df, best_model_name, best_model


def _save_all_models_results(all_results, comparison_df, best_model_name):
    """Save comprehensive results for all models"""
    ensure_parent_dir(Config.RESULTS_FILE)
    with open(Config.RESULTS_FILE, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ALL MODELS COMPARISON - CROSS-DOMAIN AUTHORSHIP ATTRIBUTION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training Setup:\n")
        
        # Get info from first successful model
        first_result = next((r for r in all_results.values() if r is not None), None)
        if first_result:
            f.write(f"  - Blog data: ALL instances ({first_result.get('n_blog_instances', 'N/A')})\n")
            f.write(f"  - Tweet data: {(1-Config.TEST_SIZE)*100:.0f}% train / {Config.TEST_SIZE*100:.0f}% test\n\n")
        
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON TABLE\n")
        f.write("="*70 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"BEST MODEL: {best_model_name.upper().replace('_', ' ')}\n")
        f.write("="*70 + "\n")
        best_results = all_results[best_model_name]
        f.write(f"Accuracy: {best_results['accuracy']*100:.2f}%\n")
        f.write(f"Improvement over random: {best_results['improvement']:.2f}%\n\n")
        f.write("Macro-averaged Metrics:\n")
        f.write(f"  Precision: {best_results['precision_macro']*100:.2f}%\n")
        f.write(f"  Recall: {best_results['recall_macro']*100:.2f}%\n")
        f.write(f"  F1-Score: {best_results['f1_score_macro']*100:.2f}%\n\n")
        f.write("Weighted-averaged Metrics:\n")
        f.write(f"  Precision: {best_results['precision_weighted']*100:.2f}%\n")
        f.write(f"  Recall: {best_results['recall_weighted']*100:.2f}%\n")
        f.write(f"  F1-Score: {best_results['f1_score_weighted']*100:.2f}%\n\n")
        
        f.write("="*70 + "\n")
        f.write("DETAILED RESULTS FOR EACH MODEL\n")
        f.write("="*70 + "\n\n")
        
        for model_name, results in all_results.items():
            if results is None:
                continue
                
            f.write(f"\n{'-'*70}\n")
            f.write(f"{model_name.upper().replace('_', ' ')}\n")
            f.write(f"{'-'*70}\n\n")
            f.write(f"Accuracy: {results['accuracy']*100:.2f}%\n")
            f.write(f"Random baseline: {results['random_baseline']:.2f}%\n")
            f.write(f"Improvement: {results['improvement']:.2f}%\n\n")
            f.write("Macro-averaged Metrics:\n")
            f.write(f"  Precision: {results['precision_macro']*100:.2f}%\n")
            f.write(f"  Recall: {results['recall_macro']*100:.2f}%\n")
            f.write(f"  F1-Score: {results['f1_score_macro']*100:.2f}%\n\n")
            f.write("Weighted-averaged Metrics:\n")
            f.write(f"  Precision: {results['precision_weighted']*100:.2f}%\n")
            f.write(f"  Recall: {results['recall_weighted']*100:.2f}%\n")
            f.write(f"  F1-Score: {results['f1_score_weighted']*100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n")
    
    print(f"[SUCCESS] Comprehensive results saved to: {Config.RESULTS_FILE}")


def _plot_model_comparison(comparison_df):
    """Plot comparison of all models"""
    # Convert percentage strings to floats
    metrics = ['Accuracy (%)', 'Recall (Macro) (%)', 'F1-Score (Macro) (%)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [float(v) for v in comparison_df[metric]]
        models = comparison_df['Model']
        
        bars = ax.bar(range(len(models)), values, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(metric)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # Highlight best model
        best_idx = values.index(max(values))
        bars[best_idx].set_color('darkgreen')
        bars[best_idx].set_alpha(1.0)
    
    plt.tight_layout()
    comparison_img = Config.CONFUSION_MATRIX_IMG.replace('.png', '_comparison.png')
    ensure_parent_dir(comparison_img)
    plt.savefig(comparison_img, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Model comparison plot saved to: {comparison_img}")
    plt.close()
def _aggregate_cv_results(model_fold_results):
    """Aggregate stratified K-fold CV metrics (mean/std/min/max) for each model."""
    metric_keys = [
        'accuracy', 'precision_macro', 'recall_macro', 'f1_score_macro',
        'precision_weighted', 'recall_weighted', 'f1_score_weighted'
    ]
    
    aggregated = {}
    for model_name, fold_results in model_fold_results.items():
        if not fold_results:
            continue
        
        model_summary = {'n_folds': len(fold_results)}
        for key in metric_keys:
            values = [float(fr[key]) for fr in fold_results if key in fr]
            if not values:
                continue
            model_summary[key] = float(np.mean(values))
            model_summary[f'{key}_std'] = float(np.std(values))
            model_summary[f'{key}_min'] = float(np.min(values))
            model_summary[f'{key}_max'] = float(np.max(values))
        
        # Keep key metadata expected by downstream code
        n_authors_values = [fr.get('n_authors') for fr in fold_results if fr.get('n_authors') is not None]
        if n_authors_values:
            model_summary['n_authors'] = int(round(float(np.mean(n_authors_values))))
        
        aggregated[model_name] = model_summary
    
    return aggregated

def _build_cv_comparison_df(aggregated_results):
    rows = []
    for model_name, stats in aggregated_results.items():
        rows.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy Mean (%)': f"{stats.get('accuracy', 0) * 100:.2f}",
            'Accuracy Std (%)': f"{stats.get('accuracy_std', 0) * 100:.2f}",
            'F1 Mean (%)': f"{stats.get('f1_score_macro', 0) * 100:.2f}",
            'F1 Std (%)': f"{stats.get('f1_score_macro_std', 0) * 100:.2f}",
            'Recall Mean (%)': f"{stats.get('recall_macro', 0) * 100:.2f}",
            'Recall Std (%)': f"{stats.get('recall_macro_std', 0) * 100:.2f}",
            'Folds': stats.get('n_folds', 0),
        })
    return pd.DataFrame(rows)

def _save_cv_results(aggregated_results, best_model_name, effective_splits):
    """Save stratified K-fold CV summary to results file."""
    ensure_parent_dir(Config.RESULTS_FILE)
    with open(Config.RESULTS_FILE, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STRATIFIED K-FOLD RESULTS - CROSS-DOMAIN AUTHORSHIP\n")
        f.write("="*70 + "\n\n")
        f.write(f"CV setup:\n")
        f.write(f"  - Splits: {effective_splits}\n")
        f.write(f"  - Total folds: {effective_splits}\n")
        f.write(f"  - Stratification target: tweet author labels\n\n")
        f.write(f"Best model (by mean accuracy): {best_model_name.upper().replace('_', ' ')}\n\n")
        
        for model_name, stats in aggregated_results.items():
            f.write("-"*70 + "\n")
            f.write(f"{model_name.upper().replace('_', ' ')}\n")
            f.write("-"*70 + "\n")
            f.write(f"Folds: {stats.get('n_folds', 0)}\n")
            if 'accuracy' in stats:
                f.write(
                    f"Accuracy: {stats['accuracy']*100:.2f}% +/- {stats.get('accuracy_std', 0)*100:.2f}% "
                    f"[{stats.get('accuracy_min', 0)*100:.2f}%, {stats.get('accuracy_max', 0)*100:.2f}%]\n"
                )
            if 'precision_macro' in stats:
                f.write(
                    f"Macro Precision: {stats['precision_macro']*100:.2f}% +/- {stats.get('precision_macro_std', 0)*100:.2f}%\n"
                )
            if 'recall_macro' in stats:
                f.write(
                    f"Macro Recall: {stats['recall_macro']*100:.2f}% +/- {stats.get('recall_macro_std', 0)*100:.2f}%\n"
                )
            if 'f1_score_macro' in stats:
                f.write(
                    f"Macro F1: {stats['f1_score_macro']*100:.2f}% +/- {stats.get('f1_score_macro_std', 0)*100:.2f}%\n"
                )
            f.write("\n")
    print(f"[SUCCESS] Stratified K-fold results saved to: {Config.RESULTS_FILE}")

def run_stratified_kfold_cv(blog_texts, blog_labels, tweet_texts, tweet_labels):
    """Run stratified K-fold CV on tweet author labels."""
    print("\n" + "="*70)
    print("STRATIFIED K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    if len(tweet_texts) == 0 or len(tweet_labels) == 0:
        raise ValueError("No tweet groups available for cross-validation.")
    
    tweet_labels_arr = np.array(tweet_labels)
    label_counts = pd.Series(tweet_labels_arr).value_counts()
    min_samples_per_author = int(label_counts.min())
    requested_splits = int(Config.CV_N_SPLITS)
    effective_splits = min(requested_splits, min_samples_per_author)
    
    if effective_splits < 2:
        raise ValueError(
            f"Not enough tweet groups per author for K-fold CV. "
            f"Minimum groups per author is {min_samples_per_author}, need at least 2."
        )
    if effective_splits < requested_splits:
        print(
            f"[WARNING] Reducing CV splits from {requested_splits} to {effective_splits} "
            f"because some authors have limited tweet groups."
        )
    
    total_folds = effective_splits
    print(f"CV setup: {effective_splits} folds")
    print("Stratification target: tweet author labels")
    
    cv = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=Config.RANDOM_STATE
    )
    domain_weights = {'tweet': Config.TWEET_WEIGHT, 'blog': Config.BLOG_WEIGHT}
    
    model_fold_results = {}
    best_model_obj = None
    best_model_name = None
    best_model_fold_accuracy = -1.0
    
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(np.zeros(len(tweet_labels_arr)), tweet_labels_arr),
        start=1
    ):
        print(f"\n{'='*70}")
        print(f"CV FOLD {fold_idx}/{total_folds}")
        print(f"{'='*70}")
        
        tweet_texts_train = [tweet_texts[i] for i in train_idx]
        tweet_texts_test = [tweet_texts[i] for i in test_idx]
        tweet_labels_train = [tweet_labels[i] for i in train_idx]
        tweet_labels_test = [tweet_labels[i] for i in test_idx]
        
        fold_extractor = make_feature_extractor()
        
        if Config.USE_4GRAMS:
            fold_extractor.build_4gram_vocabulary(blog_texts, tweet_texts_train)
        else:
            fold_extractor.vocab_4grams = []
            fold_extractor.feature_names = fold_extractor._build_fixed_feature_names()
        
        # Feature extraction
        blog_features, blog_labels_series = fold_extractor.process_texts(blog_texts, blog_labels, 'blog')
        tweet_train_features, tweet_train_labels_series = fold_extractor.process_texts(
            tweet_texts_train, tweet_labels_train, 'tweet-train'
        )
        tweet_test_features, tweet_test_labels_series = fold_extractor.process_texts(
            tweet_texts_test, tweet_labels_test, 'tweet-test'
        )
        
        # Feature selection per fold
        if Config.USE_FEATURE_SELECTION:
            combined_train_features = pd.concat([blog_features, tweet_train_features], ignore_index=True)
            combined_train_labels = pd.concat([blog_labels_series, tweet_train_labels_series], ignore_index=True)
            selected_features, _ = select_features_mi(
                combined_train_features,
                combined_train_labels,
                ratio=Config.FEATURE_SELECTION_RATIO,
                random_state=Config.RANDOM_STATE
            )
            blog_features = blog_features[selected_features]
            tweet_train_features = tweet_train_features[selected_features]
            tweet_test_features = tweet_test_features[selected_features]
        else:
            selected_features = list(blog_features.columns)
        
        X_train = pd.concat([blog_features, tweet_train_features], axis=0, ignore_index=True)
        y_train = pd.concat([blog_labels_series, tweet_train_labels_series], axis=0, ignore_index=True)
        X_test = tweet_test_features
        y_test = tweet_test_labels_series
        domain_labels_train = pd.Series(
            ['blog'] * blog_features.shape[0] + ['tweet'] * tweet_train_features.shape[0]
        )
        
        if Config.MODEL_TYPE == 'all':
            fold_all_results, _, fold_best_model_name, fold_best_model = train_and_compare_all_models(
                X_train, y_train, X_test, y_test,
                n_blog=blog_features.shape[0],
                n_tweet_train=tweet_train_features.shape[0],
                extractor=fold_extractor,
                selected_features=selected_features,
                domain_labels_train=domain_labels_train,
                domain_weights=domain_weights,
                save_artifacts=False
            )
            
            for model_name, fold_result in fold_all_results.items():
                if fold_result is None:
                    continue
                model_fold_results.setdefault(model_name, []).append(fold_result)
            
            fold_best_accuracy = fold_all_results[fold_best_model_name]['accuracy']
            if fold_best_accuracy > best_model_fold_accuracy:
                best_model_fold_accuracy = fold_best_accuracy
                best_model_name = fold_best_model_name
                best_model_obj = fold_best_model
        else:
            model = IntegratedAuthorshipModel(model_type=Config.MODEL_TYPE)
            model.feature_extractor = fold_extractor
            model.selected_features = selected_features
            model.train(
                X_train,
                y_train,
                n_blog=blog_features.shape[0],
                n_tweet_train=tweet_train_features.shape[0],
                domain_labels=domain_labels_train,
                domain_weights=domain_weights
            )
            model.training_history['n_tweet_test_instances'] = X_test.shape[0]
            fold_result = model.evaluate(X_test, y_test, save_results=False)
            
            model_fold_results.setdefault(Config.MODEL_TYPE, []).append(fold_result)
            if fold_result['accuracy'] > best_model_fold_accuracy:
                best_model_fold_accuracy = fold_result['accuracy']
                best_model_name = Config.MODEL_TYPE
                best_model_obj = model
    
    aggregated_results = _aggregate_cv_results(model_fold_results)
    if not aggregated_results:
        raise RuntimeError("No valid CV results were produced.")
    
    cv_comparison_df = _build_cv_comparison_df(aggregated_results)
    best_model_name = max(
        aggregated_results.items(),
        key=lambda x: x[1].get('accuracy', 0)
    )[0]
    best_results = aggregated_results[best_model_name]
    
    print("\n" + "="*70)
    print("STRATIFIED K-FOLD CV SUMMARY")
    print("="*70)
    print(cv_comparison_df.to_string(index=False))
    print(f"\nBest model (mean accuracy): {best_model_name.upper().replace('_', ' ')}")
    print(
        f"Mean Accuracy: {best_results.get('accuracy', 0)*100:.2f}% +/- "
        f"{best_results.get('accuracy_std', 0)*100:.2f}%"
    )
    
    _save_cv_results(aggregated_results, best_model_name, effective_splits)
    if best_model_obj is not None:
        best_model_obj.save_model(Config.MODEL_FILE)
    
    if Config.MODEL_TYPE == 'all':
        return best_model_obj, best_results, aggregated_results, cv_comparison_df
    return best_model_obj, best_results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("="*70)
    print("CROSS-DOMAIN AUTHORSHIP ATTRIBUTION")
    if Config.USE_STRATIFIED_KFOLD:
        print("Training: ALL Blog Data + Tweet Train Folds")
        print("Testing: Stratified K-Fold on tweet authors")
    else:
        print("Training: ALL Blog Data + Tweet Train Data")
        print("Testing: Tweet Test Data")
    print("="*70)
    
    # Initialize feature extractor
    print("\n[1/7] Initializing feature extractor...")
    extractor = make_feature_extractor()
    
    # Load blog data 
    print("\n[2/7] Loading blog data...")
    try:
        print(f"  Loading from: {Config.BLOG_CSV}")
        blog_df = pd.read_csv(Config.BLOG_CSV)
        print(f"  Blog data: {blog_df.shape}")
        
        # Preprocess blog texts
        blog_texts = []
        blog_labels = []
        for idx, row in blog_df.iterrows():
            text = row[Config.TEXT_COLUMN]
            title = row.get(Config.BLOG_TITLE_COLUMN, "") if Config.USE_BLOG_TITLE else ""
            processed = extractor.preprocess_blog(text, title)
            if processed:
                blog_texts.append(processed)
                blog_labels.append(row[Config.AUTHOR_COLUMN])
        
        print(f"  Processed {len(blog_texts)} blog instances (using ALL)")
    except Exception as e:
        print(f"  [ERROR] Could not load blog data: {e}")
        return
    
    # Load tweet data 
    print("\n[3/7] Loading tweet data...")
    try:
        print(f"  Loading from: {Config.TWEET_CSV}")
        tweet_df = pd.read_csv(Config.TWEET_CSV)
        print(f"  Tweet data: {tweet_df.shape}")
        
        # Group tweets
        print(f"  Grouping tweets (group_size={Config.TWEET_GROUP_SIZE})...")
        grouped_tweets = extractor.group_tweets(tweet_df, Config.TWEET_GROUP_SIZE)
        print(f"  Created {len(grouped_tweets)} tweet groups")
        
        # Preprocess tweet groups
        tweet_texts = []
        tweet_labels = []
        for group in grouped_tweets:
            combined = ' '.join([extractor.preprocess_tweet(t) for t in group['tweets']])
            if combined.strip():
                tweet_texts.append(combined)
                tweet_labels.append(group['author'])
        
        print(f"  Processed {len(tweet_texts)} tweet groups")
    except Exception as e:
        print(f"  [ERROR] Could not load tweet data: {e}")
        return
    
    if Config.USE_STRATIFIED_KFOLD:
        return run_stratified_kfold_cv(
            blog_texts=blog_texts,
            blog_labels=blog_labels,
            tweet_texts=tweet_texts,
            tweet_labels=tweet_labels
        )
    
    # Split tweet data into train/test
    print(f"\n[4/7] Splitting tweet data ({(1-Config.TEST_SIZE)*100:.0f}% train / {Config.TEST_SIZE*100:.0f}% test)...")
    tweet_texts_train, tweet_texts_test, tweet_labels_train, tweet_labels_test = train_test_split(
        tweet_texts,
        tweet_labels,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=tweet_labels
    )
    
    print(f"  Tweet train: {len(tweet_texts_train)} instances")
    print(f"  Tweet test: {len(tweet_texts_test)} instances")
    
    # Build 4-gram vocabulary from blog + tweet train 
    if Config.USE_4GRAMS:
        print("\n[5/7] Building 4-gram vocabulary (from blog + tweet train)...")
        extractor.build_4gram_vocabulary(blog_texts, tweet_texts_train)
    else:
        print("\n[5/7] Skipping 4-grams (disabled)")
        extractor.vocab_4grams = []
        extractor.feature_names = extractor._build_fixed_feature_names()
    
    print(f"  Total fixed features: {len(extractor.feature_names)}")
    
    # Extract features
    print("\n[6/7] Extracting features...")
    
    # Blog features 
    blog_features, blog_labels_series = extractor.process_texts(blog_texts, blog_labels, 'blog')
    
    # Tweet train features
    tweet_train_features, tweet_train_labels_series = extractor.process_texts(
        tweet_texts_train, tweet_labels_train, 'tweet-train'
    )
    
    # Tweet test features
    tweet_test_features, tweet_test_labels_series = extractor.process_texts(
        tweet_texts_test, tweet_labels_test, 'tweet-test'
    )
    
    print(f"\n  Blog features: {blog_features.shape}")
    print(f"  Tweet train features: {tweet_train_features.shape}")
    print(f"  Tweet test features: {tweet_test_features.shape}")
    
    # Feature selection 
    if Config.USE_FEATURE_SELECTION:
        print("\n[7/7] Feature selection (on blog + tweet train)...")
        combined_train_features = pd.concat([blog_features, tweet_train_features], ignore_index=True)
        combined_train_labels = pd.concat([blog_labels_series, tweet_train_labels_series], ignore_index=True)
        
        selected_features, mi_scores = select_features_mi(
            combined_train_features,
            combined_train_labels,
            ratio=Config.FEATURE_SELECTION_RATIO,
            random_state=Config.RANDOM_STATE
        )
        
        # Apply selection to all datasets
        blog_features = blog_features[selected_features]
        tweet_train_features = tweet_train_features[selected_features]
        tweet_test_features = tweet_test_features[selected_features]
        
        print(f"  Selected {len(selected_features)} features")
    else:
        print("\n[7/7] Skipping feature selection")
        selected_features = list(blog_features.columns)
    
    # Combine blog + tweet train for training
    print("\n" + "="*70)
    print("COMBINING TRAINING DATA")
    print("="*70)
    
    X_train = pd.concat([blog_features, tweet_train_features], axis=0, ignore_index=True)
    y_train = pd.concat([blog_labels_series, tweet_train_labels_series], axis=0, ignore_index=True)
    domain_labels_train = pd.Series(
        ['blog'] * blog_features.shape[0] + ['tweet'] * tweet_train_features.shape[0]
    )
    X_test = tweet_test_features
    y_test = tweet_test_labels_series
    
    print(f"\nTraining set composition:")
    print(f"  - Blog (all): {blog_features.shape[0]} instances")
    print(f"  - Tweet (train): {tweet_train_features.shape[0]} instances")
    print(f"  - Total training: {X_train.shape[0]} instances")
    print(f"\nTest set:")
    print(f"  - Tweet (test): {X_test.shape[0]} instances")
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Authors: {y_train.nunique()}")
    
    # Show author distribution
    print(f"\nAuthor distribution in training set:")
    author_counts = y_train.value_counts()
    print(author_counts.head(10))
    if len(author_counts) > 10:
        print("...")
    
    # Verify no data leakage
    train_authors = set(y_train.unique())
    test_authors = set(y_test.unique())
    authors_only_in_test = test_authors - train_authors
    if authors_only_in_test:
        print(f"\n[WARNING] Found {len(authors_only_in_test)} authors in test set not in training:")
        print(f"  {list(authors_only_in_test)[:10]}")
    
    # Initialize and train model(s)
    if Config.MODEL_TYPE == 'all':
        # Train and compare all models
        print(f"\n[TRAINING] Running ALL models for comparison...")
        domain_weights = {'tweet': Config.TWEET_WEIGHT, 'blog': Config.BLOG_WEIGHT}
        all_results, comparison_df, best_model_name, model = train_and_compare_all_models(
            X_train, y_train, X_test, y_test,
            n_blog=blog_features.shape[0],
            n_tweet_train=tweet_train_features.shape[0],
            extractor=extractor,
            selected_features=selected_features,
            domain_labels_train=domain_labels_train,
            domain_weights=domain_weights
        )
        
        results = all_results[best_model_name]
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY - ALL MODELS COMPARISON")
        print("="*70)
        print(f"\nTraining Setup:")
        print(f"  - Blog instances (all): {blog_features.shape[0]}")
        print(f"  - Tweet train instances: {tweet_train_features.shape[0]}")
        print(f"  - Total training instances: {X_train.shape[0]}")
        print(f"  - Tweet test instances: {X_test.shape[0]}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"\nModels Tested: {len(all_results)}")
        print(f"Best Model: {best_model_name.upper().replace('_', ' ')}")
        print(f"\nBest Model Results (Tweet Test Set):")
        print(f"  - Test accuracy: {results['accuracy']*100:.2f}%")
        print(f"  - Macro F1-score: {results['f1_score_macro']*100:.2f}%")
        print(f"  - Macro Recall: {results['recall_macro']*100:.2f}%")
        print(f"  - Number of authors: {results['n_authors']}")
        
        return model, results, all_results, comparison_df
    else:
        # Train single model
        print(f"\n[TRAINING] Initializing {Config.MODEL_TYPE.upper().replace('_', ' ')} model...")
        model = IntegratedAuthorshipModel(model_type=Config.MODEL_TYPE)
        model.feature_extractor = extractor
        model.selected_features = selected_features
        
        model.train(
            X_train, 
            y_train,
            n_blog=blog_features.shape[0],
            n_tweet_train=tweet_train_features.shape[0],
            domain_labels=domain_labels_train,
            domain_weights={'tweet': Config.TWEET_WEIGHT, 'blog': Config.BLOG_WEIGHT}
        )
        
        # Store test set size
        model.training_history['n_tweet_test_instances'] = X_test.shape[0]
        
        # Evaluate
        results = model.evaluate(X_test, y_test, save_results=True)
        
        # Save model
        model.save_model()
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\nTraining Setup:")
        print(f"  - Blog instances (all): {blog_features.shape[0]}")
        print(f"  - Tweet train instances: {tweet_train_features.shape[0]}")
        print(f"  - Total training instances: {X_train.shape[0]}")
        print(f"  - Tweet test instances: {X_test.shape[0]}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"\nResults (Tweet Test Set):")
        print(f"  - Test accuracy: {results['accuracy']*100:.2f}%")
        print(f"  - Macro F1-score: {results['f1_score_macro']*100:.2f}%")
        print(f"  - Macro Recall: {results['recall_macro']*100:.2f}%")
        print(f"  - Number of authors: {results['n_authors']}")
        
        return model, results


if __name__ == "__main__":
    result = main()
    if len(result) == 4:
        # 'all' model type returns 4 values
        model, results, all_results, comparison_df = result
    else:
        # Single model type returns 2 values
        model, results = result

