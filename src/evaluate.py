"""
Combined Script: Data Selection + Multiple Model Training Runs + Averaged Evaluation
Selects random authors, trains models multiple times with different random seeds, 
and reports averaged results with standard deviations.
"""

import pandas as pd
import numpy as np
import random
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

from utils.paths import data_path, project_path, results_path
from utils.config_loader import apply_cli_overrides, load_experiment_config


# ============================================================================
# CONFIGURATION 
# ============================================================================

CONFIG = {
    # Data Selection Parameters
    'num_authors': 30,
    'tweet_word_limit': 3000,
    'blog_word_limit': 3000,
    
    # Input file paths for data selection
    'source_tweet_file': str(data_path('Tweet30.csv')),
    'source_blog_file': str(data_path('Blog30.csv')),
    
    # Output directory for selected data
    'selected_data_dir': str(project_path('training_data')),
    
    # Training Parameters
    'num_runs': 5,  # Number of training runs
    'base_seed': 42,  # Base seed for training runs
    'selection_seed': None,  # Random author selection
    'skip_selection': False,
    
    # Model Comparison Configuration
    'compare_unified': True,  # Unified Blog+Tweet model (mixed_train.py)
    'compare_tweet_only': True,  # Tweet-only model with tweet-specific features (separate.py)
    'compare_neural_network': False,
    'compare_distilbert': False,
    
    # Model configuration for each approach
    'unified_model_type': 'all',  # For unified: 'svm', 'logistic_regression', 'xgboost', 'bagging', 'voting', 'all'
    'tweet_only_model_type': 'voting',  # For tweet-only approach
    'neural_network_model_type': 'all',  # For neural network approach: 'mlp', 'bilstm', 'all'
    
    # Training parameters (shared across approaches)
    'test_size': 0.2,
    'tweet_group_size': 10,
    'use_blog_title': True,
    'use_semantic_clustering': False,
    'semantic_cluster_k': 10,
    'semantic_model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
    'use_domain_weighting': False,
    'tweet_weight': 3.0,
    'blog_weight': 1.0,
    'use_4grams': True,
    'top_4grams': 1000,
    'use_feature_selection': True,
    'feature_selection_ratio': 0.05,
    
    # Tweet model feature selection (Unnecessary)
    'tweet_use_feature_selection': False,  # Disable feature selection for tweet models 
    
    # Output files
    'results_dir': str(results_path('averaged')),
    'averaged_results_file': 'averaged_evaluation_results.txt',
    'detailed_results_file': 'detailed_run_results.json',
    'comparison_summary_file': 'model_comparison_summary.csv',

    # DistilBERT settings
    'distilbert_model_name': 'distilbert-base-uncased',
    'distilbert_max_length': 256,
    'distilbert_batch_size': 8,
    'distilbert_learning_rate': 2e-5,
    'distilbert_num_epochs': 5,
    'distilbert_use_blog_data': True,
    'distilbert_use_tweet_data': True,
}

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


class MultiRunTrainer:
    """Handles multiple training runs and result averaging for different model approaches"""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        self.all_runs_results = []
        
    def run_single_training(self, run_number, seed):
        """Run a single training iteration with specified seed, comparing all enabled approaches"""
        print("\n" + "="*80)
        print(f"TRAINING RUN {run_number}/{self.config['num_runs']} (seed={seed})")
        print("="*80)
        
        run_results = {
            'run': run_number,
            'seed': seed,
            'unified_results': None,
            'tweet_only_results': None,
            'neural_network_results': None,
            'distilbert_results': None,
        }
        
        # [1] Train Unified Blog+Tweet Model
        if self.config['compare_unified']:
            print("\n" + "-"*80)
            print("[APPROACH 1] UNIFIED BLOG+TWEET MODEL")
            print("-"*80)
            try:
                unified_result = self._train_unified_model(run_number, seed)
                run_results['unified_results'] = unified_result
                
                # Get best model's accuracy
                if isinstance(unified_result, dict) and len(unified_result) > 0:
                    best_acc = max(r.get('accuracy', 0) for r in unified_result.values() if r)
                    print(f"[OK] Unified model complete - Best Accuracy: {best_acc*100:.2f}%")
            except Exception as e:
                print(f"[ERROR] Unified model failed: {e}")
                import traceback
                traceback.print_exc()
        
        # [2] Train Tweet-Only Model
        if self.config['compare_tweet_only']:
            print("\n" + "-"*80)
            print("[APPROACH 2] TWEET-ONLY MODEL")
            print("-"*80)
            try:
                separate_results = self._train_separate_models(run_number, seed)
                
                if self.config['compare_tweet_only']:
                    run_results['tweet_only_results'] = separate_results['tweet_results']
                    if separate_results['tweet_results']:
                        best_tweet_acc = max(r.get('accuracy', 0) for r in separate_results['tweet_results'].values() if r)
                        print(f"[OK] Tweet-only model complete - Best Accuracy: {best_tweet_acc*100:.2f}%")
                
            except Exception as e:
                print(f"[ERROR] Separate models failed: {e}")
                import traceback
                traceback.print_exc()

        # [3] Train Neural Network Model
        if self.config.get('compare_neural_network', False):
            print("\n" + "-"*80)
            print("[APPROACH 3] NEURAL NETWORK MODEL")
            print("-"*80)
            try:
                nn_results = self._train_neural_network_model(run_number, seed)
                run_results['neural_network_results'] = nn_results
                if nn_results:
                    best_nn_acc = max(r.get('accuracy', 0) for r in nn_results.values() if r)
                    print(f"[OK] Neural network complete - Best Accuracy: {best_nn_acc*100:.2f}%")
            except Exception as e:
                print(f"[ERROR] Neural network model failed: {e}")
                import traceback
                traceback.print_exc()

        # [4] Train DistilBERT Model
        if self.config.get('compare_distilbert', False):
            print("\n" + "-"*80)
            print("[APPROACH 4] DISTILBERT MODEL")
            print("-"*80)
            try:
                distilbert_results = self._train_distilbert_model(run_number, seed)
                run_results['distilbert_results'] = distilbert_results
                if distilbert_results:
                    best_distilbert_acc = max(r.get('accuracy', 0) for r in distilbert_results.values() if r)
                    print(f"[OK] DistilBERT complete - Best Accuracy: {best_distilbert_acc*100:.2f}%")
            except Exception as e:
                print(f"[ERROR] DistilBERT model failed: {e}")
                import traceback
                traceback.print_exc()
        
        self.all_runs_results.append(run_results)
        return run_results
    
    def _train_unified_model(self, run_number, seed):
        """Train unified blog+tweet model using mixed_train.py"""
        from mixed_train import main as train_model, Config as TrainConfig
        
        # Update training config
        TrainConfig.BLOG_CSV = str(Path(self.config['selected_data_dir']) / 'selected_blogs.csv')
        TrainConfig.TWEET_CSV = str(Path(self.config['selected_data_dir']) / 'selected_tweets.csv')
        TrainConfig.TEST_SIZE = self.config['test_size']
        TrainConfig.RANDOM_STATE = seed
        TrainConfig.MODEL_TYPE = self.config['unified_model_type']
        TrainConfig.TWEET_GROUP_SIZE = self.config['tweet_group_size']
        TrainConfig.USE_BLOG_TITLE = self.config['use_blog_title']
        TrainConfig.USE_SEMANTIC_CLUSTERING = self.config['use_semantic_clustering']
        TrainConfig.SEMANTIC_CLUSTER_K = self.config['semantic_cluster_k']
        TrainConfig.SEMANTIC_MODEL_NAME = self.config['semantic_model_name']
        TrainConfig.USE_DOMAIN_WEIGHTING = self.config['use_domain_weighting']
        TrainConfig.TWEET_WEIGHT = self.config['tweet_weight']
        TrainConfig.BLOG_WEIGHT = self.config['blog_weight']
        TrainConfig.USE_4GRAMS = self.config['use_4grams']
        TrainConfig.TOP_4GRAMS = self.config['top_4grams']
        TrainConfig.USE_FEATURE_SELECTION = self.config['use_feature_selection']
        TrainConfig.FEATURE_SELECTION_RATIO = self.config['feature_selection_ratio']
        
        # Update output files
        TrainConfig.MODEL_FILE = str(self.results_dir / f'run_{run_number}_unified_model.pkl')
        TrainConfig.RESULTS_FILE = str(self.results_dir / f'run_{run_number}_unified_results.txt')
        TrainConfig.CONFUSION_MATRIX_IMG = str(self.results_dir / f'run_{run_number}_unified_cm.png')
        
        # Run training
        result = train_model()
        
        # Extract results
        if len(result) == 4:
            model, results, all_results, comparison_df = result
            return all_results  # Return all model results for averaging
        else:
            model, results = result
            return {self.config['unified_model_type']: results}
    
    def _train_separate_models(self, run_number, seed):
        """Train tweet-only model using tweet_only.py"""
        # If file doesn't exist
        try:
            from tweet_only import Config as TweetConfig
        except ImportError:
            print("Warning: tweet_only.py not found. Skipping tweet-only comparison.")
            return {'tweet_results': {}}
        
        # Configure paths
        TweetConfig.TWEET_CSV = str(Path(self.config['selected_data_dir']) / 'selected_tweets.csv')
        TweetConfig.TEST_SIZE = self.config['test_size']
        TweetConfig.RANDOM_STATE = seed
        TweetConfig.TWEET_MODEL_TYPE = self.config['tweet_only_model_type']
        TweetConfig.TWEET_GROUP_SIZE = self.config['tweet_group_size']
        
        # Disable feature selection for tweet models
        TweetConfig.TWEET_USE_FEATURE_SELECTION = self.config.get('tweet_use_feature_selection', False)
        
        # Output files
        results_file = self.results_dir / f'run_{run_number}_tweet_only_results.txt'
        TweetConfig.RESULTS_FILE = str(results_file)
        TweetConfig.TWEET_MODEL_FILE = str(self.results_dir / f'run_{run_number}_tweet_model.pkl')
        TweetConfig.CONFUSION_MATRIX_IMG = str(self.results_dir / f'run_{run_number}_tweet_cm.png')
        
        # Train tweet-only model
        from tweet_only import main as train_tweet_only
        tweet_model, best_results, all_results = train_tweet_only()
        
        # Parse results file to extract all model results
        tweet_results = self._parse_tweet_only_results(results_file, all_results)
        
        return {
            'tweet_results': tweet_results,
        }

    def _train_neural_network_model(self, run_number, seed):
        """Train neural-network model using neural_network.py."""
        from neural_network import main as train_neural_network, Config as NNConfig

        selected_dir = Path(self.config['selected_data_dir'])
        NNConfig.BLOG_CSV = str(selected_dir / 'selected_blogs.csv')
        NNConfig.TWEET_CSV = str(selected_dir / 'selected_tweets.csv')
        NNConfig.TEST_SIZE = self.config['test_size']
        NNConfig.RANDOM_STATE = seed
        NNConfig.MODEL_TYPE = self.config.get('neural_network_model_type', 'all')
        NNConfig.TWEET_GROUP_SIZE = self.config['tweet_group_size']
        NNConfig.USE_BLOG_TITLE = self.config['use_blog_title']
        NNConfig.USE_SEMANTIC_CLUSTERING = self.config['use_semantic_clustering']
        NNConfig.SEMANTIC_CLUSTER_K = self.config['semantic_cluster_k']
        NNConfig.SEMANTIC_MODEL_NAME = self.config['semantic_model_name']
        NNConfig.USE_DOMAIN_WEIGHTING = self.config['use_domain_weighting']
        NNConfig.TWEET_WEIGHT = self.config['tweet_weight']
        NNConfig.BLOG_WEIGHT = self.config['blog_weight']
        NNConfig.USE_4GRAMS = self.config['use_4grams']
        NNConfig.TOP_4GRAMS = self.config['top_4grams']
        NNConfig.USE_FEATURE_SELECTION = self.config['use_feature_selection']
        NNConfig.FEATURE_SELECTION_RATIO = self.config['feature_selection_ratio']
        NNConfig.RESULTS_FILE = str(self.results_dir / f'run_{run_number}_neural_network_results.txt')
        NNConfig.BEST_MODEL_FILE = str(self.results_dir / f'run_{run_number}_neural_network_model.pkl')

        _, _, all_results = train_neural_network()
        return all_results or {}

    def _train_distilbert_model(self, run_number, seed):
        """Train DistilBERT model using distilbert.py."""
        from distilbert import main as train_distilbert, Config as DistilConfig

        selected_dir = Path(self.config['selected_data_dir'])
        DistilConfig.BLOG_CSV = str(selected_dir / 'selected_blogs.csv')
        DistilConfig.TWEET_CSV = str(selected_dir / 'selected_tweets.csv')
        DistilConfig.TEST_SIZE = self.config['test_size']
        DistilConfig.RANDOM_STATE = seed
        DistilConfig.TWEET_GROUP_SIZE = self.config['tweet_group_size']
        DistilConfig.MODEL_NAME = self.config.get('distilbert_model_name', DistilConfig.MODEL_NAME)
        DistilConfig.MAX_LENGTH = self.config.get('distilbert_max_length', DistilConfig.MAX_LENGTH)
        DistilConfig.BATCH_SIZE = self.config.get('distilbert_batch_size', DistilConfig.BATCH_SIZE)
        DistilConfig.LEARNING_RATE = self.config.get('distilbert_learning_rate', DistilConfig.LEARNING_RATE)
        DistilConfig.NUM_EPOCHS = self.config.get('distilbert_num_epochs', DistilConfig.NUM_EPOCHS)
        DistilConfig.USE_BLOG_DATA = self.config.get('distilbert_use_blog_data', DistilConfig.USE_BLOG_DATA)
        DistilConfig.USE_TWEET_DATA = self.config.get('distilbert_use_tweet_data', DistilConfig.USE_TWEET_DATA)
        DistilConfig.MODEL_SAVE_DIR = str(self.results_dir / f'run_{run_number}_distilbert_model')
        DistilConfig.LABEL_ENCODER_FILE = str(self.results_dir / f'run_{run_number}_distilbert_label_encoder.pkl')
        DistilConfig.RESULTS_FILE = str(self.results_dir / f'run_{run_number}_distilbert_results.txt')
        DistilConfig.CONFUSION_MATRIX_IMG = str(self.results_dir / f'run_{run_number}_distilbert_cm.png')

        _, results = train_distilbert()
        return {'distilbert': results} if results else {}
    
    def _parse_tweet_only_results(self, results_file, all_results):
        """Parse the results from tweet_only.py"""
        tweet_results = {}
        
        # Use the all_results directly since tweet_only.py returns them
        if all_results:
            for model_name, results in all_results.items():
                if results:
                    tweet_results[model_name] = {
                        'accuracy': results['accuracy'],
                        'f1_score_macro': results['f1_score_macro'],
                        'recall_macro': results['recall_macro'],
                        'precision_macro': results.get('precision_macro', 0),
                    }
        
        # Fallback: Parse from file if direct results not available
        if not tweet_results and results_file.exists():
            with open(results_file, 'r') as f:
                content = f.read()
            
            import re
            # Look for best model results
            best_section = re.search(r'Best Model:\s+(\w+).*?Accuracy:\s+([\d.]+)%.*?Macro F1:\s+([\d.]+)%.*?Macro Recall:\s+([\d.]+)%',
                                    content, re.DOTALL)
            if best_section:
                model_name = best_section.group(1).lower().replace(' ', '_')
                tweet_results[model_name] = {
                    'accuracy': float(best_section.group(2)) / 100,
                    'f1_score_macro': float(best_section.group(3)) / 100,
                    'recall_macro': float(best_section.group(4)) / 100,
                }
        
        return tweet_results
    
    def _parse_separate_results(self, results_file):
        """Parse the results file from tweet_only.py to extract metrics"""
        tweet_results = {}
        
        if not results_file.exists():
            return tweet_results
        
        with open(results_file, 'r') as f:
            content = f.read()
        
        # Parse tweet model results
        import re
        
        # Look for "TWEET MODEL" section
        tweet_section = re.search(r'2\.\s+TWEET MODEL.*?Test Accuracy:\s+([\d.]+)%.*?Macro F1:\s+([\d.]+)%.*?Macro Recall:\s+([\d.]+)%', 
                                  content, re.DOTALL)
        if tweet_section:
            tweet_results['best'] = {
                'accuracy': float(tweet_section.group(1)) / 100,
                'f1_score_macro': float(tweet_section.group(2)) / 100,
                'recall_macro': float(tweet_section.group(3)) / 100,
            }
        
        return tweet_results
    
    
    def run_all_training(self):
        """Execute all training runs"""
        print("\n" + "="*80)
        print("MULTI-RUN TRAINING WITH MODEL COMPARISON")
        print("="*80)
        print(f"Number of runs: {self.config['num_runs']}")
        print(f"Base seed: {self.config['base_seed']}")
        print(f"\nApproaches to compare:")
        if self.config['compare_unified']:
            print(f"  [OK] Unified Blog+Tweet (model: {self.config['unified_model_type']})")
        if self.config['compare_tweet_only']:
            print(f"  [OK] Tweet-Only (model: {self.config['tweet_only_model_type']})")
        if self.config.get('compare_neural_network', False):
            print(f"  [OK] Neural Network (model: {self.config['neural_network_model_type']})")
        if self.config.get('compare_distilbert', False):
            print(f"  [OK] DistilBERT (model: {self.config['distilbert_model_name']})")
        print(f"\nData directory: {self.config['selected_data_dir']}")
        print(f"Results directory: {self.results_dir}")
        print("="*80)
        
        for run_num in range(1, self.config['num_runs'] + 1):
            # Use different seed for each run
            seed = self.config['base_seed'] + run_num - 1
            
            try:
                run_result = self.run_single_training(run_num, seed)
                
                print(f"\n[OK] Run {run_num} completed successfully")
                
            except Exception as e:
                print(f"\n[ERROR] Run {run_num} failed with error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not self.all_runs_results:
            raise RuntimeError("All training runs failed!")
        
        print(f"\n[OK] Completed {len(self.all_runs_results)}/{self.config['num_runs']} runs successfully")
    
    def compute_averaged_results(self):
        """Compute mean and std of metrics across all runs for all approaches"""
        print("\n" + "="*80)
        print("COMPUTING AVERAGED RESULTS")
        print("="*80)
        
        averaged_results = {}
        
        if self.config['compare_unified']:
            print("Processing Unified Blog+Tweet results...")
            averaged_results['unified'] = self._average_approach_results('unified_results')
        
        if self.config['compare_tweet_only']:
            print("Processing Tweet-Only results...")
            averaged_results['tweet_only'] = self._average_approach_results('tweet_only_results')

        if self.config.get('compare_neural_network', False):
            print("Processing Neural Network results...")
            averaged_results['neural_network'] = self._average_approach_results('neural_network_results')

        if self.config.get('compare_distilbert', False):
            print("Processing DistilBERT results...")
            averaged_results['distilbert'] = self._average_approach_results('distilbert_results')
        
        return averaged_results
    
    def _average_approach_results(self, approach_key):
        """Average results for a specific approach across all runs"""
        model_metrics = {}
        
        for run in self.all_runs_results:
            approach_results = run.get(approach_key)
            
            if not approach_results or not isinstance(approach_results, dict):
                continue
            
            for model_name, results in approach_results.items():
                if not results or not isinstance(results, dict):
                    continue
                
                if model_name not in model_metrics:
                    model_metrics[model_name] = {
                        'accuracy': [],
                        'precision_macro': [],
                        'recall_macro': [],
                        'f1_score_macro': [],
                        'precision_weighted': [],
                        'recall_weighted': [],
                        'f1_score_weighted': [],
                    }
                
                for metric in model_metrics[model_name].keys():
                    if metric in results:
                        model_metrics[model_name][metric].append(results[metric])
        
        # Compute mean and std for each model
        averaged_by_model = {}
        for model_name, metrics in model_metrics.items():
            averaged_by_model[model_name] = {}
            for metric, values in metrics.items():
                if values:
                    averaged_by_model[model_name][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }
        
        return averaged_by_model
    
    def save_results(self, averaged_results):
        """Save averaged results to files"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save detailed JSON results
        detailed_file = self.results_dir / self.config['detailed_results_file']
        detailed_data = {
            'config': {k: v for k, v in self.config.items() if not callable(v)},
            'num_successful_runs': len(self.all_runs_results),
            'timestamp': datetime.now().isoformat(),
            'averaged_results': self._serialize_results(averaged_results)
        }
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        print(f"[OK] Saved detailed results to: {detailed_file}")
        
        # Save summary
        summary_file = self.results_dir / self.config['averaged_results_file']
        with open(summary_file, 'w') as f:
            self._write_summary_report(f, averaged_results)
        print(f"[OK] Saved summary report to: {summary_file}")
        
        # Save comparison CSV
        self._save_comparison_csv(averaged_results)
    
    def _serialize_results(self, results):
        """Convert numpy values to Python types for JSON serialization"""
        serialized = {}
        for approach_name, models in results.items():
            serialized[approach_name] = {}
            for model, metrics in models.items():
                serialized[approach_name][model] = {}
                for metric, stats in metrics.items():
                    serialized[approach_name][model][metric] = {
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'min': float(stats['min']),
                        'max': float(stats['max']),
                    }
        return serialized
    
    def _write_summary_report(self, file, averaged_results):
        """Write summary report"""
        def write(text):
            print(text)
            file.write(text + '\n')
        
        write("="*80)
        write("MODEL COMPARISON - AVERAGED EVALUATION RESULTS")
        write("="*80)
        write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write(f"\nConfiguration:")
        write(f"  Number of runs: {len(self.all_runs_results)}/{self.config['num_runs']}")
        write(f"  Base seed: {self.config['base_seed']}")
        write(f"  Number of authors: {self.config['num_authors']}")
        write(f"  Tweet word limit: {self.config['tweet_word_limit']}")
        write(f"  Blog word limit: {self.config['blog_word_limit']}")
        write(f"  Test size: {self.config['test_size']}")
        
        write("\n" + "="*80)
        write("RESULTS BY APPROACH")
        write("="*80)
        
        all_approaches = []
        
        # [1] Unified Blog+Tweet
        if 'unified' in averaged_results and averaged_results['unified']:
            write("\n[APPROACH 1] UNIFIED BLOG+TWEET MODEL")
            write("-" * 80)
            write("Uses unified feature extraction for both blog and tweet data")
            write("")
            
            best_unified = self._get_best_model(averaged_results['unified'])
            if best_unified:
                model_name, metrics = best_unified
                write(f"Best Model: {model_name.upper().replace('_', ' ')}")
                self._write_metrics(write, metrics)
                all_approaches.append(('Unified Blog+Tweet', model_name, metrics['accuracy']['mean']))
            
            # Show all models if multiple
            if len(averaged_results['unified']) > 1:
                write("\nAll Models:")
                for model_name, metrics in sorted(averaged_results['unified'].items(),
                                                 key=lambda x: x[1].get('accuracy', {}).get('mean', 0),
                                                 reverse=True):
                    write(f"  {model_name.replace('_', ' ').title():20s}: "
                         f"Acc={metrics['accuracy']['mean']*100:.2f}% +/- {metrics['accuracy']['std']*100:.2f}%")
        
        # [2] Tweet-Only
        if 'tweet_only' in averaged_results and averaged_results['tweet_only']:
            write("\n[APPROACH 2] TWEET-ONLY MODEL")
            write("-" * 80)
            write("Uses tweet-specific features (hashtags, mentions, emojis, etc.)")
            write("")
            
            best_tweet = self._get_best_model(averaged_results['tweet_only'])
            if best_tweet:
                model_name, metrics = best_tweet
                write(f"Best Model: {model_name.upper().replace('_', ' ')}")
                self._write_metrics(write, metrics)
                all_approaches.append(('Tweet-Only', model_name, metrics['accuracy']['mean']))
            
            # Show all models if multiple
            if len(averaged_results['tweet_only']) > 1:
                write("\nAll Models:")
                for model_name, metrics in sorted(averaged_results['tweet_only'].items(),
                                                 key=lambda x: x[1].get('accuracy', {}).get('mean', 0),
                                                 reverse=True):
                    write(f"  {model_name.replace('_', ' ').title():20s}: "
                         f"Acc={metrics['accuracy']['mean']*100:.2f}% +/- {metrics['accuracy']['std']*100:.2f}%")

        for approach_key, approach_label, description, approach_number in [
            ('neural_network', 'Neural Network', 'Uses MLP/BiLSTM models on shared feature vectors', 3),
            ('distilbert', 'DistilBERT', 'Uses a transformer text classifier', 4),
        ]:
            if approach_key in averaged_results and averaged_results[approach_key]:
                write(f"\n[APPROACH {approach_number}] {approach_label.upper()} MODEL")
                write("-" * 80)
                write(description)
                write("")

                best_model = self._get_best_model(averaged_results[approach_key])
                if best_model:
                    model_name, metrics = best_model
                    write(f"Best Model: {model_name.upper().replace('_', ' ')}")
                    self._write_metrics(write, metrics)
                    all_approaches.append((approach_label, model_name, metrics['accuracy']['mean']))

                if len(averaged_results[approach_key]) > 1:
                    write("\nAll Models:")
                    for model_name, metrics in sorted(
                        averaged_results[approach_key].items(),
                        key=lambda x: x[1].get('accuracy', {}).get('mean', 0),
                        reverse=True
                    ):
                        write(f"  {model_name.replace('_', ' ').title():20s}: "
                              f"Acc={metrics['accuracy']['mean']*100:.2f}% +/- {metrics['accuracy']['std']*100:.2f}%")
        
        # Overall winner
        if all_approaches:
            write("\n" + "="*80)
            write("OVERALL BEST APPROACH")
            write("="*80)
            
            winner = max(all_approaches, key=lambda x: x[2])
            approach_name, model_name, accuracy = winner
            write(f"\nWinner: {approach_name}")
            write(f"  Model: {model_name.upper().replace('_', ' ')}")
            write(f"  Accuracy: {accuracy*100:.2f}%")
            
            # Find full metrics for winner
            for approach_key in averaged_results:
                if any(approach_key in name.lower() for name in [approach_name.lower().replace(' ', '_')]):
                    if model_name in averaged_results[approach_key]:
                        write("")
                        self._write_metrics(write, averaged_results[approach_key][model_name])
                        break
        
        # Detailed results for all models from both approaches
        write("\n" + "="*80)
        write("DETAILED RESULTS - ALL MODELS")
        write("="*80)
        
        # [1] All Unified Models
        if 'unified' in averaged_results and averaged_results['unified']:
            write("\n[APPROACH 1] UNIFIED BLOG+TWEET MODEL - ALL MODELS")
            write("-" * 80)
            
            for rank, (model_name, metrics) in enumerate(
                sorted(averaged_results['unified'].items(),
                      key=lambda x: x[1].get('accuracy', {}).get('mean', 0),
                      reverse=True), 1
            ):
                write(f"\n{rank}. {model_name.upper().replace('_', ' ')}")
                self._write_metrics(write, metrics)
        
        # [2] All Tweet-Only Models
        if 'tweet_only' in averaged_results and averaged_results['tweet_only']:
            write("\n[APPROACH 2] TWEET-ONLY MODEL - ALL MODELS")
            write("-" * 80)
            
            for rank, (model_name, metrics) in enumerate(
                sorted(averaged_results['tweet_only'].items(),
                      key=lambda x: x[1].get('accuracy', {}).get('mean', 0),
                      reverse=True), 1
            ):
                write(f"\n{rank}. {model_name.upper().replace('_', ' ')}")
                self._write_metrics(write, metrics)

        for approach_key, approach_label, approach_number in [
            ('neural_network', 'NEURAL NETWORK', 3),
            ('distilbert', 'DISTILBERT', 4),
        ]:
            if approach_key in averaged_results and averaged_results[approach_key]:
                write(f"\n[APPROACH {approach_number}] {approach_label} MODEL - ALL MODELS")
                write("-" * 80)

                for rank, (model_name, metrics) in enumerate(
                    sorted(
                        averaged_results[approach_key].items(),
                        key=lambda x: x[1].get('accuracy', {}).get('mean', 0),
                        reverse=True
                    ),
                    1
                ):
                    write(f"\n{rank}. {model_name.upper().replace('_', ' ')}")
                    self._write_metrics(write, metrics)
        
        write("\n" + "="*80)
    
    def _get_best_model(self, models_dict):
        """Find model with highest accuracy"""
        if not models_dict:
            return None
        
        best = max(models_dict.items(),
                  key=lambda x: x[1].get('accuracy', {}).get('mean', 0))
        return best
    
    def _write_metrics(self, write_fn, metrics):
        """Write formatted metrics"""
        for metric_name in ['accuracy', 'f1_score_macro', 'recall_macro', 'precision_macro']:
            if metric_name in metrics:
                stats = metrics[metric_name]
                write_fn(f"  {metric_name.replace('_', ' ').title()}: "
                        f"{stats['mean']*100:6.2f}% +/- {stats['std']*100:5.2f}%  "
                        f"[{stats['min']*100:.2f}% - {stats['max']*100:.2f}%]")
    
    def _save_comparison_csv(self, averaged_results):
        """Save model comparison as CSV"""
        comparison_file = self.results_dir / self.config['comparison_summary_file']
        
        rows = []
        
        for approach_name, approach_label in [
            ('unified', 'Unified Blog+Tweet'),
            ('tweet_only', 'Tweet-Only'),
            ('neural_network', 'Neural Network'),
            ('distilbert', 'DistilBERT'),
        ]:
            if approach_name not in averaged_results:
                continue
                
            for model_name, metrics in averaged_results[approach_name].items():
                row = {
                    'Approach': approach_label,
                    'Model': model_name.replace('_', ' ').title()
                }
                for metric_name, stats in metrics.items():
                    row[f"{metric_name}_mean"] = f"{stats['mean']*100:.2f}"
                    row[f"{metric_name}_std"] = f"{stats['std']*100:.2f}"
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('accuracy_mean', ascending=False)
            df.to_csv(comparison_file, index=False)
            print(f"[OK] Saved model comparison to: {comparison_file}")


def select_authors_for_training(config):
    """Run author selection using the select_data.py"""
    print("\n" + "="*80)
    print("STEP 1: SELECTING AUTHORS AND PREPARING DATA")
    print("="*80)
    
    from utils.select_data import select_random_authors
    
    select_random_authors(
        tweet_file=config['source_tweet_file'],
        blog_file=config['source_blog_file'],
        num_authors=config['num_authors'],
        tweet_word_limit=config['tweet_word_limit'],
        blog_word_limit=config['blog_word_limit'],
        output_dir=config['selected_data_dir'],
        seed=config.get('selection_seed')
    )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Select authors, train models multiple times, and compute averaged results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    # Data selection arguments
    parser.add_argument('--config', type=str, default=None,
                       help="Path to a JSON experiment config file")
    parser.add_argument('--num-authors', type=int, default=None,
                       help=f"Number of authors to select (default: {CONFIG['num_authors']})")
    parser.add_argument('--tweet-word-limit', type=int, default=None,
                       help=f"Max words for tweets per author (default: {CONFIG['tweet_word_limit']})")
    parser.add_argument('--blog-word-limit', type=int, default=None,
                       help=f"Max words for blogs per author (default: {CONFIG['blog_word_limit']})")
    parser.add_argument('--source-tweet-file', type=str, default=None,
                       help="Path to source tweet CSV file")
    parser.add_argument('--source-blog-file', type=str, default=None,
                       help="Path to source blog CSV file")
    parser.add_argument('--selected-data-dir', type=str, default=None,
                       help="Directory for selected data")
    
    # Training arguments
    parser.add_argument('--num-runs', type=int, default=None,
                       help=f"Number of training runs (default: {CONFIG['num_runs']})")
    parser.add_argument('--base-seed', type=int, default=None,
                       help=f"Base training seed (default: {CONFIG['base_seed']})")
    parser.add_argument('--selection-seed', type=int, default=None,
                       help="Seed for author selection. Omit or set null in config for random selection.")
    parser.add_argument('--unified-model-type', type=str, default=None,
                       choices=['svm', 'logistic_regression', 'xgboost', 'bagging', 'voting', 'all'],
                       help=f"Model type for unified approach (default: {CONFIG['unified_model_type']})")
    parser.add_argument('--tweet-only-model-type', type=str, default=None,
                       choices=['svm', 'logistic_regression', 'xgboost', 'bagging', 'voting', 'all'],
                       help=f"Model type for tweet-only approach (default: {CONFIG['tweet_only_model_type']})")
    parser.add_argument('--neural-network-model-type', type=str, default=None,
                       choices=['mlp', 'bilstm', 'all'],
                       help=f"Model type for neural network approach (default: {CONFIG['neural_network_model_type']})")
    parser.add_argument('--test-size', type=float, default=None,
                       help=f"Test set size (default: {CONFIG['test_size']})")
    parser.add_argument('--use-semantic-clustering', action='store_true', default=None,
                       help="Enable semantic clustering for tweet grouping in unified model")
    parser.add_argument('--no-semantic-clustering', dest='use_semantic_clustering', action='store_false',
                       help="Disable semantic clustering for unified model")
    parser.add_argument('--semantic-cluster-k', type=int, default=None,
                       help=f"Number of semantic clusters for unified model (default: {CONFIG['semantic_cluster_k']})")
    parser.add_argument('--semantic-model-name', type=str, default=None,
                       help="SentenceTransformer model name used for semantic clustering")
    parser.add_argument('--use-domain-weighting', action='store_true', default=None,
                       help="Enable domain weighting (tweet vs blog) in unified model")
    parser.add_argument('--no-domain-weighting', dest='use_domain_weighting', action='store_false',
                       help="Disable domain weighting in unified model")
    parser.add_argument('--tweet-weight', type=float, default=None,
                       help=f"Training weight for tweet samples (default: {CONFIG['tweet_weight']})")
    parser.add_argument('--blog-weight', type=float, default=None,
                       help=f"Training weight for blog samples (default: {CONFIG['blog_weight']})")
    
    # Output arguments
    parser.add_argument('--results-dir', type=str, default=None,
                       help="Directory for results")
    
    # Comparison control arguments
    parser.add_argument('--compare-unified', action='store_true', default=None,
                       help="Compare unified blog+tweet model")
    parser.add_argument('--compare-tweet-only', action='store_true', default=None,
                       help="Compare tweet-only model")
    parser.add_argument('--compare-neural-network', action='store_true', default=None,
                       help="Compare neural network model")
    parser.add_argument('--compare-distilbert', action='store_true', default=None,
                       help="Compare DistilBERT model")
    parser.add_argument('--no-unified', dest='compare_unified', action='store_false',
                       help="Skip unified model comparison")
    parser.add_argument('--no-tweet-only', dest='compare_tweet_only', action='store_false',
                       help="Skip tweet-only model comparison")
    parser.add_argument('--no-neural-network', dest='compare_neural_network', action='store_false',
                       help="Skip neural network model comparison")
    parser.add_argument('--no-distilbert', dest='compare_distilbert', action='store_false',
                       help="Skip DistilBERT model comparison")
    
    # Control arguments
    parser.add_argument('--skip-selection', action='store_true', default=None,
                       help="Skip data selection (use existing selected data)")
    
    args = parser.parse_args()
    
    run_config = load_experiment_config(CONFIG, args.config)
    run_config = apply_cli_overrides(run_config, args, ignored_keys={'config'})
    
    print("\n" + "="*80)
    print("MULTI-RUN TRAINING WITH MODEL COMPARISON")
    print("="*80)
    print(f"\nConfiguration Summary:")
    print(f"  Authors: {run_config['num_authors']}")
    print(f"  Training runs: {run_config['num_runs']}")
    print(f"  Unified model type: {run_config['unified_model_type']}")
    print(f"  Tweet-only model type: {run_config['tweet_only_model_type']}")
    print(f"  Neural network model type: {run_config['neural_network_model_type']}")
    print(f"  Test size: {run_config['test_size']}")
    print(f"  Semantic clustering: {run_config['use_semantic_clustering']} (k={run_config['semantic_cluster_k']})")
    print(f"  Domain weighting: {run_config['use_domain_weighting']} (tweet={run_config['tweet_weight']}, blog={run_config['blog_weight']})")
    print(f"  Base seed: {run_config['base_seed']}")
    print(f"  Data selection seed: {run_config.get('selection_seed', None) if run_config.get('selection_seed', None) is not None else 'Random'}")
    print(f"\nApproaches enabled:")
    print(f"  Unified Blog+Tweet: {run_config['compare_unified']}")
    print(f"  Tweet-Only: {run_config['compare_tweet_only']}")
    print(f"  Neural Network: {run_config['compare_neural_network']}")
    print(f"  DistilBERT: {run_config['compare_distilbert']}")
    print("="*80)
    
    try:
        # Step 1: Select authors (unless skipped)
        if not run_config.get('skip_selection', False):
            select_authors_for_training(run_config)
        else:
            print("\n[SKIPPED] Data selection (using existing data)")
        
        # Step 2: Multiple training runs
        print("\n" + "="*80)
        print("STEP 2: MULTIPLE TRAINING RUNS")
        print("="*80)
        
        trainer = MultiRunTrainer(run_config)
        trainer.run_all_training()
        
        # Step 3: Compute and save averaged results
        print("\n" + "="*80)
        print("STEP 3: COMPUTING AVERAGED RESULTS")
        print("="*80)
        
        averaged_results = trainer.compute_averaged_results()
        trainer.save_results(averaged_results)
        
        print("\n" + "="*80)
        print("[OK] ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {run_config['results_dir']}/")
        print(f"  - Summary: {CONFIG['averaged_results_file']}")
        print(f"  - Details: {CONFIG['detailed_results_file']}")
        print(f"  - Comparison: {CONFIG['comparison_summary_file']}")
        
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        sys.exit(1)


if __name__ == '__main__':
    main()
